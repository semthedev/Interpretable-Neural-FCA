from dataclasses import dataclass
from typing import List, Tuple, FrozenSet, Set, Dict, ClassVar, Optional
import pandas as pd
import matplotlib.pyplot as plt

from fcapy.lattice import ConceptLattice
from fcapy.lattice.formal_concept import FormalConcept
from fcapy.poset import POSet
from fcapy.visualizer.line_layouts import calc_levels

import torch
from sparselinear import SparseLinear

from sklearn.utils import shuffle
import random


@dataclass(eq=False)
class DisjunctiveNeuron:
    """
    Узел решётки/сети.
    Хранит и исходный intent (множество строк), и его битовую маску для быстрых операций.
    """
    intent: FrozenSet[str]
    level: int
    is_target: bool = False  # помечаем целевые нейроны

    # Общий (для всех нейронов) словарь "строка -> индекс бита"
    attr2idx: ClassVar[Dict[str, int]] = {}

    def __post_init__(self):
        # Предварительно считаем битовую маску intent один раз
        b = 0
        mapping = self.attr2idx
        for a in self.intent:
            idx = mapping.get(a)
            if idx is not None:
                b |= 1 << idx
        self._intent_bits = b

    @property
    def intent_bits(self) -> int:
        return self._intent_bits

    def __eq__(self, other: 'DisjunctiveNeuron'):
        # Равенство по битовой маске intent и уровню
        return (
            isinstance(other, DisjunctiveNeuron)
            and self.intent_bits == other.intent_bits
            and self.level == other.level
            and self.is_target == other.is_target
        )

    def __lt__(self, other: 'DisjunctiveNeuron'):
        # Порядок как раньше, но через битовые маски:
        # self < other  <=>  other.intent ⊆ self.intent и self.level > other.level
        return (
            (self.intent_bits & other.intent_bits) == other.intent_bits
            and self.level > other.level
        )

    def __le__(self, other: 'DisjunctiveNeuron'):
        return self < other or self == other

    def __hash__(self):
        return hash((self.intent_bits, self.level, self.is_target))


class ConceptNetwork:
    def __init__(
            self,
            poset: Optional[POSet],
            network: Optional[torch.nn.Sequential] = None,
            attributes: Tuple[str, ...] = None,
            targets: Tuple[str, ...] = None,
            attr2idx: Optional[Dict[str, int]] = None,
    ):
        self._poset = poset
        self._network = network
        self._attributes = attributes
        self._targets = targets

        # Отображение "атрибут/метка -> индекс бита" для описаний
        self._attr2idx: Dict[str, int] = attr2idx or {}

        # Кэши для ускорения
        self._nodes_per_levels: Optional[List[List[int]]] = None
        self._descendants_cache: Dict[int, FrozenSet[int]] = {}
        self._bottoms_set: Set[int] = set()

        if self._poset is not None:
            self._build_caches()

    # ---------- Свойства ----------

    @property
    def poset(self) -> POSet:
        return self._poset

    @property
    def network(self) -> torch.nn.Sequential:
        return self._network

    @property
    def attributes(self) -> Tuple[str, ...]:
        return self._attributes

    @property
    def targets(self) -> Tuple[str, ...]:
        return self._targets

    # ---------- Внутренние кэши ----------

    def _build_caches(self):
        """
        Кэшируем:
        - узлы по уровням
        - множества потомков для верхних узлов (tops)
        - множество bottom-узлов
        """
        P = self.poset
        if P is None or len(P) == 0:
            return

        # Узлы по уровням
        max_level = P[P.bottoms[0]].level if P.bottoms else 0
        nodes_per_levels: List[List[int]] = [[] for _ in range(max_level + 1)]
        for node_i, node in enumerate(P):
            nodes_per_levels[node.level].append(node_i)
        self._nodes_per_levels = nodes_per_levels

        # bottom-узлы
        self._bottoms_set = set(P.bottoms)

        # потомки только для tops — это используется в trace_description
        self._descendants_cache = {}
        for node in P.tops:
            self._descendants_cache[node] = frozenset(P.descendants(node))

    def _ensure_caches(self):
        if self._nodes_per_levels is None or not self._descendants_cache or not self._bottoms_set:
            self._build_caches()

    # ---------- Помощники для битовых масок ----------

    def _description_to_bits(self, description: FrozenSet[str]) -> int:
        """
        Перевод описания (набор атрибутов) в битовую маску.
        """
        if not self._attr2idx and self._attributes is not None:
            # на всякий случай восстановим маппинг, если его нет
            self._attr2idx = {a: i for i, a in enumerate(self._attributes)}

        b = 0
        mapping = self._attr2idx
        for a in description:
            idx = mapping.get(a)
            if idx is not None:
                b |= 1 << idx
        return b

    # ---------- Поиск активированных концептов ----------

    def trace_description(self, description: FrozenSet[str], include_targets: bool = False) -> Set[int]:
        """
        Быстрый поиск активированных узлов по описанию.
        Использует:
        - битовые маски intents
        - кэш потомков для верхних узлов
        """
        P = self.poset
        self._ensure_caches()

        if not isinstance(description, frozenset):
            description = frozenset(description)

        desc_bits = self._description_to_bits(description)

        # Активируются только те атрибуты (tops), intent которых ⊆ description
        tops_activated = [
            node_i
            for node_i in P.tops
            if isinstance(P[node_i], DisjunctiveNeuron)
            and (P[node_i].intent_bits & desc_bits) == P[node_i].intent_bits
        ]

        activated_nodes: Set[int] = set(tops_activated)

        # Добавляем всех потомков активированных tops (из кэша)
        for node_i in tops_activated:
            activated_nodes |= self._descendants_cache.get(node_i, frozenset())

        if not include_targets:
            activated_nodes -= self._bottoms_set

        return activated_nodes

    # ---------- Конструирование сети из решётки ----------

    @classmethod
    def from_lattice(
            cls,
            lattice: ConceptLattice,
            best_concepts_indices: List[int],
            targets: Tuple[str, ...],
    ) -> 'ConceptNetwork':
        assert lattice.is_monotone, 'The lattice should be monotone'

        targets = tuple(targets)

        # Базовый набор атрибутов (intent нижнего элемента решётки)
        attrs_tpl = tuple(lattice[lattice.bottom].intent)

        # Строим общий словарь токенов (атрибуты + метки классов) -> индексы битов
        all_tokens: List[str] = list(attrs_tpl) + [f"y={y}" for y in targets]
        attr2idx: Dict[str, int] = {
            token: i for i, token in enumerate(all_tokens)}

        # Прокидываем маппинг в DisjunctiveNeuron,
        # чтобы __post_init__ мог сразу посчитать intent_bits
        DisjunctiveNeuron.attr2idx = attr2idx

        # Строим POS-решётку из «лучших» концептов
        P = cls._poset_from_best_concepts(
            lattice[best_concepts_indices], targets, attrs_tpl)
        P = cls._fill_levels(P)

        # Создаём сеть и сразу строим кэши
        return cls(P, None, attributes=attrs_tpl, targets=targets, attr2idx=attr2idx)

    @staticmethod
    def _poset_from_best_concepts(
            best_concepts: List[FormalConcept],
            targets: Tuple[str, ...],
            attrs_tpl: Tuple[str, ...],
    ) -> POSet:
        P_best = POSet(best_concepts)
        lvls = calc_levels(P_best)[0]
        lvls = [lvl + 1 for lvl in lvls]
        target_lvl = max(lvls) + 1

        attrs_set = set(attrs_tpl)

        # Концепты-нейроны внутренних уровней
        best_neurons = [
            DisjunctiveNeuron(frozenset(c.intent), lvl)
            for c, lvl in zip(P_best, lvls)
        ]

        # Первый уровень — отдельные атрибуты
        first_level_neurons = [
            DisjunctiveNeuron(frozenset({m}), 0)
            for m in attrs_tpl
        ]

        # Последний уровень — целевые классы
        last_level_neurons = [
            DisjunctiveNeuron(
                frozenset({f"y={y}"} | attrs_set), target_lvl, is_target=True)
            for y in targets
        ]

        return POSet(first_level_neurons + best_neurons + last_level_neurons)

    @staticmethod
    def _fill_levels(poset: POSet) -> POSet:
        """
        Заполняем промежуточные уровни копиями узлов (как в исходном коде),
        но переносим флаг is_target.
        """
        nodes_i = sorted(range(len(poset)),
                         key=lambda node_i: poset[node_i].level)
        for node_i in nodes_i:
            children_i = poset.children(node_i)
            if len(children_i) == 0:
                continue

            parent = poset[node_i]
            max_children_level = max(
                poset[child_i].level for child_i in children_i)
            for lvl in range(parent.level + 1, max_children_level):
                poset.add(
                    DisjunctiveNeuron(parent.intent, lvl,
                                      is_target=parent.is_target)
                )
        return poset

    # ---------- Преобразование POS-решётки в сеть ----------

    def _poset_to_network(
            self,
            poset: POSet,
            nonlinearity: type = torch.nn.ReLU,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
            dropout_rate: float = 0.5,
    ) -> 'torch.nn.Sequential':
        """
        Используем кэш уровней, избавляемся от дорогостоящих .index() в цикле.
        
        Args:
            poset: частично упорядоченное множество узлов
            nonlinearity: функция активации
            use_batch_norm: использовать ли Batch Normalization
            use_dropout: использовать ли Dropout
            dropout_rate: процент dropout (по умолчанию 0.5)
        """
        self._ensure_caches()
        nodes_per_levels = self._nodes_per_levels
        if not nodes_per_levels:
            raise ValueError(
                "nodes_per_levels cache is empty; poset may be invalid.")

        max_level = len(nodes_per_levels) - 1

        connectivities: List[List[Tuple[int, int]]] = []

        for layer_i in range(1, max_level + 1):
            layer_nodes = nodes_per_levels[layer_i]
            prev_nodes = nodes_per_levels[layer_i - 1]

            parent_idx_map = {node: idx for idx, node in enumerate(prev_nodes)}
            child_idx_map = {node: idx for idx, node in enumerate(layer_nodes)}

            layer_con: List[Tuple[int, int]] = []
            for child_node in layer_nodes:
                child_pos = child_idx_map[child_node]
                for parent_node in poset.parents(child_node):
                    parent_pos = parent_idx_map[parent_node]
                    layer_con.append((child_pos, parent_pos))

            # На случай если слой оказывается пустым (маловероятно, но на всякий случай)
            if not layer_con and prev_nodes and layer_nodes:
                for i in range(min(len(prev_nodes), len(layer_nodes))):
                    layer_con.append((i, i))

            connectivities.append(layer_con)

        # Строим линейные слои
        linear_layers = []
        for layer_i in range(max_level):
            con = torch.tensor(connectivities[layer_i]).T
            layer = SparseLinear(
                len(nodes_per_levels[layer_i]),
                len(nodes_per_levels[layer_i + 1]),
                connectivity=con,
            )
            linear_layers.append(layer)

        # Собираем финальный список слоёв
        layers = []
        for i, linear_layer in enumerate(linear_layers):
            layers.append(linear_layer)
            
            # Добавляем BatchNorm после линейного слоя (если включен)
            if use_batch_norm and i < len(linear_layers) - 1:  # не добавляем после последнего слоя
                layers.append(torch.nn.BatchNorm1d(len(nodes_per_levels[i + 1])))
            
            # Добавляем активацию (кроме последнего слоя)
            if i < len(linear_layers) - 1:
                layers.append(nonlinearity())
            
            # Добавляем Dropout после активации (если включен)
            if use_dropout and i < len(linear_layers) - 1:  # не добавляем после последнего слоя
                layers.append(torch.nn.Dropout(p=dropout_rate))
        
        # Добавляем Softmax в конце
        layers.append(torch.nn.Softmax(dim=1))
        
        model_sparse = torch.nn.Sequential(*layers)
        return model_sparse

    # ---------- Обучение и предсказание ----------

    def fit(
            self,
            X_df: 'pd.DataFrame[bool]',
            y: 'pd.Series[bool]',
            n_epochs: int = 2000,
            loss_fn=torch.nn.CrossEntropyLoss(),
            nonlinearity=torch.nn.Sigmoid,
            use_batch_norm: bool = False,
            use_dropout: bool = False,
            dropout_rate: float = 0.5,
            plot_loss: bool = False,        
    ):
        """
        Обучение сети.
        
        Args:
            X_df: датафрейм с признаками
            y: целевая переменная
            n_epochs: количество эпох обучения
            loss_fn: функция потерь
            nonlinearity: функция активации
            use_batch_norm: использовать ли Batch Normalization
            use_dropout: использовать ли Dropout
            dropout_rate: процент dropout (по умолчанию 0.5)
            plot_loss: построить график значения функции потерь по эпохам
        """
        # Преобразуем данные один раз
        X = torch.from_numpy(
            X_df[list(self.attributes)].values.astype('float32'))
        y_tensor = torch.from_numpy(y.values.astype('int64'))

        # Перемешиваем один раз (full-batch обучение)
        X_train_shuffle, y_train_shuffle = shuffle(
            X, y_tensor, random_state=random.randint(0, 1000)
        )

        # Строим сеть из POS-решётки
        self._network = self._poset_to_network(
            self.poset, 
            nonlinearity,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Переводим сеть в режим обучения
        self.network.train()
        
        loss_history = []  # для хранения значений функции потерь

        for _ in range(n_epochs):
            optimizer.zero_grad()
            y_pred = self.network(X_train_shuffle)
            loss = loss_fn(y_pred, y_train_shuffle)
            loss.backward()
            optimizer.step()

            if plot_loss:
                loss_history.append(loss.item())

        # Построение графика функции потерь
        if plot_loss and len(loss_history) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(loss_history) + 1), loss_history, label="Train loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def predict_proba(self, X_df: 'pd.DataFrame[bool]') -> torch.Tensor:
        """
        Предсказание вероятностей классов.
        """
        # Переводим сеть в режим inference (отключает dropout и batch norm в режиме оценки)
        self.network.eval()
        
        X = torch.from_numpy(
            X_df[list(self.attributes)].values.astype('float32'))
        
        with torch.no_grad():  # отключаем вычисление градиентов для inference
            return self.network(X)

    def predict(self, X_df: 'pd.DataFrame[bool]'):
        """
        Предсказание классов.
        """
        proba = self.predict_proba(X_df)
        return [proba, proba.argmax(1)]

    # ---------- Восстановление весов рёбер ----------

    def edge_weights_from_network(self) -> Dict[Tuple[int, int], float]:
        """
        Извлекает веса рёбер из обученной сети.
        Учитывает, что между линейными слоями могут быть BatchNorm и Dropout.
        """
        self._ensure_caches()
        nodes_per_levels = self._nodes_per_levels
        if not nodes_per_levels:
            raise ValueError(
                "nodes_per_levels cache is empty; poset may be invalid.")

        edge_weights: Dict[Tuple[int, int], float] = {}

        # Собираем только линейные слои из Sequential
        linear_layers = [layer for layer in self.network if isinstance(layer, SparseLinear)]

        for layer_i, (nodes, nn_layer) in enumerate(zip(nodes_per_levels[:-1], linear_layers)):
            next_nodes = nodes_per_levels[layer_i + 1]

            idxs = nn_layer.weight.indices().cpu().numpy().T.tolist()
            vals = nn_layer.weight.values().cpu().numpy()

            for (child_i, parent_i), v in zip(idxs, vals):
                edge_weights[(nodes[parent_i], next_nodes[child_i])] = float(v)
                
        return edge_weights


def neuron_label_func(el_i: int, P: POSet, M: set, only_new_attrs: bool = True) -> str:
    """
    Функция для подписи нейрона (оставил по сути без изменений, т.к. она обычно
    вызывается редко — при визуализации — и не критична для производительности).
    """
    el = P[el_i]

    if len(el.intent - M) > 0:  # если целевой узел
        attrs_to_show = list(el.intent - M)
    else:
        attrs_to_show = set(el.intent)
        if only_new_attrs:
            for parent_i in P.parents(el_i):
                attrs_to_show = attrs_to_show - P[parent_i].intent

        attrs_to_show = list(attrs_to_show)
    return ','.join(attrs_to_show)
