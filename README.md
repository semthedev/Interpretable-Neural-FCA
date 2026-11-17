# Interpretable-Neural-FCA

**Neural networks you can read.** Neural FCA builds the architecture of a neural network from a *formal concept lattice*: every hidden neuron is a human-readable concept ("thal = 3 AND cp = 4"), and every connection follows a subset relation between concepts. The result is a classifier whose decisions can be traced step by step, while staying competitive with black-box models.

Applied to heart disease diagnosis (UCI Cleveland), the optimized Neural FCA reaches **F1 = 0.847**, outperforming tuned XGBoost, SVM and Random Forest baselines.

## Highlights

- **Interpretable by design.** Each neuron corresponds to a formal concept, so a prediction can be explained by the set of concepts that fired.
- **Fast lattice traversal.** Concept neurons use bitwise intent representations instead of set operations.
- **Modern training loop.** Optional Batch Normalization and Dropout, L1 sparsity regularization, early stopping, and learning-rate decay.
- **Bagging ensemble.** Aggregates 20 lattices built on bootstrap samples to reduce sensitivity to the data subset.
- **Validation-based model selection.** Several lattices are built on different subsets and the best one is selected on a held-out validation set.

## Results

Heart Disease UCI (Cleveland), binary target (healthy / sick), test F1:

| Model | F1 | Notes |
|-------|----|-------|
| **Neural FCA (CV selection)** | **0.8467** | Low LR (1e-4), validation-based selection |
| Random Forest | 0.8364 | Tuned with 5-fold GridSearchCV |
| Neural FCA (L1 + early stopping) | 0.8318 | |
| Neural FCA (Bagging) | 0.8235 | 20 bootstrap lattices |
| SVM / XGBoost | 0.8148 | Tuned with 5-fold GridSearchCV |
| Neural FCA (base) | 0.7755 | Original architecture |
| Logistic Regression | 0.7308 | Linear baseline |

Cross-validated F1: 0.8053 ± 0.0459 for the improved Neural FCA and 0.82 ± 0.0605 for the bagging ensemble. The optimizations also make training about 30% faster.

## How it works

1. **Binarization.** Numerical features are split into 3 quantile bins; categorical features are one-hot encoded. The 10 binary attributes most correlated with the target are kept to avoid lattice explosion.
2. **Concept lattice.** A monotone concept lattice is built on a subset of the training data with [fcapy](https://github.com/EgorDudyrev/FCApy).
3. **Concept selection.** Concepts are ranked by F1 of the rule "object belongs to the concept → positive class"; the top 15 form the hidden layer.
4. **Network.** The selected concepts become a sparse network (Input → Concepts → Output) whose edges follow the lattice order, then it is trained with PyTorch.
5. **Explanation.** For any input, `trace_description` returns the concept neurons that are activated, which can be drawn on the lattice.

## Installation

```bash
git clone https://github.com/semthedev/Interpretable-Neural-FCA.git
cd Interpretable-Neural-FCA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> `sparselinear` depends on `torch_sparse`; if installation fails, install a `torch_sparse` wheel matching your PyTorch version first (see the [PyG installation guide](https://github.com/rusty1s/pytorch_sparse#installation)).

## Usage

Run the end-to-end example:

```bash
python examples/heart_disease.py
```

Or use the library directly:

```python
from neural_fca import ConceptNetwork, build_lattice, select_best_concepts
from neural_fca.datasets import train_test_binary

X_train, X_test, y_train, y_test = train_test_binary(n_attributes=10)

X_fca = X_train.sample(n=60, random_state=42)
lattice = build_lattice(X_fca)
best = select_best_concepts(lattice, y_train.loc[X_fca.index], n_concepts=15)

model = ConceptNetwork.from_lattice(lattice, best, targets=(0, 1))
model.fit(X_train, y_train, n_epochs=2000)

proba, y_pred = model.predict(X_test)

# Which concepts fired for the first patient?
sample = X_test.iloc[0]
print(model.trace_description(frozenset(sample[sample].index)))
```

All experiments, baselines and visualizations are in [`fca.ipynb`](fca.ipynb).

## Project structure

```
neural_fca/
  network.py      # ConceptNetwork: lattice -> sparse neural network, training, tracing
  concepts.py     # lattice construction and concept selection
  datasets.py     # Heart Disease loader, binarization, attribute selection
examples/
  heart_disease.py
fca.ipynb         # full experiments and comparison with baselines
```

## License

[MIT](LICENSE)
