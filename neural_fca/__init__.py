"""Interpretable Neural FCA: neural networks built from formal concept lattices."""

from .network import ConceptNetwork, DisjunctiveNeuron, neuron_label_func
from .concepts import build_lattice, select_best_concepts

__all__ = [
    "ConceptNetwork",
    "DisjunctiveNeuron",
    "neuron_label_func",
    "build_lattice",
    "select_best_concepts",
]

__version__ = "0.1.0"
