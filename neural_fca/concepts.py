"""Helpers for building a concept lattice and choosing concepts for the hidden layer."""

from typing import List

import numpy as np
import pandas as pd
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from sklearn.metrics import f1_score


def build_lattice(X: pd.DataFrame) -> ConceptLattice:
    """Build a monotone concept lattice from a binary (bool) feature table."""
    X_ctx = X.copy()
    X_ctx.index = X_ctx.index.astype(str)
    context = FormalContext.from_pandas(X_ctx)
    return ConceptLattice.from_context(context, is_monotone=True)


def select_best_concepts(lattice: ConceptLattice, y: pd.Series, n_concepts: int = 15) -> List[int]:
    """Rank concepts by F1 of the rule "object in extent -> positive class" and return the top indices."""
    y_arr = np.asarray(y)
    scores = np.zeros(len(lattice))
    for i, concept in enumerate(lattice):
        y_pred = np.zeros_like(y_arr)
        y_pred[list(concept.extent_i)] = 1
        scores[i] = f1_score(y_arr, y_pred, zero_division=0)
        concept.measures["f1_score"] = scores[i]
    return scores.argsort()[::-1][:n_concepts].tolist()
