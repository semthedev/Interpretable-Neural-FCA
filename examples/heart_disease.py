"""Train an interpretable Neural FCA model on the UCI Heart Disease dataset.

Usage:
    python examples/heart_disease.py
"""

import random

import numpy as np
import torch
from sklearn.metrics import f1_score

from neural_fca import ConceptNetwork, build_lattice, select_best_concepts
from neural_fca.datasets import train_test_binary

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Data: binarized features, top-10 attributes by correlation with the target
X_train, X_test, y_train, y_test = train_test_binary(n_attributes=10)

# 2. Concept lattice on a subset of the training data (keeps the lattice small)
X_fca = X_train.sample(n=min(60, len(X_train)), random_state=SEED)
y_fca = y_train.loc[X_fca.index]
lattice = build_lattice(X_fca)
best = select_best_concepts(lattice, y_fca, n_concepts=15)
print(f"Lattice size: {len(lattice)}, hidden concepts: {len(best)}")

# 3. Build the network from the lattice and train it
model = ConceptNetwork.from_lattice(lattice, best, targets=tuple(sorted(set(y_train))))
model.fit(X_train, y_train, n_epochs=2000)

# 4. Evaluate
_, y_pred = model.predict(X_test)
print(f"Test F1: {f1_score(y_test, y_pred.numpy()):.4f}")

# 5. Explain one prediction: which concepts fire for the first test patient
sample = X_test.iloc[0]
active = model.trace_description(frozenset(sample[sample].index))
print("Active concept neurons for patient #0:", sorted(active))
