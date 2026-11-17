"""Dataset loading and binarization utilities."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

HEART_DISEASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def load_heart_disease() -> Tuple[pd.DataFrame, pd.Series]:
    """Download the UCI Heart Disease (Cleveland) dataset. Target: 0 = healthy, 1 = sick."""
    df = pd.read_csv(HEART_DISEASE_URL, names=HEART_COLUMNS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    y = (df.pop("target") > 0).astype(int)
    return df, y


def binarize(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical: List[str] = NUMERICAL,
    categorical: List[str] = CATEGORICAL,
    n_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quantile-bin numerical features and one-hot encode categorical ones into bool columns."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("bins", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    train = preprocessor.fit_transform(X_train)
    test = preprocessor.transform(X_test)
    cols = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(train, columns=cols, index=X_train.index).astype(bool),
        pd.DataFrame(test, columns=cols, index=X_test.index).astype(bool),
    )


def top_correlated(X: pd.DataFrame, y: pd.Series, k: int = 10) -> List[str]:
    """Return the k binary attributes with the highest |correlation| with the target."""
    y_arr = np.asarray(y)
    corrs = {}
    for col in X.columns:
        x = X[col].astype(int).to_numpy()
        corrs[col] = 0.0 if x.std() == 0 else abs(np.corrcoef(x, y_arr)[0, 1])
    return sorted(corrs, key=corrs.get, reverse=True)[:k]


def train_test_binary(test_size: float = 0.2, n_attributes: int = 10, random_state: int = 42):
    """Load, split and binarize the Heart Disease dataset, keeping the top correlated attributes."""
    X, y = load_heart_disease()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_tr, X_te = binarize(X_tr, X_te)
    attrs = top_correlated(X_tr, y_tr, k=n_attributes)
    return X_tr[attrs], X_te[attrs], y_tr, y_te
