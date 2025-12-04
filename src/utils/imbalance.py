"""
Imbalance handling utilities: SMOTE/ADASYN wrappers for tabular data.
"""

from typing import Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_adasyn(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply ADASYN to balance classes."""
    ada = ADASYN(random_state=random_state)
    X_res, y_res = ada.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
