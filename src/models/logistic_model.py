"""
Concrete Logistic Regression model implementing BaseModel.
"""
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class LogisticModel(BaseModel):
    """Simple logistic regression model with optional scaling."""

    def __init__(self,
                 C: float = 1.0,
                 penalty: str = 'l2',
                 random_state: Optional[int] = None,
                 scaler: bool = True):
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.random_state = random_state
        self.scaler = scaler
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticModel':
        """Train a logistic regression model using a small pipeline."""
        steps = []
        if self.scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('clf', LogisticRegression(C=self.C,
                                                penalty=self.penalty,
                                                solver='lbfgs',
                                                max_iter=1000,
                                                random_state=self.random_state)))

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        # keep the wrapped estimator for save/load compatibility
        self.model = self.pipeline
        # preserve feature names
        try:
            self.feature_names = X.columns.tolist()
        except Exception:
            self.feature_names = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def load_model(self, filepath: str) -> None:
        """Load model from disk and restore pipeline."""
        super().load_model(filepath)
        # Restore pipeline reference from model
        self.pipeline = self.model
    
    def get_feature_importance(self) -> np.ndarray:
        # For logistic regression, coef_ is a good proxy
        if self.pipeline is None:
            raise ValueError("Model is not trained")
        # classifier is the last step
        clf = self.pipeline.named_steps['clf']
        coefs = clf.coef_.ravel()
        return coefs
