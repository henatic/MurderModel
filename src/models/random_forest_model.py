"""
Concrete Random Forest model implementing BaseModel.
"""
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier with optional scaling."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 random_state: Optional[int] = None,
                 scaler: bool = False,
                 n_jobs: int = -1,
                 class_weight: Optional[str] = None):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            scaler: Whether to apply StandardScaler preprocessing
            n_jobs: Number of parallel jobs (-1 = use all processors)
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """Train a random forest model using a pipeline."""
        steps = []
        if self.scaler:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('clf', RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )))

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        # Keep the wrapped estimator for save/load compatibility
        self.model = self.pipeline
        # Preserve feature names
        try:
            self.feature_names = X.columns.tolist()
        except Exception:
            self.feature_names = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""
        if self.pipeline is None:
            raise ValueError("Model is not trained")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.pipeline is None:
            raise ValueError("Model is not trained")
        return self.pipeline.predict_proba(X)

    def load_model(self, filepath: str) -> None:
        """Load model from disk and restore pipeline."""
        super().load_model(filepath)
        # Restore pipeline reference from model
        self.pipeline = self.model
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance values from the random forest
        """
        if self.pipeline is None:
            raise ValueError("Model is not trained")
        # Classifier is the last step
        clf = self.pipeline.named_steps['clf']
        return clf.feature_importances_

