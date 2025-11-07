"""
Base model class for the Murder Model project.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Tuple, Dict, Any, Optional
import joblib
import os

class BaseModel(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for all models in the project.
    """
    
    def __init__(self):
        """Initialize model."""
        self.model = None
        self.feature_names = None
    
    @staticmethod
    def split_data(X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = 0.2,
                   val_size: float = 0.2,
                   stratify: bool = True,
                   random_state: Optional[int] = None
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                             pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets with optional stratification.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for test set
            val_size (float): Proportion of data to use for validation set
            stratify (bool): Whether to perform stratified split
            random_state (int, optional): Random state for reproducibility
            
        Returns:
            Tuple containing:
                X_train (pd.DataFrame): Training features
                X_val (pd.DataFrame): Validation features
                X_test (pd.DataFrame): Test features
                y_train (pd.Series): Training targets
                y_val (pd.Series): Validation targets
                y_test (pd.Series): Test targets
        """
        # Determine stratification
        stratify_y = None
        if stratify:
            # Check if stratification is feasible
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            # Need at least 2 samples per class for stratification
            if min_class_count >= 2:
                stratify_y = y
            else:
                import warnings
                warnings.warn(
                    f"Stratification requested but minimum class has only {min_class_count} samples. "
                    "Falling back to non-stratified split."
                )
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_y,
            random_state=random_state
        )
        
        # Second split: separate validation set from remaining data
        # Adjust validation size to account for reduced dataset size
        adjusted_val_size = val_size / (1 - test_size)
        
        # Re-check stratification for second split
        stratify_y_temp = None
        if stratify and stratify_y is not None:
            class_counts_temp = y_temp.value_counts()
            if class_counts_temp.min() >= 2:
                stratify_y_temp = y_temp
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_size,
            stratify=stratify_y_temp,
            random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            
        Returns:
            self: Trained model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features to predict probabilities for
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X (pd.DataFrame): Input features
            y_true (pd.Series): True labels
            
        Returns:
            Dict[str, float]: Dictionary containing performance metrics
        """
        y_pred = self.predict(X)
        
        # Calculate basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Add ROC AUC if predict_proba is implemented
        try:
            y_prob = self.predict_proba(X)[:, 1]  # Probability of positive class
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            pass
            
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath (str): Path to save model to
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = joblib.load(filepath)
        
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            np.ndarray: Feature importance scores
        """
        pass