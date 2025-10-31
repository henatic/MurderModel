"""
Base model class for the Murder Model project.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaseModel(ABC):
    """
    Abstract base class for all models in the project.
    """
    
    def __init__(self):
        self.model = None
        
    @abstractmethod
    def train(self, X, y):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y_true: True labels
            
        Returns:
            dict: Dictionary containing various performance metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }