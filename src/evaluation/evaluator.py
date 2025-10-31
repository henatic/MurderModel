"""
Evaluation utilities for the Murder Model project.
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation.
    
    Args:
        model: The model to evaluate
        X: Features
        y: Labels
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores
    }

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))