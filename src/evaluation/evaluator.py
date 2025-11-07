"""
Evaluation utilities for the Murder Model project.
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path('data/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation on all datasets.
        
        Args:
            model: Trained model
            X_train, X_val, X_test: Feature datasets
            y_train, y_val, y_test: Target datasets
            
        Returns:
            Dictionary with metrics for each dataset
        """
        results = {}
        
        for name, X, y in [('train', X_train, y_train),
                          ('validation', X_val, y_val),
                          ('test', X_test, y_test)]:
            if X is None or y is None:
                continue
                
            y_pred = model.predict(X)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }
            
            # Add ROC AUC if predict_proba available
            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
            except:
                pass
                
            results[name] = metrics
            
        return results
    
    def print_evaluation_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "="*80)
        print("MODEL EVALUATION REPORT")
        print("="*80)
        
        for dataset_name, metrics in results.items():
            print(f"\n{dataset_name.upper()} SET:")
            print("-" * 40)
            for metric_name, value in metrics.items():
                print(f"  {metric_name:15s}: {value:.4f}")
    
    def save_evaluation_report(self, results: Dict[str, Dict[str, float]], 
                               model_name: str, timestamp: Optional[str] = None) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation metrics
            model_name: Name of the model
            timestamp: Optional timestamp string
            
        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = self.output_dir / f"{model_name}_evaluation_{timestamp}.json"
        
        report = {
            'model_name': model_name,
            'timestamp': timestamp,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation report saved to: {output_file}")
        return str(output_file)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, 
                             title='Confusion Matrix', save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
            save_path: Optional path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels)
            plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, title='ROC Curve', 
                       save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            title: Plot title
            save_path: Optional path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance: np.ndarray, feature_names: List[str],
                               title='Feature Importance', top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            importance: Feature importance scores
            feature_names: Names of features
            title: Plot title
            top_n: Number of top features to show
            save_path: Optional path to save figure
        """
        # Sort by absolute importance
        indices = np.argsort(np.abs(importance))[::-1][:top_n]
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        colors = ['red' if importance[i] < 0 else 'green' for i in indices]
        
        plt.barh(range(len(indices)), importance[indices], color=colors, alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance (Coefficient)')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\nDetailed Classification Report:")
        print("="*60)
        print(classification_report(y_true, y_pred))


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