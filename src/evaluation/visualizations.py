"""Advanced model evaluation with learning curves and PR curves.

This script generates comprehensive visualizations for model performance analysis:
1. Learning curves (training vs. validation performance)
2. Precision-Recall curves
3. ROC curves with confidence intervals
4. Threshold tuning visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import sys

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, confusion_matrix
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_processor import DataProcessor


class ModelVisualizer:
    """Generate advanced visualizations for model evaluation."""

    def __init__(self, output_dir: str = None):
        """Initialize visualizer."""
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300

    def plot_learning_curves(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = 'Model',
        cv_folds: int = 5,
        train_sizes: np.ndarray = None
    ) -> str:
        """Plot learning curves showing training and validation scores."""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        print(f"\nGenerating learning curves for {model_name}...")
        
        # Calculate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot learning curves
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#2ecc71', 
                label='Training score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.2, color='#2ecc71')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='#e74c3c',
                label='Validation score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.2, color='#e74c3c')
        
        # Formatting
        ax.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
        
        # Save
        filename = f'learning_curve_{model_name.lower().replace(" ", "_")}_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved learning curve to {filepath}")
        return str(filepath)

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Model'
    ) -> str:
        """Plot Precision-Recall curve."""
        print(f"\nGenerating PR curve for {model_name}...")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, color='#3498db', linewidth=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Plot baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='#95a5a6', linestyle='--', linewidth=2,
                  label=f'Baseline (y={baseline:.3f})')
        
        # Formatting
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save
        filename = f'pr_curve_{model_name.lower().replace(" ", "_")}_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved PR curve to {filepath}")
        return str(filepath)

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'Model'
    ) -> str:
        """Plot precision, recall, and F1 score vs. classification threshold."""
        print(f"\nGenerating threshold analysis for {model_name}...")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        optimal_threshold = thresholds[optimal_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot metrics vs. threshold
        ax.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, recall[:-1], 'g-', label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score', linewidth=2)
        
        # Mark optimal threshold
        ax.axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2,
                  label=f'Optimal (F1={f1_scores[optimal_idx]:.3f}, t={optimal_threshold:.3f})')
        
        # Formatting
        ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Threshold Analysis - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save
        filename = f'threshold_analysis_{model_name.lower().replace(" ", "_")}_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved threshold analysis to {filepath}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"At optimal threshold: P={precision[optimal_idx]:.3f}, R={recall[optimal_idx]:.3f}, F1={f1_scores[optimal_idx]:.3f}")
        
        return str(filepath)

    def plot_roc_comparison(
        self,
        results: List[Tuple[str, np.ndarray, np.ndarray]],
        title: str = 'ROC Curve Comparison'
    ) -> str:
        """Plot multiple ROC curves for comparison.
        
        Args:
            results: List of (model_name, y_true, y_proba) tuples
        """
        print(f"\nGenerating ROC comparison plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (name, y_true, y_proba) in enumerate(results):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            color = colors[i % len(colors)]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save
        filename = f'roc_comparison_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC comparison to {filepath}")
        return str(filepath)

    def plot_confusion_matrices(
        self,
        results: List[Tuple[str, np.ndarray, np.ndarray]],
        threshold: float = 0.5
    ) -> str:
        """Plot confusion matrices for multiple models.
        
        Args:
            results: List of (model_name, y_true, y_proba) tuples
            threshold: Classification threshold
        """
        print(f"\nGenerating confusion matrices (threshold={threshold})...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, y_true, y_proba) in zip(axes, results):
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=True, square=True)
            
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_title(f'{name}\n(threshold={threshold})', fontweight='bold')
            ax.set_xticklabels(['Unsolved', 'Solved'])
            ax.set_yticklabels(['Unsolved', 'Solved'])
        
        # Save
        filename = f'confusion_matrices_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrices to {filepath}")
        return str(filepath)


def main():
    """Example usage of visualizer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # Load and prepare data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'data.csv'
    
    print("Loading data...")
    df = pd.read_csv(data_path, nrows=30000)
    
    y = df['Crime Solved'].map({'Yes': 1, 'No': 0})
    
    processor = DataProcessor(drop_leakage_features=True)
    X, _ = processor.fit_transform(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    # Generate visualizations
    visualizer.plot_learning_curves(lr, X, y, 'Logistic Regression', cv_folds=3)
    visualizer.plot_learning_curves(rf, X, y, 'Random Forest', cv_folds=3)
    
    visualizer.plot_precision_recall_curve(y_test, lr_proba, 'Logistic Regression')
    visualizer.plot_precision_recall_curve(y_test, rf_proba, 'Random Forest')
    
    visualizer.plot_threshold_analysis(y_test, lr_proba, 'Logistic Regression')
    visualizer.plot_threshold_analysis(y_test, rf_proba, 'Random Forest')
    
    visualizer.plot_roc_comparison([
        ('Logistic Regression', y_test, lr_proba),
        ('Random Forest', y_test, rf_proba)
    ])
    
    visualizer.plot_confusion_matrices([
        ('Logistic Regression', y_test, lr_proba),
        ('Random Forest', y_test, rf_proba)
    ], threshold=0.5)
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
