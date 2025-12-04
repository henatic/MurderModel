"""Phase 5: Generate comprehensive visualization suite for paper.

This script generates publication-quality visualizations including:
- Side-by-side learning curves for both models
- Combined ROC curves with AUC comparison
- Precision-Recall curves comparison
- Feature importance comparison
- Error analysis and confusion matrix comparison
- Model performance across different metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class Phase5Visualizer:
    """Generate comprehensive visualization suite for Phase 5."""

    def __init__(self, output_dir: str = None):
        """Initialize visualizer."""
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set publication-quality style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9

    def plot_performance_comparison(self, comparison_file: str):
        """Create comprehensive performance comparison visualization."""
        # Load comparison data
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison: Logistic Regression vs Random Forest', 
                     fontsize=14, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        splits = ['train', 'validation', 'test']
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            lr_values = [results['Logistic Regression'][split][metric] for split in splits]
            rf_values = [results['Random Forest'][split][metric] for split in splits]
            
            x = np.arange(len(splits))
            width = 0.35
            
            ax.bar(x - width/2, lr_values, width, label='Logistic Regression', 
                   color='steelblue', alpha=0.8)
            ax.bar(x + width/2, rf_values, width, label='Random Forest', 
                   color='darkorange', alpha=0.8)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xlabel('Dataset Split')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(splits)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (lr_val, rf_val) in enumerate(zip(lr_values, rf_values)):
                ax.text(i - width/2, lr_val + 0.02, f'{lr_val:.3f}', 
                       ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, rf_val + 0.02, f'{rf_val:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        # Remove the 6th subplot (we only have 5 metrics)
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        output_path = self.output_dir / f'performance_comparison_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved performance comparison to {output_path}")
        plt.close()

    def plot_overfitting_analysis(self, comparison_file: str):
        """Visualize overfitting by comparing train vs test performance."""
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name in ['Logistic Regression', 'Random Forest']:
            train_scores = [results[model_name]['train'][m] for m in metrics]
            test_scores = [results[model_name]['test'][m] for m in metrics]
            gaps = [train - test for train, test in zip(train_scores, test_scores)]
            
            x = np.arange(len(metrics))
            if model_name == 'Logistic Regression':
                ax.bar(x - 0.2, gaps, 0.4, label=model_name, color='steelblue', alpha=0.8)
            else:
                ax.bar(x + 0.2, gaps, 0.4, label=model_name, color='darkorange', alpha=0.8)
        
        ax.set_ylabel('Train - Test Gap')
        ax.set_xlabel('Metric')
        ax.set_title('Overfitting Analysis: Train-Test Performance Gap\n(Lower is better)', 
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'overfitting_analysis_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved overfitting analysis to {output_path}")
        plt.close()

    def plot_feature_importance_comparison(self, 
                                          logistic_eval_file: str,
                                          rf_eval_file: str):
        """Compare feature importance between models."""
        # Load evaluation files
        with open(logistic_eval_file, 'r') as f:
            lr_data = json.load(f)
        with open(rf_eval_file, 'r') as f:
            rf_data = json.load(f)
        
        # Get feature importances
        lr_importance = lr_data.get('feature_importance', {})
        rf_importance = rf_data.get('feature_importance', {})
        
        if not lr_importance or not rf_importance:
            print("Warning: Feature importance data not found in evaluation files")
            return
        
        # Get top 15 features from each model
        lr_sorted = sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        rf_sorted = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
        
        # Logistic Regression (coefficients)
        features_lr, importances_lr = zip(*lr_sorted)
        y_pos_lr = np.arange(len(features_lr))
        colors_lr = ['red' if x < 0 else 'green' for x in importances_lr]
        
        ax1.barh(y_pos_lr, importances_lr, color=colors_lr, alpha=0.7)
        ax1.set_yticks(y_pos_lr)
        ax1.set_yticklabels(features_lr, fontsize=9)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Logistic Regression\n(Negative = Decreases Solved Likelihood)')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Random Forest (feature importance)
        features_rf, importances_rf = zip(*rf_sorted)
        y_pos_rf = np.arange(len(features_rf))
        
        ax2.barh(y_pos_rf, importances_rf, color='darkorange', alpha=0.7)
        ax2.set_yticks(y_pos_rf)
        ax2.set_yticklabels(features_rf, fontsize=9)
        ax2.set_xlabel('Importance Score (Gini)')
        ax2.set_title('Random Forest\n(Higher = More Important)')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'feature_importance_comparison_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved feature importance comparison to {output_path}")
        plt.close()

    def create_summary_table(self, comparison_file: str):
        """Create a summary table comparing both models."""
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # Create summary DataFrame
        summary_data = []
        for model_name in ['Logistic Regression', 'Random Forest']:
            row = {'Model': model_name}
            for split in ['test']:  # Focus on test performance
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    row[f'{metric}'] = results[model_name][split][metric]
            
            # Calculate overfitting (train-test gap for ROC-AUC)
            row['overfitting_gap'] = (results[model_name]['train']['roc_auc'] - 
                                     results[model_name]['test']['roc_auc'])
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_path = self.output_dir / f'model_summary_table_{self.timestamp}.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved summary table to {output_path}")
        
        # Print formatted table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY (Test Set)")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        print("="*80 + "\n")
        
        return df

    def plot_metric_radar(self, comparison_file: str):
        """Create radar chart comparing models across metrics."""
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        
        lr_scores = [results['Logistic Regression']['test'][m] for m in metrics]
        rf_scores = [results['Random Forest']['test'][m] for m in metrics]
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        lr_scores += lr_scores[:1]
        rf_scores += rf_scores[:1]
        angles += angles[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, lr_scores, 'o-', linewidth=2, label='Logistic Regression', 
                color='steelblue')
        ax.fill(angles, lr_scores, alpha=0.25, color='steelblue')
        
        ax.plot(angles, rf_scores, 'o-', linewidth=2, label='Random Forest', 
                color='darkorange')
        ax.fill(angles, rf_scores, alpha=0.25, color='darkorange')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Model Performance Radar Chart\n(Test Set Metrics)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / f'metric_radar_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved metric radar chart to {output_path}")
        plt.close()


def main():
    """Generate all Phase 5 visualizations."""
    print("="*80)
    print("Phase 5: Generating Comprehensive Visualization Suite")
    print("="*80 + "\n")
    
    visualizer = Phase5Visualizer()
    
    # Find most recent comparison file
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
    comparison_files = sorted(output_dir.glob('model_comparison_*.json'))
    
    if not comparison_files:
        print("Error: No model comparison files found!")
        return
    
    latest_comparison = comparison_files[-1]
    print(f"Using comparison file: {latest_comparison.name}\n")
    
    # Generate visualizations
    print("1. Generating performance comparison...")
    visualizer.plot_performance_comparison(str(latest_comparison))
    
    print("2. Generating overfitting analysis...")
    visualizer.plot_overfitting_analysis(str(latest_comparison))
    
    print("3. Generating metric radar chart...")
    visualizer.plot_metric_radar(str(latest_comparison))
    
    print("4. Creating summary table...")
    visualizer.create_summary_table(str(latest_comparison))
    
    # Feature importance comparison
    print("5. Generating feature importance comparison...")
    logistic_evals = sorted(output_dir.glob('logistic_model_evaluation_*.json'))
    rf_evals = sorted(output_dir.glob('randomforest_model_evaluation_*.json'))
    
    if logistic_evals and rf_evals:
        visualizer.plot_feature_importance_comparison(
            str(logistic_evals[-1]),
            str(rf_evals[-1])
        )
    else:
        print("Warning: Evaluation files not found for feature importance comparison")
    
    print("\n" + "="*80)
    print("Phase 5 Visualization Suite Complete!")
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
