"""Extract and compare feature importance from saved models."""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class FeatureImportanceExtractor:
    """Extract and visualize feature importance from saved models."""

    def __init__(self):
        """Initialize extractor."""
        self.output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300

    def load_latest_models(self):
        """Load the most recent trained models."""
        # Find latest model files
        logistic_models = sorted(self.output_dir.glob('logistic_model_*.pkl'))
        rf_models = sorted(self.output_dir.glob('randomforest_model_*.pkl'))
        
        if not logistic_models or not rf_models:
            print("Error: Model files not found!")
            return None, None
        
        # Load models
        with open(logistic_models[-1], 'rb') as f:
            logistic_model = pickle.load(f)
        
        with open(rf_models[-1], 'rb') as f:
            rf_model = pickle.load(f)
        
        print(f"Loaded Logistic Regression model: {logistic_models[-1].name}")
        print(f"Loaded Random Forest model: {rf_models[-1].name}")
        
        return logistic_model, rf_model

    def get_feature_names(self):
        """Get feature names from data processor."""
        from preprocessing.data_processor import DataProcessor
        
        processor = DataProcessor()
        df = processor.load_data()
        X, y = processor.prepare_features(df)
        
        return X.columns.tolist()

    def extract_feature_importance(self, logistic_model, rf_model):
        """Extract feature importance from both models."""
        feature_names = self.get_feature_names()
        
        # Logistic Regression coefficients
        lr_coef = logistic_model.model.coef_[0]
        lr_importance = dict(zip(feature_names, lr_coef))
        
        # Random Forest feature importance
        rf_importance_values = rf_model.model.feature_importances_
        rf_importance = dict(zip(feature_names, rf_importance_values))
        
        return lr_importance, rf_importance

    def plot_comparison(self, lr_importance, rf_importance, top_n=15):
        """Plot feature importance comparison."""
        # Get top N features from each model
        lr_sorted = sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        rf_sorted = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance Comparison\n(Top 15 Features per Model)', 
                     fontsize=14, fontweight='bold')
        
        # Logistic Regression (coefficients)
        features_lr, importances_lr = zip(*lr_sorted)
        y_pos_lr = np.arange(len(features_lr))
        colors_lr = ['#d62728' if x < 0 else '#2ca02c' for x in importances_lr]
        
        ax1.barh(y_pos_lr, importances_lr, color=colors_lr, alpha=0.7)
        ax1.set_yticks(y_pos_lr)
        ax1.set_yticklabels(features_lr, fontsize=9)
        ax1.set_xlabel('Coefficient Value', fontsize=11)
        ax1.set_title('Logistic Regression\n(Red = Decreases Solved Prob., Green = Increases)', 
                     fontsize=11)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Add coefficient values as text
        for i, (feat, val) in enumerate(lr_sorted):
            ax1.text(val, i, f' {val:.3f}', va='center', 
                    ha='left' if val > 0 else 'right', fontsize=8)
        
        # Random Forest (feature importance)
        features_rf, importances_rf = zip(*rf_sorted)
        y_pos_rf = np.arange(len(features_rf))
        
        ax2.barh(y_pos_rf, importances_rf, color='#ff7f0e', alpha=0.7)
        ax2.set_yticks(y_pos_rf)
        ax2.set_yticklabels(features_rf, fontsize=9)
        ax2.set_xlabel('Importance Score (Gini Impurity)', fontsize=11)
        ax2.set_title('Random Forest\n(Higher = More Important for Splits)', fontsize=11)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add importance values as text
        for i, (feat, val) in enumerate(rf_sorted):
            ax2.text(val, i, f' {val:.4f}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / f'feature_importance_comparison_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nSaved feature importance comparison to {output_path}")
        plt.close()

    def plot_common_features(self, lr_importance, rf_importance, top_n=10):
        """Plot features that are important in both models."""
        # Get top features from both
        lr_top = set([feat for feat, _ in 
                     sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]])
        rf_top = set([feat for feat, _ in 
                     sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:20]])
        
        # Find common features
        common = lr_top & rf_top
        
        if not common:
            print("No common features in top 20 of both models")
            return
        
        # Get values for common features
        common_lr = {feat: lr_importance[feat] for feat in common}
        common_rf = {feat: rf_importance[feat] for feat in common}
        
        # Sort by RF importance
        sorted_features = sorted(common, key=lambda x: common_rf[x], reverse=True)[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(sorted_features))
        width = 0.35
        
        lr_values = [common_lr[f] for f in sorted_features]
        rf_values = [common_rf[f] for f in sorted_features]
        
        # Normalize RF values to similar scale as LR for visualization
        rf_max = max(rf_values)
        lr_max = max([abs(v) for v in lr_values])
        scale_factor = lr_max / rf_max if rf_max > 0 else 1
        rf_scaled = [v * scale_factor for v in rf_values]
        
        ax.bar(x - width/2, lr_values, width, label='Logistic Regression (Coefficient)', 
               color='steelblue', alpha=0.8)
        ax.bar(x + width/2, rf_scaled, width, 
               label=f'Random Forest (Importance × {scale_factor:.2f})', 
               color='darkorange', alpha=0.8)
        
        ax.set_ylabel('Importance Value')
        ax.set_xlabel('Feature')
        ax.set_title(f'Common Important Features Across Both Models\n(Top {len(sorted_features)} features)', 
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'common_features_{self.timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved common features plot to {output_path}")
        plt.close()

    def create_importance_table(self, lr_importance, rf_importance):
        """Create a table of feature importances."""
        # Combine into DataFrame
        df = pd.DataFrame({
            'Feature': list(lr_importance.keys()),
            'LR_Coefficient': list(lr_importance.values()),
            'RF_Importance': [rf_importance[f] for f in lr_importance.keys()]
        })
        
        # Add absolute LR coefficient for ranking
        df['LR_Abs'] = df['LR_Coefficient'].abs()
        
        # Sort by RF importance
        df = df.sort_values('RF_Importance', ascending=False)
        
        # Save to CSV
        output_path = self.output_dir / f'feature_importance_table_{self.timestamp}.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved feature importance table to {output_path}")
        
        # Print top 20
        print("\n" + "="*80)
        print("TOP 20 FEATURES BY RANDOM FOREST IMPORTANCE")
        print("="*80)
        print(df[['Feature', 'LR_Coefficient', 'RF_Importance']].head(20).to_string(index=False))
        print("="*80 + "\n")


def main():
    """Extract and visualize feature importance."""
    print("="*80)
    print("Extracting Feature Importance from Saved Models")
    print("="*80 + "\n")
    
    extractor = FeatureImportanceExtractor()
    
    # Load models
    print("Loading models...")
    logistic_model, rf_model = extractor.load_latest_models()
    
    if logistic_model is None or rf_model is None:
        return
    
    # Extract importance
    print("\nExtracting feature importance...")
    lr_importance, rf_importance = extractor.extract_feature_importance(
        logistic_model, rf_model
    )
    
    print(f"Extracted importance for {len(lr_importance)} features")
    
    # Generate visualizations
    print("\n1. Plotting feature importance comparison...")
    extractor.plot_comparison(lr_importance, rf_importance)
    
    print("2. Plotting common important features...")
    extractor.plot_common_features(lr_importance, rf_importance)
    
    print("3. Creating feature importance table...")
    extractor.create_importance_table(lr_importance, rf_importance)
    
    print("\n" + "="*80)
    print("Feature Importance Extraction Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
