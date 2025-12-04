"""Advanced feature analysis for model improvement.

This script provides:
1. Feature correlation analysis (with target and between features)
2. Feature importance ranking from trained models
3. Redundancy detection
4. Feature selection recommendations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class FeatureAnalyzer:
    """Analyze features for correlation, importance, and redundancy."""

    def __init__(self, data_path: str = None, preprocessor=None):
        """Initialize analyzer."""
        self.data_path = Path(data_path) if data_path else None
        self.preprocessor = preprocessor
        self.df = None
        self.df_processed = None
        self.target = None
        self.feature_names = []
        self.results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'correlations': {},
            'feature_importance': {},
            'redundant_pairs': [],
            'recommendations': {}
        }

    def load_and_preprocess(self, sample_size: int = 50000) -> None:
        """Load and preprocess data."""
        if self.data_path:
            print(f"Loading data from {self.data_path}...")
            chunks = []
            chunk_size = 50000
            rows_read = 0
            
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunks.append(chunk)
                rows_read += len(chunk)
                if rows_read >= sample_size:
                    break
            
            self.df = pd.concat(chunks, ignore_index=True)
            if len(self.df) > sample_size:
                self.df = self.df.sample(n=sample_size, random_state=42)
            
            print(f"Loaded {len(self.df)} rows")
            
            # Extract target before preprocessing
            if 'Crime Solved' in self.df.columns:
                self.target = self.df['Crime Solved'].map({'Yes': 1, 'No': 0})
            
            # Preprocess if processor provided
            if self.preprocessor:
                self.df_processed, messages = self.preprocessor.fit_transform(self.df)
                self.feature_names = list(self.df_processed.columns)
                print(f"Preprocessed to {len(self.feature_names)} features")

    def analyze_target_correlation(self) -> Dict[str, float]:
        """Calculate correlation between each feature and target."""
        if self.df_processed is None or self.target is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        print("\nAnalyzing correlations with target...")
        correlations = {}
        
        for col in self.df_processed.columns:
            try:
                corr = self.df_processed[col].corr(self.target)
                if not pd.isna(corr):
                    correlations[col] = float(corr)
            except Exception as e:
                print(f"Could not calculate correlation for {col}: {e}")
        
        # Sort by absolute correlation
        self.results['correlations']['with_target'] = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return correlations

    def analyze_feature_correlations(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs (potential redundancy)."""
        if self.df_processed is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        print(f"\nFinding redundant features (correlation > {threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = self.df_processed.corr()
        
        # Find high correlations
        redundant_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    redundant_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        self.results['redundant_pairs'] = redundant_pairs
        self.results['correlations']['feature_matrix'] = corr_matrix.to_dict()
        
        print(f"Found {len(redundant_pairs)} highly correlated pairs")
        return redundant_pairs

    def rank_features_by_importance(self, model=None, model_name: str = "model") -> Dict[str, float]:
        """Extract feature importances from a trained model."""
        if model is None:
            print(f"\nNo {model_name} provided, skipping importance ranking")
            return {}
        
        print(f"\nRanking features by {model_name} importance...")
        
        importances = {}
        
        # Try to get feature importances
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            # For sklearn models
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = self.feature_names
            
            for name, importance in zip(feature_names, model.feature_importances_):
                importances[name] = float(importance)
        elif hasattr(model, 'coef_'):
            # For logistic regression - use absolute coefficients
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = self.feature_names
            
            coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            for name, coef in zip(feature_names, coefs):
                importances[name] = float(abs(coef))
        
        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        self.results['feature_importance'][model_name] = importances
        
        return importances

    def generate_recommendations(self, correlation_threshold: float = 0.15) -> Dict:
        """Generate feature selection recommendations."""
        print("\nGenerating feature selection recommendations...")
        
        recommendations = {
            'high_importance': [],
            'low_importance': [],
            'highly_correlated_with_target': [],
            'weakly_correlated_with_target': [],
            'redundant_to_drop': [],
            'keep_features': []
        }
        
        # Based on correlation with target
        target_corr = self.results['correlations'].get('with_target', {})
        for feature, corr in target_corr.items():
            if abs(corr) > correlation_threshold:
                recommendations['highly_correlated_with_target'].append({
                    'feature': feature,
                    'correlation': corr
                })
            else:
                recommendations['weakly_correlated_with_target'].append({
                    'feature': feature,
                    'correlation': corr
                })
        
        # Based on feature importance (if available)
        for model_name, importances in self.results['feature_importance'].items():
            if not importances:
                continue
            
            # Top 25% as high importance
            sorted_features = list(importances.items())
            top_n = max(1, len(sorted_features) // 4)
            
            for feature, importance in sorted_features[:top_n]:
                if feature not in [f['feature'] for f in recommendations['high_importance']]:
                    recommendations['high_importance'].append({
                        'feature': feature,
                        'importance': importance,
                        'model': model_name
                    })
        
        # Handle redundant pairs - keep one with higher importance/correlation
        for pair in self.results['redundant_pairs']:
            f1, f2 = pair['feature1'], pair['feature2']
            
            # Compare importance (if available)
            f1_imp = 0
            f2_imp = 0
            for importances in self.results['feature_importance'].values():
                f1_imp += importances.get(f1, 0)
                f2_imp += importances.get(f2, 0)
            
            # If no importance, use correlation with target
            if f1_imp == 0 and f2_imp == 0:
                f1_imp = abs(target_corr.get(f1, 0))
                f2_imp = abs(target_corr.get(f2, 0))
            
            # Drop the less important one
            to_drop = f1 if f1_imp < f2_imp else f2
            if to_drop not in [f['feature'] for f in recommendations['redundant_to_drop']]:
                recommendations['redundant_to_drop'].append({
                    'feature': to_drop,
                    'reason': f'Redundant with {f2 if to_drop == f1 else f1} (corr={pair["correlation"]:.3f})'
                })
        
        # Generate keep list
        drop_set = set([f['feature'] for f in recommendations['redundant_to_drop']] +
                      [f['feature'] for f in recommendations['weakly_correlated_with_target']])
        
        recommendations['keep_features'] = [
            f for f in self.feature_names if f not in drop_set
        ]
        
        self.results['recommendations'] = recommendations
        
        return recommendations

    def plot_correlation_heatmap(self, output_path: str = None, top_n: int = 20) -> None:
        """Plot correlation heatmap for top features."""
        if self.df_processed is None:
            return
        
        print("\nGenerating correlation heatmap...")
        
        # Get top features by correlation with target
        target_corr = self.results['correlations'].get('with_target', {})
        top_features = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        
        # Calculate correlation matrix for top features
        corr_matrix = self.df_processed[top_feature_names].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, linewidths=1)
        plt.title(f'Feature Correlation Heatmap (Top {top_n} Features)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {output_path}")
        else:
            plt.show()
        
        plt.close()

    def save_results(self, output_dir: str = None) -> str:
        """Save analysis results to JSON."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.results['timestamp']
        output_file = output_dir / f'feature_analysis_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nAnalysis results saved to: {output_file}")
        return str(output_file)

    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*80)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*80)
        
        # Correlation with target
        target_corr = self.results['correlations'].get('with_target', {})
        if target_corr:
            print(f"\nTop 10 features by correlation with target:")
            for i, (feature, corr) in enumerate(list(target_corr.items())[:10], 1):
                print(f"  {i}. {feature}: {corr:.4f}")
        
        # Feature importance
        for model_name, importances in self.results['feature_importance'].items():
            if importances:
                print(f"\nTop 10 features by {model_name} importance:")
                for i, (feature, imp) in enumerate(list(importances.items())[:10], 1):
                    print(f"  {i}. {feature}: {imp:.4f}")
        
        # Redundant pairs
        if self.results['redundant_pairs']:
            print(f"\nRedundant feature pairs: {len(self.results['redundant_pairs'])}")
            for pair in self.results['redundant_pairs'][:5]:
                print(f"  • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        # Recommendations
        recs = self.results.get('recommendations', {})
        if recs:
            print(f"\nRecommendations:")
            print(f"  • High importance features: {len(recs.get('high_importance', []))}")
            print(f"  • Highly correlated with target: {len(recs.get('highly_correlated_with_target', []))}")
            print(f"  • Weakly correlated with target: {len(recs.get('weakly_correlated_with_target', []))}")
            print(f"  • Redundant features to drop: {len(recs.get('redundant_to_drop', []))}")
            print(f"  • Recommended features to keep: {len(recs.get('keep_features', []))}")


def main():
    """Run feature analysis from command line."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from preprocessing.data_processor import DataProcessor
    
    # Initialize
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'data.csv'
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
    
    processor = DataProcessor(drop_leakage_features=True)
    analyzer = FeatureAnalyzer(str(data_path), processor)
    
    # Run analysis
    analyzer.load_and_preprocess(sample_size=50000)
    analyzer.analyze_target_correlation()
    analyzer.analyze_feature_correlations(threshold=0.8)
    analyzer.generate_recommendations(correlation_threshold=0.05)
    
    # Save and display
    analyzer.print_summary()
    analyzer.save_results()
    
    # Generate heatmap
    heatmap_path = output_dir / f'correlation_heatmap_{analyzer.results["timestamp"]}.png'
    analyzer.plot_correlation_heatmap(str(heatmap_path), top_n=15)


if __name__ == '__main__':
    main()
