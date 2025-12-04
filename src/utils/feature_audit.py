"""Feature audit utility for identifying potential data leakage.

This script analyzes all features in the dataset to identify:
1. Features that contain post-hoc information (known only after crime is solved)
2. Features with suspiciously high correlation to target
3. Features with minimal predictive value
4. Redundant features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path


class FeatureAuditor:
    """Analyze features for data leakage and relevance."""

    def __init__(self, data_path: str):
        """Initialize auditor with path to raw data."""
        self.data_path = Path(data_path)
        self.df = None
        self.audit_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_features': 0,
            'leakage_features': [],
            'high_correlation_features': [],
            'low_variance_features': [],
            'recommended_drops': [],
            'safe_features': [],
            'feature_info': {}
        }

    def load_data(self, sample_size: int = 100000) -> None:
        """Load and sample data for analysis."""
        print(f"Loading data from {self.data_path}...")
        # Read in chunks to handle large file
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
        
        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        self.audit_results['total_features'] = len(self.df.columns) - 1  # Exclude target

    def identify_leakage_features(self) -> List[str]:
        """
        Identify features that likely contain post-hoc information.
        
        Categories of leakage:
        1. Perpetrator information (only known after arrest/solving)
        2. Relationship (often only known after investigation)
        3. Any field that would only be available post-investigation
        """
        leakage_features = []
        
        # Known perpetrator fields (already handled in preprocessing)
        perpetrator_fields = [
            'Perpetrator Sex',
            'Perpetrator Race', 
            'Perpetrator Ethnicity',
            'Perpetrator Age'
        ]
        
        # Additional potentially leaking fields
        relationship_fields = ['Relationship']
        
        # Check if these exist in data
        for col in self.df.columns:
            if col in perpetrator_fields:
                leakage_features.append({
                    'name': col,
                    'category': 'perpetrator_info',
                    'reason': 'Only known after suspect identified (post-solving)'
                })
            elif col in relationship_fields:
                # Relationship might be known from crime scene, but often revealed during investigation
                # Check correlation to see how predictive it is
                if 'Crime Solved' in self.df.columns:
                    # Calculate correlation via encoding
                    temp_df = self.df[[col, 'Crime Solved']].copy()
                    temp_df[col] = pd.Categorical(temp_df[col]).codes
                    temp_df['Crime Solved'] = temp_df['Crime Solved'].map({'Yes': 1, 'No': 0})
                    corr = temp_df[col].corr(temp_df['Crime Solved'])
                    
                    if abs(corr) > 0.3:  # Suspiciously high correlation
                        leakage_features.append({
                            'name': col,
                            'category': 'high_correlation',
                            'reason': f'High correlation with target ({corr:.3f}), may be post-hoc',
                            'correlation': float(corr)
                        })
        
        self.audit_results['leakage_features'] = leakage_features
        return leakage_features

    def analyze_correlations(self) -> Dict[str, float]:
        """Calculate correlations between features and target variable."""
        if 'Crime Solved' not in self.df.columns:
            print("Warning: 'Crime Solved' target not found")
            return {}
        
        correlations = {}
        df_copy = self.df.copy()
        
        # Encode target
        df_copy['Crime Solved'] = df_copy['Crime Solved'].map({'Yes': 1, 'No': 0})
        
        for col in df_copy.columns:
            if col == 'Crime Solved':
                continue
                
            try:
                # Encode categorical
                if df_copy[col].dtype == 'object':
                    df_copy[col] = pd.Categorical(df_copy[col]).codes
                
                # Calculate correlation
                corr = df_copy[col].corr(df_copy['Crime Solved'])
                
                if not pd.isna(corr):
                    correlations[col] = float(corr)
                    
                    # Flag high correlations
                    if abs(corr) > 0.25:
                        self.audit_results['high_correlation_features'].append({
                            'name': col,
                            'correlation': float(corr),
                            'abs_correlation': float(abs(corr))
                        })
            except Exception as e:
                print(f"Could not calculate correlation for {col}: {e}")
        
        # Sort high correlation features
        self.audit_results['high_correlation_features'].sort(
            key=lambda x: x['abs_correlation'], 
            reverse=True
        )
        
        return correlations

    def analyze_variance(self) -> Dict[str, float]:
        """Identify low-variance features that may not be useful."""
        variances = {}
        
        for col in self.df.columns:
            if col == 'Crime Solved':
                continue
            
            try:
                if self.df[col].dtype == 'object':
                    # For categorical, use value_counts to check if dominated by one category
                    value_counts = self.df[col].value_counts(normalize=True)
                    if len(value_counts) > 0:
                        max_freq = value_counts.iloc[0]
                        variance = 1 - max_freq  # "Variance" = 1 - dominant frequency
                        variances[col] = float(variance)
                        
                        if max_freq > 0.95:  # 95%+ of values are the same
                            self.audit_results['low_variance_features'].append({
                                'name': col,
                                'dominant_value': value_counts.index[0],
                                'frequency': float(max_freq)
                            })
                else:
                    # For numeric, use standard variance
                    var = self.df[col].var()
                    if not pd.isna(var):
                        variances[col] = float(var)
            except Exception as e:
                print(f"Could not calculate variance for {col}: {e}")
        
        return variances

    def generate_feature_info(self, correlations: Dict[str, float]) -> None:
        """Generate detailed information about each feature."""
        for col in self.df.columns:
            if col == 'Crime Solved':
                continue
            
            info = {
                'name': col,
                'dtype': str(self.df[col].dtype),
                'missing_count': int(self.df[col].isna().sum()),
                'missing_pct': float(self.df[col].isna().sum() / len(self.df) * 100),
                'unique_values': int(self.df[col].nunique()),
                'correlation': correlations.get(col, None)
            }
            
            # Add sample values for categorical
            if self.df[col].dtype == 'object':
                value_counts = self.df[col].value_counts().head(5)
                info['top_values'] = {k: int(v) for k, v in value_counts.items()}
            else:
                info['mean'] = float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None
                info['std'] = float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None
                info['min'] = float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None
                info['max'] = float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None
            
            self.audit_results['feature_info'][col] = info

    def generate_recommendations(self) -> None:
        """Generate recommendations for feature selection."""
        recommended_drops = set()
        safe_features = []
        
        # Add known leakage features
        for feature in self.audit_results['leakage_features']:
            recommended_drops.add(feature['name'])
        
        # Consider dropping very low variance features
        for feature in self.audit_results['low_variance_features']:
            if feature['frequency'] > 0.98:  # 98%+ same value
                recommended_drops.add(feature['name'])
        
        # Identify safe features
        for col in self.df.columns:
            if col == 'Crime Solved':
                continue
            if col not in recommended_drops:
                # Check if it has reasonable properties
                info = self.audit_results['feature_info'].get(col, {})
                missing_pct = info.get('missing_pct', 100)
                
                # If less than 50% missing and not flagged, consider safe
                if missing_pct < 50:
                    safe_features.append(col)
        
        self.audit_results['recommended_drops'] = list(recommended_drops)
        self.audit_results['safe_features'] = safe_features

    def run_audit(self) -> Dict:
        """Run complete feature audit."""
        print("\n" + "="*80)
        print("FEATURE AUDIT FOR DATA LEAKAGE")
        print("="*80 + "\n")
        
        self.load_data()
        
        print("\n1. Identifying potential leakage features...")
        leakage_features = self.identify_leakage_features()
        print(f"   Found {len(leakage_features)} potential leakage features")
        
        print("\n2. Analyzing correlations with target...")
        correlations = self.analyze_correlations()
        print(f"   Analyzed {len(correlations)} features")
        print(f"   Found {len(self.audit_results['high_correlation_features'])} high correlation features (|r| > 0.25)")
        
        print("\n3. Analyzing feature variance...")
        variances = self.analyze_variance()
        print(f"   Found {len(self.audit_results['low_variance_features'])} low variance features")
        
        print("\n4. Generating detailed feature information...")
        self.generate_feature_info(correlations)
        
        print("\n5. Generating recommendations...")
        self.generate_recommendations()
        
        return self.audit_results

    def print_summary(self) -> None:
        """Print audit summary to console."""
        print("\n" + "="*80)
        print("AUDIT SUMMARY")
        print("="*80)
        
        print(f"\nTotal features analyzed: {self.audit_results['total_features']}")
        print(f"Recommended to drop: {len(self.audit_results['recommended_drops'])}")
        print(f"Safe to use: {len(self.audit_results['safe_features'])}")
        
        if self.audit_results['leakage_features']:
            print("\n" + "-"*80)
            print("LEAKAGE FEATURES (must drop):")
            print("-"*80)
            for feature in self.audit_results['leakage_features']:
                print(f"  • {feature['name']}")
                print(f"    Category: {feature['category']}")
                print(f"    Reason: {feature['reason']}")
                if 'correlation' in feature:
                    print(f"    Correlation: {feature['correlation']:.3f}")
        
        if self.audit_results['high_correlation_features']:
            print("\n" + "-"*80)
            print("HIGH CORRELATION FEATURES (review carefully):")
            print("-"*80)
            for feature in self.audit_results['high_correlation_features'][:10]:  # Top 10
                print(f"  • {feature['name']}: {feature['correlation']:.3f}")
        
        if self.audit_results['low_variance_features']:
            print("\n" + "-"*80)
            print("LOW VARIANCE FEATURES (consider dropping):")
            print("-"*80)
            for feature in self.audit_results['low_variance_features'][:5]:  # Top 5
                print(f"  • {feature['name']}: {feature['frequency']:.1%} are '{feature['dominant_value']}'")

    def save_results(self, output_dir: str = None) -> str:
        """Save audit results to JSON file."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.audit_results['timestamp']
        output_file = output_dir / f'feature_audit_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        print(f"\nAudit results saved to: {output_file}")
        return str(output_file)


def main():
    """Run feature audit from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audit features for data leakage')
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/data.csv',
        help='Path to raw data CSV file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100000,
        help='Number of rows to sample for analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute if needed
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent.parent / data_path
    
    auditor = FeatureAuditor(str(data_path))
    auditor.run_audit()
    auditor.print_summary()
    auditor.save_results(args.output)


if __name__ == '__main__':
    main()
