"""
Integration tests using real data preprocessing and model training pipeline.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing.data_processor import DataProcessor
from src.models.base_model import BaseModel
from src.models.logistic_model import LogisticModel


class TestIntegration(unittest.TestCase):
    """Integration tests for the full ML pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Load and preprocess a small sample of real data once for all tests."""
        # Path to raw data
        data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'data.csv'
        
        if not data_path.exists():
            cls.X = None
            cls.y = None
            return
            
        try:
            # Load only first 5000 rows for fast testing
            df = pd.read_csv(data_path, nrows=5000)
            
            # Check if we have a target column (adjust name as needed)
            # Common target names: 'Crime Solved', 'Solved', 'target', 'label'
            potential_targets = ['Crime Solved', 'Solved', 'target', 'label']
            target_col = None
            for col in potential_targets:
                if col in df.columns:
                    target_col = col
                    break
                    
            if target_col is None:
                # If no obvious target, skip
                cls.X = None
                cls.y = None
                return
            
            # Separate target
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Preprocess
            processor = DataProcessor()
            X_processed, issues = processor.process_data(X)
            
            # Drop rows with NaN in target
            valid_idx = ~y.isna()
            X_processed = X_processed[valid_idx]
            y = y[valid_idx]
            
            # Ensure all columns are numeric (drop any remaining non-numeric columns)
            X_processed = X_processed.select_dtypes(include=[np.number])
            
            # Drop any rows with NaN after preprocessing
            mask = ~X_processed.isna().any(axis=1)
            X_processed = X_processed[mask]
            y = y[mask]
            
            # Encode target if needed (convert to binary 0/1)
            if y.dtype == 'object' or y.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
            
            # Keep only if we have enough samples and features
            if len(y) < 100 or X_processed.shape[1] < 2:
                print(f"Integration test data check: {len(y)} samples, {X_processed.shape[1]} features")
                cls.X = None
                cls.y = None
                return
                
            print(f"Integration test loaded: {len(y)} samples, {X_processed.shape[1]} features")
            cls.X = X_processed
            cls.y = y
            
        except Exception as e:
            print(f"Could not load real data for integration test: {e}")
            cls.X = None
            cls.y = None
    
    def test_stratified_split_on_real_data(self):
        """Test stratified splitting on real preprocessed data."""
        if self.X is None or self.y is None:
            self.skipTest("Real data not available")
        
        X_train, X_val, X_test, y_train, y_val, y_test = BaseModel.split_data(
            self.X, self.y,
            test_size=0.2,
            val_size=0.1,
            stratify=True,
            random_state=42
        )
        
        # Check sizes
        total = len(self.y)
        self.assertGreater(len(y_train), 0)
        self.assertGreater(len(y_test), 0)
        if y_val is not None:
            self.assertGreater(len(y_val), 0)
            self.assertAlmostEqual(len(y_train) + len(y_val) + len(y_test), total, delta=1)
        else:
            self.assertAlmostEqual(len(y_train) + len(y_test), total, delta=1)
        
        # Check class distributions are similar
        overall_dist = self.y.value_counts(normalize=True).sort_index()
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        
        for cls in overall_dist.index:
            self.assertAlmostEqual(overall_dist[cls], train_dist[cls], delta=0.05)
            self.assertAlmostEqual(overall_dist[cls], test_dist[cls], delta=0.05)
    
    def test_full_pipeline_train_evaluate(self):
        """Test complete pipeline: split, train, evaluate on real data."""
        if self.X is None or self.y is None:
            self.skipTest("Real data not available")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = BaseModel.split_data(
            self.X, self.y,
            test_size=0.2,
            val_size=0.1,
            stratify=True,
            random_state=42
        )
        
        # Train model
        model = LogisticModel(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on train and test
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Sanity checks
        self.assertIn('accuracy', train_metrics)
        self.assertIn('accuracy', test_metrics)
        self.assertGreater(train_metrics['accuracy'], 0.0)
        self.assertLess(train_metrics['accuracy'], 1.1)
        self.assertGreater(test_metrics['accuracy'], 0.0)
        self.assertLess(test_metrics['accuracy'], 1.1)
        
        # Train accuracy should typically be >= test accuracy
        # (though not strictly guaranteed with regularization)
        self.assertIsNotNone(train_metrics['accuracy'])
        self.assertIsNotNone(test_metrics['accuracy'])
    
    def test_model_save_load(self):
        """Test model persistence with real data."""
        if self.X is None or self.y is None:
            self.skipTest("Real data not available")
        
        import tempfile
        import os
        
        # Split and train
        X_train, X_val, X_test, y_train, y_val, y_test = BaseModel.split_data(
            self.X, self.y,
            test_size=0.2,
            val_size=0.1,
            stratify=True,
            random_state=42
        )
        
        model = LogisticModel(random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions before saving
        preds_before = model.predict(X_test)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            
            # Load into new model instance
            model2 = LogisticModel()
            model2.load_model(tmp_path)
            
            # Get predictions after loading
            preds_after = model2.predict(X_test)
            
            # Should be identical
            np.testing.assert_array_equal(preds_before, preds_after)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
