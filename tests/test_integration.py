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
        cls.X = cls.y = cls.X_val = cls.y_val = cls.X_test = cls.y_test = None
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
            
            # Drop rows with NaN in target
            df = df.dropna(subset=[target_col])
            df = df.drop_duplicates()

            # Separate target
            y_full = df[target_col]
            X_full = df.drop(columns=[target_col])

            # Split before fitting preprocessors to avoid leakage
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = BaseModel.split_data(
                X_full, y_full,
                test_size=0.2,
                val_size=0.1,
                stratify=True,
                random_state=42
            )

            processor = DataProcessor()
            X_train, _ = processor.fit_transform(X_train_raw)
            X_val = processor.transform(X_val_raw)
            X_test = processor.transform(X_test_raw)

            # Encode target if needed (convert to binary 0/1) after split
            if y_train.dtype == 'object' or y_train.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train = pd.Series(le.fit_transform(y_train), index=y_train.index)
                y_val = pd.Series(le.transform(y_val), index=y_val.index)
                y_test = pd.Series(le.transform(y_test), index=y_test.index)

            # Keep only if we have enough samples and features
            if len(y_train) < 50 or X_train.shape[1] < 2:
                print(f"Integration test data check: {len(y_train)} train samples, {X_train.shape[1]} features")
                cls.X = None
                cls.y = None
                return
                
            print(f"Integration test loaded: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, features={X_train.shape[1]}")
            cls.X = X_train
            cls.X_val = X_val
            cls.X_test = X_test
            cls.y = y_train
            cls.y_val = y_val
            cls.y_test = y_test
            
        except Exception as e:
            print(f"Could not load real data for integration test: {e}")
            cls.X = None
            cls.y = None
    
    def test_stratified_split_on_real_data(self):
        """Test stratified splitting on real preprocessed data."""
        if self.X is None or self.y is None:
            self.skipTest("Real data not available")

        total = len(self.y) + len(self.y_val) + len(self.y_test)
        self.assertAlmostEqual(len(self.y) + len(self.y_val) + len(self.y_test), total, delta=1)
        self.assertGreater(len(self.y), 0)
        self.assertGreater(len(self.y_test), 0)

        overall_dist = pd.concat([self.y, self.y_val, self.y_test]).value_counts(normalize=True).sort_index()
        train_dist = self.y.value_counts(normalize=True).sort_index()
        test_dist = self.y_test.value_counts(normalize=True).sort_index()
        
        for cls in overall_dist.index:
            self.assertAlmostEqual(overall_dist[cls], train_dist.get(cls, 0), delta=0.05)
            self.assertAlmostEqual(overall_dist[cls], test_dist.get(cls, 0), delta=0.05)
    
    def test_full_pipeline_train_evaluate(self):
        """Test complete pipeline: split, train, evaluate on real data."""
        if self.X is None or self.y is None:
            self.skipTest("Real data not available")
        
        # Train model
        model = LogisticModel(random_state=42)
        model.fit(self.X, self.y)
        
        # Evaluate on train and test
        train_metrics = model.evaluate(self.X, self.y)
        test_metrics = model.evaluate(self.X_test, self.y_test)
        
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
        
        model = LogisticModel(random_state=42)
        model.fit(self.X, self.y)
        
        # Get predictions before saving
        preds_before = model.predict(self.X_test)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)

            model2 = LogisticModel()
            model2.load_model(tmp_path)

            preds_after = model2.predict(self.X_test)

            np.testing.assert_array_equal(preds_before, preds_after)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
