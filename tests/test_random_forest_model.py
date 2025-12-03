"""
Unit tests for RandomForestModel.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from src.models.random_forest_model import RandomForestModel


class TestRandomForestModel(unittest.TestCase):
    """Test cases for Random Forest model."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple synthetic data
        n_samples = 200
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
        })
        # Create a simple binary target with some pattern
        self.y = pd.Series(
            (self.X['feature1'] + self.X['feature2'] > 0).astype(int),
            name='target'
        )
    
    def test_initialization(self):
        """Test model initialization with various parameters."""
        model = RandomForestModel(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 10)
        self.assertEqual(model.random_state, 42)
        self.assertIsNone(model.pipeline)
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(p in [0, 1] for p in predictions))
        
        # Should achieve reasonable accuracy on this simple pattern
        accuracy = (predictions == self.y).mean()
        self.assertGreater(accuracy, 0.7)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        probas = model.predict_proba(self.X)
        self.assertEqual(probas.shape, (len(self.X), 2))
        
        # Probabilities should sum to 1
        row_sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.X)))
        
        # All probabilities should be between 0 and 1
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        importance = model.get_feature_importance()
        self.assertEqual(len(importance), 3)
        
        # Importances should sum to approximately 1
        self.assertAlmostEqual(importance.sum(), 1.0, places=5)
        
        # All importances should be non-negative
        self.assertTrue(np.all(importance >= 0))
    
    def test_save_load(self):
        """Test model persistence."""
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        
        original_predictions = model.predict(self.X)
        original_importance = model.get_feature_importance()
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'rf_model.pkl')
            model.save_model(filepath)
            
            # Load model
            loaded_model = RandomForestModel()
            loaded_model.load_model(filepath)
            
            # Verify predictions match
            loaded_predictions = loaded_model.predict(self.X)
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
            # Verify feature importance matches
            loaded_importance = loaded_model.get_feature_importance()
            np.testing.assert_array_almost_equal(original_importance, loaded_importance)
    
    def test_with_scaler(self):
        """Test model with StandardScaler in pipeline."""
        model = RandomForestModel(n_estimators=10, random_state=42, scaler=True)
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
        # Should still work with scaling
        accuracy = (predictions == self.y).mean()
        self.assertGreater(accuracy, 0.6)


if __name__ == '__main__':
    unittest.main()

