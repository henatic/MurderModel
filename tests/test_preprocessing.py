import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_processor import DataProcessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.data_processor = DataProcessor()
        
    def test_initialization(self):
        """Test DataProcessor initialization."""
        self.assertIsInstance(self.data_processor, DataProcessor)
        
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with sample data
        test_data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        processed_data = self.data_processor.load_data(test_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(processed_data), 3)

if __name__ == "__main__":
    unittest.main()
