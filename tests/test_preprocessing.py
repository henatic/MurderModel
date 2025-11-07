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
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'Year': [1980, 1981, 1982, np.nan, 1984],
            'Month': [1, 13, 6, 7, 8],
            'Agency Type': ['City', 'County', 'State', 'City', None],
            'State': ['CA', 'NY', 'TX', 'FL', 'CA'],
            'Crime Type': ['Murder', 'Murder', 'Murder', 'Murder', 'Murder'],
            'Crime Solved': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Victim Sex': ['M', 'F', 'M', 'F', None],
            'Victim Age': [25, 30, -1, 40, 1000],
            'Victim Race': ['White', 'Black', 'Asian', 'White', None],
            'Victim Ethnicity': ['Hispanic', 'Non-Hispanic', 'Hispanic', 'Non-Hispanic', None],
            'Perpetrator Sex': ['M', 'M', 'F', None, 'M'],
            'Perpetrator Age': [30, 25, -5, 35, np.nan],
            'Perpetrator Race': ['White', 'Black', 'White', None, 'Asian'],
            'Perpetrator Ethnicity': ['Hispanic', 'Non-Hispanic', None, 'Hispanic', 'Non-Hispanic'],
            'Relationship': ['Stranger', 'Friend', 'Family', None, 'Stranger'],
            'Weapon': ['Gun', 'Knife', 'Other', None, 'Gun']
        })
        
    def test_initialization(self):
        """Test DataProcessor initialization."""
        self.assertIsInstance(self.data_processor, DataProcessor)
        self.assertIsInstance(self.data_processor.label_encoders, dict)
        
    def test_validate_data(self):
        """Test data validation."""
        processed_data, messages = self.data_processor.validate_data(self.test_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIsInstance(messages, list)
        
    def test_handle_missing_values(self):
        """Test missing value handling."""
        processed_data = self.data_processor.handle_missing_values(self.test_data)
        self.assertEqual(processed_data.isna().sum().sum(), 0)
        
    def test_handle_outliers(self):
        """Test outlier handling."""
        processed_data = self.data_processor.handle_outliers(self.test_data)
        self.assertTrue((processed_data['Victim Age'] <= 100).all())
        self.assertTrue((processed_data['Perpetrator Age'] >= 0).all())
        
    def test_encode_categorical(self):
        """Test categorical encoding."""
        processed_data = self.data_processor.encode_categorical(self.test_data)
        for col in self.data_processor.categorical_columns:
            if col in processed_data.columns:
                self.assertTrue(processed_data[col].dtype in [np.int32, np.int64])
                
    def test_scale_numeric(self):
        """Test numeric scaling."""
        # First handle missing values to avoid scaling issues
        clean_data = self.data_processor.handle_missing_values(self.test_data)
        processed_data = self.data_processor.scale_numeric(clean_data)
        for col in self.data_processor.numeric_columns:
            if col in processed_data.columns:
                self.assertAlmostEqual(processed_data[col].mean(), 0, places=1)
                self.assertAlmostEqual(processed_data[col].std(), 1, places=1)
                
    def test_engineer_features(self):
        """Test feature engineering."""
        processed_data = self.data_processor.engineer_features(self.test_data)
        self.assertIn('Season', processed_data.columns)
        self.assertIn('Victim_Age_Group', processed_data.columns)
        self.assertIn('Perpetrator_Age_Group', processed_data.columns)
        
    def test_process_data(self):
        """Test complete data processing pipeline."""
        processed_data, messages = self.data_processor.process_data(self.test_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.isna().sum().sum(), 0)
        self.assertGreater(len(processed_data.columns), len(self.test_data.columns))

if __name__ == '__main__':
    unittest.main()
