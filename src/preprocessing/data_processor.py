"""
Data preprocessing utilities for the Murder Model project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    """Class for handling data preprocessing tasks."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process input data.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        return data.copy()
