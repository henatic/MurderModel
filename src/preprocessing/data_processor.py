"""
Data preprocessing utilities for the Murder Model project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Tuple, Optional

class DataProcessor:
    """Class for handling data preprocessing tasks."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_columns = [
            'Agency Type', 'State', 'Crime Type', 'Crime Solved',
            'Victim Sex', 'Victim Race', 'Victim Ethnicity',
            'Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Ethnicity',
            'Relationship', 'Weapon'
        ]
        self.numeric_columns = ['Year', 'Month', 'Victim Age', 'Perpetrator Age']
        
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate input data for required columns and data types.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Cleaned dataframe and list of validation messages
        """
        messages = []
        
        # Check required columns
        required_columns = self.categorical_columns + self.numeric_columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            messages.append(f"Missing required columns: {missing_columns}")
            
        # Validate numeric columns
        for col in self.numeric_columns:
            if col in data.columns:
                # Convert to numeric, coercing errors to NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
                invalid_count = data[col].isna().sum()
                if invalid_count > 0:
                    messages.append(f"Found {invalid_count} invalid values in {col}")
                    
        return data, messages
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # For numeric columns, fill with median
        for col in self.numeric_columns:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
                
        return data
    
    def handle_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in numeric columns using both z-score method and domain knowledge.
        
        Args:
            data (pd.DataFrame): Input dataframe
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        data = data.copy()
        
        # First handle missing values to avoid issues with statistics
        data = self.handle_missing_values(data)
        
        # Domain-specific constraints
        data.loc[data['Victim Age'] > 100, 'Victim Age'] = 100  # Cap at 100 years
        data.loc[data['Victim Age'] < 0, 'Victim Age'] = 0  # No negative ages
        
        data.loc[data['Perpetrator Age'] > 100, 'Perpetrator Age'] = 100  # Cap at 100 years
        data.loc[data['Perpetrator Age'] < 0, 'Perpetrator Age'] = 0  # No negative ages
        
        data.loc[data['Month'] > 12, 'Month'] = 12  # Cap at 12 months
        data.loc[data['Month'] < 1, 'Month'] = 1  # Minimum 1 month
        
        # Use z-score method for remaining outliers
        for col in self.numeric_columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > threshold, col] = data[col].median()
                
        return data
    
    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables
        """
        data = data.copy()
        for col in self.categorical_columns:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        return data
    
    def scale_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with scaled numeric features
        """
        data = data.copy()
        for col in self.numeric_columns:
            if col in data.columns:
                data[col] = (data[col] - data[col].mean()) / data[col].std()
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from existing data.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with additional engineered features
        """
        data = data.copy()
        
        # Create season feature from Month
        data['Season'] = pd.cut(data['Month'], 
                              bins=[0, 3, 6, 9, 12],
                              labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Create age group features
        data['Victim_Age_Group'] = pd.cut(data['Victim Age'],
                                        bins=[0, 18, 25, 35, 50, 65, 100],
                                        labels=['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65'])
        
        data['Perpetrator_Age_Group'] = pd.cut(data['Perpetrator Age'],
                                             bins=[0, 18, 25, 35, 50, 65, 100],
                                             labels=['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65'])
        
        # Add the new categorical columns to be encoded
        self.categorical_columns.extend(['Season', 'Victim_Age_Group', 'Perpetrator_Age_Group'])
        
        return data
    
    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete data processing pipeline.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed dataframe and validation messages
        """
        # Start with data validation
        data, messages = self.validate_data(data)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Handle outliers
        data = self.handle_outliers(data)
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Encode categorical variables
        data = self.encode_categorical(data)
        
        # Scale numeric features
        data = self.scale_numeric(data)
        
        return data, messages
