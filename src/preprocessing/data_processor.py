"""Data preprocessing utilities for the Murder Model project."""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Fit/transform data preprocessing with leakage-aware defaults."""

    def __init__(self, drop_leakage_features: bool = True):
        self.scaler = StandardScaler()
        self.scaler_params: Dict[str, pd.Series] = {}
        self.label_encoders: Dict[str, Dict[str, int]] = {}
        self.base_categorical_columns = [
            'Agency Type', 'State', 'Crime Type', 'Crime Solved',
            'Victim Sex', 'Victim Race', 'Victim Ethnicity',
            'Perpetrator Sex', 'Perpetrator Race', 'Perpetrator Ethnicity',
            'Relationship', 'Weapon'
        ]
        self.engineered_categorical_columns = [
            'Season', 'Victim_Age_Group', 'Perpetrator_Age_Group'
        ]
        self.numeric_columns = ['Year', 'Month', 'Victim Age', 'Perpetrator Age']
        self.drop_leakage_features = drop_leakage_features
        self.leakage_columns = {
            'Perpetrator Sex',
            'Perpetrator Race',
            'Perpetrator Ethnicity',
            'Perpetrator Age',
            'Perpetrator_Age_Group',
        }
        self.fitted = False
        self.fitted_categorical_columns: List[str] = []
        self.fitted_numeric_columns: List[str] = []
        # Backwards compatibility for tests/scripts that reference categorical_columns
        self.categorical_columns = self.base_categorical_columns + self.engineered_categorical_columns

    def _month_to_number(self, data: pd.DataFrame, messages: List[str]) -> pd.DataFrame:
        if 'Month' in data.columns and data['Month'].dtype == 'object':
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            data['Month'] = data['Month'].map(month_map)
            unmapped = data['Month'].isna().sum()
            if unmapped > 0:
                messages.append(f"Found {unmapped} invalid month names in Month column")
        return data

    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate input data for required columns and data types.
        """
        messages: List[str] = []
        data = data.copy()

        data = self._month_to_number(data, messages)

        # Coerce numeric columns
        for col in self.numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                invalid_count = data[col].isna().sum()
                if invalid_count > 0:
                    messages.append(f"Found {invalid_count} invalid values in {col}")

        return data, messages

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values robustly: median for numeric, mode/Unknown for categorical.
        """
        data = data.copy()

        for col in self.numeric_columns:
            if col in data.columns:
                median = data[col].median()
                fill_value = 0 if pd.isna(median) else median
                data[col] = data[col].fillna(fill_value)

        for col in self.base_categorical_columns + self.engineered_categorical_columns:
            if col in data.columns:
                mode = data[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                data[col] = data[col].fillna(fill_value)

        return data

    def handle_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers with domain caps and z-score clipping.
        """
        data = data.copy()
        data = self.handle_missing_values(data)

        if 'Victim Age' in data.columns:
            data.loc[data['Victim Age'] > 100, 'Victim Age'] = 100
            data.loc[data['Victim Age'] < 0, 'Victim Age'] = 0
        if 'Perpetrator Age' in data.columns:
            data.loc[data['Perpetrator Age'] > 100, 'Perpetrator Age'] = 100
            data.loc[data['Perpetrator Age'] < 0, 'Perpetrator Age'] = 0
        if 'Month' in data.columns:
            data.loc[data['Month'] > 12, 'Month'] = 12
            data.loc[data['Month'] < 1, 'Month'] = 1

        for col in self.numeric_columns:
            if col in data.columns:
                std = data[col].std()
                if std and not pd.isna(std) and std > 0:
                    z_scores = np.abs((data[col] - data[col].mean()) / std)
                    data.loc[z_scores > threshold, col] = data[col].median()
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered categorical features.
        """
        data = data.copy()

        if 'Month' in data.columns:
            data['Season'] = pd.cut(
                data['Month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )
        if 'Victim Age' in data.columns:
            data['Victim_Age_Group'] = pd.cut(
                data['Victim Age'],
                bins=[0, 18, 25, 35, 50, 65, 100],
                labels=['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65']
            )
        if 'Perpetrator Age' in data.columns:
            data['Perpetrator_Age_Group'] = pd.cut(
                data['Perpetrator Age'],
                bins=[0, 18, 25, 35, 50, 65, 100],
                labels=['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65']
            )
        return data

    def _get_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        cols = []
        for col in self.base_categorical_columns + self.engineered_categorical_columns:
            if col in data.columns:
                if self.drop_leakage_features and col in self.leakage_columns:
                    continue
                cols.append(col)
        return cols

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        cols = []
        for col in self.numeric_columns:
            if col in data.columns:
                if self.drop_leakage_features and col in self.leakage_columns:
                    continue
                cols.append(col)
        return cols

    def _encode_categorical(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        data = data.copy()
        if self.drop_leakage_features:
            drop_cols = [c for c in data.columns if c in self.leakage_columns]
            if drop_cols:
                data = data.drop(columns=drop_cols)
        cat_cols = self._get_categorical_columns(data)
        for col in cat_cols:
            values = data[col].astype(str)
            if fit:
                classes = {cls: idx for idx, cls in enumerate(sorted(values.unique()))}
                self.label_encoders[col] = classes
            classes = self.label_encoders.get(col, {})
            data[col] = values.map(lambda v: classes.get(v, -1)).astype(int)
        if fit:
            self.fitted_categorical_columns = cat_cols
        return data

    def _scale_numeric(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        data = data.copy()
        if self.drop_leakage_features:
            drop_cols = [c for c in data.columns if c in self.leakage_columns]
            if drop_cols:
                data = data.drop(columns=drop_cols)
        num_cols = self._get_numeric_columns(data)
        if not num_cols:
            return data
        if fit:
            means = data[num_cols].mean()
            stds = data[num_cols].std(ddof=1).replace(0, 1)
            self.scaler_params = {'mean': means, 'std': stds}
            self.fitted_numeric_columns = num_cols
        means = self.scaler_params['mean']
        stds = self.scaler_params['std']
        data[num_cols] = (data[num_cols] - means) / stds
        return data

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit preprocessing artifacts on data and return transformed copy.
        """
        data, messages = self.validate_data(data)
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        data = self.engineer_features(data)
        data = self._encode_categorical(data, fit=True)
        data = self._scale_numeric(data, fit=True)
        self.fitted = True
        keep_cols = self.fitted_categorical_columns + self.fitted_numeric_columns
        return data[keep_cols], messages

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using previously fitted encoders/scaler.
        """
        if not self.fitted:
            raise RuntimeError("DataProcessor must be fit before calling transform")
        data, _ = self.validate_data(data)
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        data = self.engineer_features(data)
        # Ensure only fitted columns are encoded/scaled to preserve shape
        data = data.copy()
        data = self._encode_categorical(data, fit=False)
        data = self._scale_numeric(data, fit=False)
        # Keep only columns seen during fit to maintain consistency
        keep_cols = self.fitted_categorical_columns + self.fitted_numeric_columns
        missing = [c for c in keep_cols if c not in data.columns]
        if missing:
            for col in missing:
                data[col] = 0
        return data[keep_cols]

    # Compatibility wrappers used in legacy tests/scripts
    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode and fit on provided data (compatibility wrapper)."""
        return self._encode_categorical(data, fit=True)

    def scale_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale and fit on provided data (compatibility wrapper)."""
        return self._scale_numeric(data, fit=True)

    def process_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Backwards-compatible wrapper that fits and transforms on the same data.
        """
        return self.fit_transform(data)
