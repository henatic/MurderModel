import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.compare import load_data
from src.models.base_model import BaseModel
from src.preprocessing.data_processor import DataProcessor
from src.models.logistic_model import LogisticModel


class TestComparePipeline(unittest.TestCase):
    """Lightweight test to ensure compare pipeline steps avoid leakage and NaN issues."""

    def test_split_then_preprocess(self):
        # synthetic data with a target
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(200, 4), columns=[f'f{i}' for i in range(4)])
        y = pd.Series((rng.rand(200) > 0.4).astype(int), name='target')

        # Split raw
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = BaseModel.split_data(
            X, y, test_size=0.2, val_size=0.1, stratify=True, random_state=42
        )

        # Fit preprocessing on train only
        processor = DataProcessor(drop_leakage_features=False)
        X_train, _ = processor.fit_transform(X_train_raw)
        X_val = processor.transform(X_val_raw)
        X_test = processor.transform(X_test_raw)

        # Align target indices to avoid dtype issues
        y_train = y_train.loc[X_train.index]
        y_val = y_val.loc[X_val.index]
        y_test = y_test.loc[X_test.index]

        self.assertFalse(X_train.isna().any().any())
        # Ensure column sets line up after transform
        self.assertEqual(X_train.columns.tolist(), X_val.columns.tolist())
        self.assertEqual(X_train.columns.tolist(), X_test.columns.tolist())

        model = LogisticModel(random_state=42, scaler=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(y_test))


if __name__ == "__main__":
    unittest.main()
