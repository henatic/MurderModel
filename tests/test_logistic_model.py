import unittest
import pandas as pd
import numpy as np

from src.models.logistic_model import LogisticModel


class TestLogisticModel(unittest.TestCase):
    def setUp(self):
        # small synthetic dataset
        rng = np.random.RandomState(0)
        X = rng.normal(size=(200, 5))
        y = (rng.rand(200) > 0.6).astype(int)
        self.X = pd.DataFrame(X, columns=[f'c{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)

    def test_fit_predict(self):
        model = LogisticModel(random_state=0)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_predict_proba_shape(self):
        model = LogisticModel(random_state=0)
        model.fit(self.X, self.y)
        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.X.shape[0], 2))

    def test_feature_importance(self):
        model = LogisticModel(random_state=0)
        model.fit(self.X, self.y)
        fi = model.get_feature_importance()
        self.assertEqual(fi.shape[0], self.X.shape[1])


if __name__ == '__main__':
    unittest.main()
