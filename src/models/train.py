"""
Small training and evaluation harness for quick experiments.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.logistic_model import LogisticModel
from src.evaluation.evaluator import ModelEvaluator


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    # quick stratified split for demo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state
    )

    model = LogisticModel(random_state=random_state)
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
    return model, results


if __name__ == '__main__':
    # tiny demo using synthetic data
    from sklearn.datasets import make_classification
    X_arr, y_arr = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f'f{i}' for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr)
    model, results = train_and_evaluate(X, y)
    print(results)
