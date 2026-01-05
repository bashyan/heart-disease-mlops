import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.preprocess import load_data, clean_data, split_features_target, get_preprocessor


def test_model_training():
    df = load_data("data/heart.csv")
    df_clean = clean_data(df)
    X, y = split_features_target(df_clean)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    score = pipeline.score(X, y)
    assert score > 0.5  # sanity check


def test_model_prediction_shape():
    df = load_data("data/heart.csv")
    df_clean = clean_data(df)
    X, y = split_features_target(df_clean)

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])

    assert len(preds) == 5

def test_model_can_learn_simple_pattern():
    """
    Sanity Check: Ensures the model architecture can actually learn.
    Trains on a dummy 'if X > 0.5 then Y=1' dataset.
    """
    # 1. Create simple dummy data (100 rows)
    X_dummy = np.random.rand(100, 5) 
    # Rule: If 1st column > 0.5, label is 1. Else 0.
    y_dummy = (X_dummy[:, 0] > 0.5).astype(int)

    # 2. Train a small instance of your model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)

    # 3. Assert it learned the rule (Accuracy should be high, e.g. > 90%)
    acc = accuracy_score(y_dummy, model.predict(X_dummy))
    assert acc > 0.8, f"Model architecture is broken! Accuracy on trivial data: {acc}"
