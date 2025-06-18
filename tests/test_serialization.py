import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.model_training.serializer import save_model, load_model
import tempfile
import os

def test_model_serialization_pkl():
    model = LogisticRegression()
    model.fit([[0, 1], [1, 0]], [0, 1])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_model(model, tmpdir, "test_model", formats=[".pkl"])
        assert os.path.exists(path[0])
        loaded_model = load_model(path[0])
        preds = loaded_model.predict([[0, 1]])
        assert preds[0] in [0, 1]
