import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.inference.predictor import predict_with_model

def test_inference_prediction():
    model = LogisticRegression()
    X_train = pd.DataFrame({'f1': [1, 2, 3], 'f2': [3, 2, 1]})
    y_train = [0, 1, 0]
    model.fit(X_train, y_train)

    test_data = pd.DataFrame({'f1': [2], 'f2': [2]})
    preds = predict_with_model(model, test_data)
    assert len(preds) == 1
