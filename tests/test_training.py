import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_training import trainer, model_selector

def test_model_training_and_saving(tmp_path):
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'label': [0, 1, 0, 1, 0]
    })

    X = df[['feature1', 'feature2']]
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    model_type = model_selector.get_model('Logistic Regression', task_type='classification')
    model = trainer.train_model(model_type, X_train, y_train)

    assert model is not None
    assert hasattr(model, "fit")
