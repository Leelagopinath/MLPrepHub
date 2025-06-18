import pandas as pd
from src.preprocessing.pipeline_manager import apply_pipeline
from src.preprocessing.steps.missing_values import handle_missing_values

def test_missing_value_step():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    processed_df = handle_missing_values(df, method="mean")
    assert processed_df.isnull().sum().sum() == 0

def test_pipeline_manager():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    steps = [{'step': 'missing_values', 'params': {'method': 'mean'}}]
    processed_df = apply_pipeline(df, steps)
    assert processed_df.isnull().sum().sum() == 0
