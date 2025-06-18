# File: src/utils/validation.py

import pandas as pd

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic DataFrame validation to ensure it's suitable for EDA and preprocessing.
    """
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    if df.shape[1] == 0:
        return False
    return True

def validate_column_exists(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns

def validate_numeric_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[column])

def validate_categorical_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])

def validate_model_type(task_type: str, model_name: str) -> bool:
    classification_models = [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "Support Vector Machine", "Naive Bayes", "KNN"
    ]
    regression_models = ["Linear Regression"]

    if task_type == "classification":
        return model_name in classification_models
    elif task_type == "regression":
        return model_name in regression_models
    return False
