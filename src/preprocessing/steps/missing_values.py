# src/preprocessing/steps/missing_values.py

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np

def handle_missing_values(
    df: pd.DataFrame,
    action: str,
    strategy: str = None,
    columns: list = None,
    fill_value=None
) -> pd.DataFrame:
    """
    Handle missing values based on action:
    - action: 'drop' or 'impute'
    - If 'drop': drop all rows with missing values in specified columns.
    - If 'impute': use the given strategy on specified columns.
        strategy must be one of:
          'mean', 'median', 'most_frequent', 'constant',
          'ffill', 'bfill', 'knn'
    - columns: list of column names to consider; defaults to all columns.
    - fill_value: only used when strategy='constant'.
    """
    df = df.copy()
    columns = columns or df.columns.tolist()

    if action not in {"drop", "impute"}:
        raise ValueError("`action` must be either 'drop' or 'impute'")

    if action == "drop":
        # Drop any row with missing values in the specified columns
        return df.dropna(axis=0, subset=columns)

    # action == "impute"
    if strategy is None:
        raise ValueError("`strategy` must be provided when action='impute'")

    strategy = strategy.lower()
    # Identify rows with any missing in the target columns
    mask = df[columns].isnull().any(axis=1)

    if strategy in {"mean", "median", "most_frequent", "constant"}:
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        # Fit only on rows where there is missing, but also need consistent columns shape
        # For SimpleImputer, it’s ok to fit on df[columns] directly
        df_loc = df[columns]
        df.loc[:, columns] = SimpleImputer(strategy=strategy, fill_value=fill_value).fit_transform(df_loc)

    elif strategy == "ffill":
        # Forward fill on specified columns
        df.loc[:, columns] = df[columns].ffill()

    elif strategy == "bfill":
        # Backward fill on specified columns
        df.loc[:, columns] = df[columns].bfill()

    elif strategy == "knn":
        # KNNImputer works feature-wise; apply on all specified columns
        imputer = KNNImputer(n_neighbors=3)
        df.loc[:, columns] = imputer.fit_transform(df[columns])

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    return df

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing missing values per column:
    - 'missing_count': number of missing entries
    - 'missing_pct': percentage of missing entries
    """
    total = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total * 100).round(2)
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    })
    return summary
