# src/preprocessing/steps/missing_values.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

def get_auto_impute_strategy(series: pd.Series) -> str:
    """
    Decide a statistical imputation strategy for a single Series:
    - Numeric:
      * If distribution roughly symmetric (|skew| < 0.5): 'mean'
      * If moderately/highly skewed (|skew| >= 0.5): 'median'
    - Categorical or object: 'most_frequent'
    - Boolean: 'most_frequent'
    - Datetime: 'ffill' (forward fill), fallback 'bfill'
    """
    if series.dtype.kind in 'biufc':  # numeric
        non_na = series.dropna()
        if non_na.empty:
            return 'median'
        skew = non_na.skew()
        if abs(skew) < 0.5:
            return 'mean'
        else:
            return 'median'
    elif pd.api.types.is_bool_dtype(series):
        return 'most_frequent'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'ffill'
    elif pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        return 'most_frequent'
    else:
        return 'most_frequent'


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


def handle_missing_values(
    df: pd.DataFrame,
    *,
    target_col: str = None,
    method: str = 'statistical',
    columns: list = None,
    fill_value=None,
    knn_neighbors: int = 3
) -> (pd.DataFrame, dict):
    """
    Handle missing values in df, returning (new_df, info).
    Two imputation modes for independent features:
      - method='statistical': per-column auto strategy (mean/median/most_frequent/ffill/bfill).
      - method='knn': use KNNImputer on numeric independent columns only.
    Rows with missing in target_col (if given) are dropped first.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str or None
        Name of the target column. Rows with missing here are dropped first.
    method : {'statistical', 'knn'}
        Imputation mode for independent columns.
    columns : list of str or None
        Specific independent columns to impute. If None, defaults to all columns (except target)
        that have missing after dropping target-missing rows.
    fill_value : scalar, optional
        Only used if statistical strategy resolves to 'constant' (not used by auto logic here).
    knn_neighbors : int
        Number of neighbors for KNNImputer when method='knn'.

    Returns
    -------
    df_result : pd.DataFrame
        DataFrame after dropping target-missing rows and imputing independent-missing.
    info : dict
        {
          'dropped_target_rows': int,
          'summary_before': pd.DataFrame,
          'imputed_columns': {
              col_name: {'strategy': str, 'missing_before': int, 'missing_after': int},
              ...
          },
          'remaining_missing': pd.Series after imputation,
          'summary_after': pd.DataFrame
        }
    """
    df_work = df.copy()
    info = {}

    # Summary before any operation
    summary_before = get_missing_summary(df_work)
    info['summary_before'] = summary_before

    # 1) Drop rows with missing target
    dropped_target_rows = 0
    if target_col is not None:
        if target_col not in df_work.columns:
            raise KeyError(f"target_col='{target_col}' not in DataFrame columns")
        mask = df_work[target_col].isnull()
        if mask.any():
            dropped_target_rows = int(mask.sum())
            df_work = df_work.loc[~mask].copy()
    info['dropped_target_rows'] = dropped_target_rows

    # 2) Determine independent columns to handle
    # After dropping target-missing, find columns with missing
    if columns is None:
        cols_missing = df_work.columns[df_work.isnull().any()].tolist()
        if target_col in cols_missing:
            cols_missing.remove(target_col)
        columns_to_handle = cols_missing
    else:
        # filter valid and exclude target
        columns_to_handle = [c for c in columns if c in df_work.columns and c != target_col]
    info['columns_to_handle'] = columns_to_handle.copy()

    # If no missing in these, return early
    if not df_work.isnull().any().any():
        info['imputed_columns'] = {}
        info['remaining_missing'] = df_work.isnull().sum()
        info['summary_after'] = get_missing_summary(df_work)
        return df_work, info

    method = method.lower()
    if method not in {'statistical', 'knn'}:
        raise ValueError("`method` must be either 'statistical' or 'knn'")

    imputed_columns = {}

    if method == 'knn':
        # Only numeric independent columns used for KNNImputer
        num_cols = [c for c in columns_to_handle if pd.api.types.is_numeric_dtype(df_work[c])]
        if num_cols:
            # KNNImputer only on numeric columns; other columns remain unchanged (they still may have missing)
            knn_imp = KNNImputer(n_neighbors=knn_neighbors)
            before_missing = df_work[num_cols].isnull().sum().to_dict()
            arr = knn_imp.fit_transform(df_work[num_cols])
            df_work[num_cols] = pd.DataFrame(arr, columns=num_cols, index=df_work.index)
            after_missing = df_work[num_cols].isnull().sum().to_dict()
            for col in num_cols:
                imputed_columns[col] = {
                    'strategy': 'knn',
                    'missing_before': int(before_missing.get(col, 0)),
                    'missing_after': int(after_missing.get(col, 0))
                }
        # For non-numeric columns with missing, we do NOT impute when method='knn'; they remain missing.
    else:
        # method == 'statistical'
        # For each independent column with missing, choose auto strategy and impute
        for col in columns_to_handle:
            series = df_work[col]
            missing_before = int(series.isnull().sum())
            if missing_before == 0:
                continue
            strat = get_auto_impute_strategy(series)
            # Apply chosen strategy
            if strat in {'mean', 'median', 'most_frequent', 'constant'}:
                if strat == 'constant' and fill_value is None:
                    raise ValueError("fill_value must be provided for constant strategy")
                imp = SimpleImputer(strategy=strat, fill_value=fill_value)
                arr = imp.fit_transform(series.to_frame())
                df_work[col] = arr.ravel()
            elif strat == 'ffill':
                df_work[col] = series.ffill()
                if df_work[col].isnull().any():
                    df_work[col] = df_work[col].bfill()
            elif strat == 'bfill':
                df_work[col] = series.bfill()
                if df_work[col].isnull().any():
                    df_work[col] = df_work[col].ffill()
            else:
                # Fallback: most_frequent
                imp = SimpleImputer(strategy='most_frequent')
                arr = imp.fit_transform(series.to_frame())
                df_work[col] = arr.ravel()
            missing_after = int(df_work[col].isnull().sum())
            imputed_columns[col] = {
                'strategy': strat,
                'missing_before': missing_before,
                'missing_after': missing_after
            }

    # Prepare info and return
    summary_after = get_missing_summary(df_work)
    remaining_missing = df_work.isnull().sum()

    info['imputed_columns'] = imputed_columns
    info['remaining_missing'] = remaining_missing
    info['summary_after'] = summary_after

    return df_work, info
