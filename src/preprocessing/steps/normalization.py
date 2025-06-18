import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def apply_scaling(df: pd.DataFrame, columns: list = None, method: str = "minmax") -> pd.DataFrame:
    """
    Normalize/scale numeric columns using the specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to normalize. If None, uses all numeric columns.
        method (str): One of ["minmax", "zscore", "robust", "maxabs"]
            - minmax: Scale to range [0,1]
            - zscore: Standardize to mean=0, std=1
            - robust: Scale using statistics robust to outliers
            - maxabs: Scale by maximum absolute value
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Validate method
    method = method.lower()
    if method not in ["minmax", "zscore", "robust", "maxabs"]:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Select appropriate scaler
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "maxabs":
        scaler = MaxAbsScaler()
        
    # Fit and transform only specified columns
    df[columns] = scaler.fit_transform(df[columns])
    
    return df