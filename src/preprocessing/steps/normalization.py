import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer
from scipy.stats import boxcox, yeojohnson

def apply_scaling(df: pd.DataFrame, columns: list = None, method: str = "auto", power: float = None, winsor_limits=(0.05, 0.05)) -> pd.DataFrame:
    """
    Normalize/scale or transform numeric columns using the specified method or auto-detect based on skewness.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to normalize/transform. If None, uses all numeric columns.
        method (str): One of ["auto", "minmax", "zscore", "robust", "maxabs", "log", "sqrt", "cbrt", "boxcox", "yeojohnson", "inverse", "power", "winsor"]
        power (float, optional): Exponent for power transformation if method="power"
        winsor_limits (tuple, optional): Limits for winsorizing if method="winsor"
    
    Returns:
        pd.DataFrame: DataFrame with transformed columns
    """
    method = method.lower()
    if method == "normalization":
        method = "minmax"
    if method == "standardization":
        method = "zscore"
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    method = method.lower()

    for col in columns:
        col_data = df[col].dropna()
        skew = col_data.skew()
        min_val = col_data.min()
        max_val = col_data.max()

        chosen_method = method
        # Auto-detect transformation if method is "auto"
        if method == "auto":
            if (skew > 1 or skew < -1) and (min_val > 0):
                # Highly skewed, strictly positive
                chosen_method = "boxcox"
            elif (skew > 1 or skew < -1) and (min_val <= 0):
                # Highly skewed, includes negatives
                chosen_method = "yeojohnson"
            elif skew > 0.5 and min_val > 0:
                # Positive skew, all positive
                chosen_method = "log"
            elif 0.2 < skew <= 0.5 and min_val >= 0:
                # Moderately skewed, count data
                chosen_method = "sqrt"
            elif abs(skew) > 0.5 and min_val < 0:
                # Moderate skew, both positive and negative
                chosen_method = "cbrt"
            elif abs(skew) > 2 and min_val > 0:
                # Highly skewed, positive, dominated by large values
                chosen_method = "inverse"
            elif abs(skew) > 0.5:
                # Flexible power transformation
                chosen_method = "power"
            elif abs(skew) > 0.5:
                # Outlier capping
                chosen_method = "winsor"
            else:
                # If not skewed, use zscore
                chosen_method = "zscore"

        # Apply the chosen transformation
        if chosen_method == "minmax":
            scaler = MinMaxScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
        elif chosen_method == "zscore":
            scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
        elif chosen_method == "robust":
            scaler = RobustScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
        elif chosen_method == "maxabs":
            scaler = MaxAbsScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
        elif chosen_method == "log":
            df[col] = np.log1p(df[col])
        elif chosen_method == "sqrt":
            df[col] = np.sqrt(df[col])
        elif chosen_method == "cbrt":
            df[col] = np.cbrt(df[col])
        elif chosen_method == "boxcox":
            # Boxcox requires strictly positive values
            df[col], _ = boxcox(df[col] + 1e-6)
        elif chosen_method == "yeojohnson":
            df[col], _ = yeojohnson(df[col])
        elif chosen_method == "inverse":
            df[col] = 1.0 / (df[col] + 1e-6)
        elif chosen_method == "power":
            k = power if power is not None else 0.5  # Default to sqrt if not specified
            df[col] = np.power(df[col], k)
        elif chosen_method == "winsor":
            lower = df[col].quantile(winsor_limits[0])
            upper = df[col].quantile(1 - winsor_limits[1])
            df[col] = np.clip(df[col], lower, upper)
        else:
            raise ValueError(f"Unknown normalization/transformation method: {chosen_method}")

    return df