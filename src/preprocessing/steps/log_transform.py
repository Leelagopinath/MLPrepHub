import pandas as pd
import numpy as np
from scipy.stats import boxcox, yeojohnson

def apply_log_transform(df: pd.DataFrame, columns: list, method: str = "log1p") -> pd.DataFrame:
    df_copy = df.copy()
    
    for col in columns:
        if method == "log":
            df_copy[col] = np.log(df_copy[col].replace(0, np.nan))
        elif method == "log10":
            df_copy[col] = np.log10(df_copy[col].replace(0, np.nan))
        elif method == "log2":
            df_copy[col] = np.log2(df_copy[col].replace(0, np.nan))
        elif method == "log1p":
            df_copy[col] = np.log1p(df_copy[col])
        elif method == "boxcox":
            df_copy[col], _ = boxcox(df_copy[col] + 1e-6)  # Shift to make positive
        elif method == "yeojohnson":
            df_copy[col], _ = yeojohnson(df_copy[col])
        else:
            raise ValueError("Unsupported log transformation method")

    return df_copy
