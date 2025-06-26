import pandas as pd
import numpy as np
from scipy.stats import boxcox, yeojohnson, skew

def detect_best_transform(series):
    """
    Suggest the best transformation to reduce skewness for a numeric pandas Series.
    Returns (best_method, original_skew, new_skew).
    """
    vals = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.size < 3 or vals.nunique() < 2:
        return None, None, None

    orig_skew = skew(vals)
    best_method = None
    best_skew = abs(orig_skew)
    new_skew = orig_skew

    # Define candidate transformations
    def safe_log(x):
        x = x - x.min() + 1 if (x <= 0).any() else x
        return np.log1p(x)

    def safe_boxcox(x):
        x = x - x.min() + 1 if (x <= 0).any() else x
        return boxcox(x)[0]

    def safe_yeojohnson(x):
        return yeojohnson(x)[0]

    methods = {
        "log": safe_log,
        "boxcox": safe_boxcox,
        "yeojohnson": safe_yeojohnson,
    }

    for name, func in methods.items():
        try:
            transformed = func(vals)
            s = skew(transformed)
            if np.isfinite(s) and abs(s) < best_skew - 0.1:  # Only update if improvement is meaningful
                best_skew = abs(s)
                best_method = name
                new_skew = s
        except Exception:
            continue

    return best_method, orig_skew, new_skew

def apply_log_transform(df: pd.DataFrame, columns: list, method: str = "log1p") -> pd.DataFrame:
    """
    Apply specified transformation to selected columns.
    Supported methods: 'log', 'log10', 'log2', 'log1p', 'boxcox', 'yeojohnson', 'auto'.
    If method='auto', applies log1p when skew > 1, else leaves column unchanged.
    """
    df_copy = df.copy()
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not in DataFrame")
        series = df_copy[col].replace([np.inf, -np.inf], np.nan)
        if method == "log":
            vals = series - series.min() + 1 if (series <= 0).any() else series
            df_copy[col] = np.log1p(vals)
        elif method == "log10":
            vals = series - series.min() + 1 if (series <= 0).any() else series
            df_copy[col] = np.log10(vals)
        elif method == "log2":
            vals = series - series.min() + 1 if (series <= 0).any() else series
            df_copy[col] = np.log2(vals)
        elif method == "log1p":
            vals = series - series.min() + 1 if (series <= 0).any() else series
            df_copy[col] = np.log1p(vals)
        elif method == "boxcox":
            arr = series.values.astype(float)
            arr = arr - np.nanmin(arr) + 1 if (arr <= 0).any() else arr
            transformed, _ = boxcox(arr)
            df_copy[col] = transformed
        elif method == "yeojohnson":
            arr = series.values.astype(float)
            transformed, _ = yeojohnson(arr)
            df_copy[col] = transformed
        elif method == "auto":
            vals = series.dropna()
            if vals.empty:
                continue
            s = skew(vals)
            if s > 1:
                vals = vals - vals.min() + 1 if (vals <= 0).any() else vals
                df_copy[col] = np.log1p(vals)
            else:
                continue
        else:
            raise ValueError(f"Unsupported log transformation method: {method}")
    return df_copy