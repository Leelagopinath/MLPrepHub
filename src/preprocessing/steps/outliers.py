# src/preprocessing/steps/outliers.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

# Autoencoder imports are inside function to avoid forcing dependencies unless used
# You need tensorflow (or keras) installed for autoencoder detection
# e.g., pip install tensorflow

def detect_outliers(
    df: pd.DataFrame,
    columns: list = None,
    method: str = "iqr",
    contamination: float = 0.05,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    autoencoder_params: dict = None
) -> dict:
    """
    Detect outliers in the DataFrame using specified method.
    Returns a dict with:
      {
        "method": method,
        "columns": [...],
        "timestamp": "...Z",
        "outlier_indices": [...],
        "num_outliers": X
      }
    Supported methods:
      - 'boxplot'       : detect via IQR thresholds (same as 'iqr')
      - 'iqr'           : Interquartile Range
      - 'zscore'        : |z| > 3 by default
      - 'isolation_forest'
      - 'lof'           : Local Outlier Factor
      - 'elliptic'      : Elliptic Envelope
      - 'dbscan'        : DBSCAN clustering noise points (labels == -1)
      - 'autoencoder'   : Deep-learning autoencoder reconstruction error
    Notes:
      * columns: list of numeric columns; if None, defaults to all numeric columns.
      * contamination: proportion of outliers, used in some methods to decide threshold.
      * For 'dbscan', uses dbscan_eps and dbscan_min_samples.
      * For 'autoencoder', autoencoder_params can include:
            {
              "hidden_layers": [...],  # e.g., [64, 32, 64]
              "epochs": 20,
              "batch_size": 32,
              "threshold_method": "quantile",  # or "std"
              "threshold_param": 0.95          # for quantile, top 5% errors; for std, mean+param*std
            }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Select numeric columns by default
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    if not columns:
        return {"method": method, "columns": [], "timestamp": datetime.utcnow().isoformat() + "Z",
                "outlier_indices": [], "num_outliers": 0}

    arr = df[columns].values
    outlier_indices = set()

    method_lower = method.lower()
    if method_lower in ["boxplot", "iqr"]:
        # For each column, detect outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        for col in columns:
            series = df[col].dropna()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (df[col] < lower) | (df[col] > upper)
            indices = df.index[mask].tolist()
            outlier_indices.update(indices)

    elif method_lower == "zscore":
        # Compute zscore per column; use threshold 3.0
        zs = df[columns].apply(lambda x: (x - x.mean())/x.std(ddof=0))
        # For each row, if any column |z| > 3
        mask = zs.abs() > 3
        indices = df.index[mask.any(axis=1)].tolist()
        outlier_indices.update(indices)

    elif method_lower == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        # Drop NaNs for fitting; but we align indices: fill NaN temporarily?
        subset = df[columns].dropna()
        preds = model.fit_predict(subset)
        # preds == -1 indicates outlier
        out_idx = subset.index[preds == -1].tolist()
        outlier_indices.update(out_idx)

    elif method_lower == "lof":
        # LOF does not accept missing; drop NaNs
        subset = df[columns].dropna()
        model = LocalOutlierFactor(contamination=contamination)
        preds = model.fit_predict(subset)
        out_idx = subset.index[preds == -1].tolist()
        outlier_indices.update(out_idx)

    elif method_lower == "elliptic":
        subset = df[columns].dropna()
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        preds = model.fit(subset).predict(subset)
        out_idx = subset.index[preds == -1].tolist()
        outlier_indices.update(out_idx)

    elif method_lower == "dbscan":
        # DBSCAN on numeric columns: noise labeled -1
        subset = df[columns].dropna()
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = clustering.fit_predict(subset)
        out_idx = subset.index[labels == -1].tolist()
        outlier_indices.update(out_idx)

    elif method_lower == "autoencoder":
        # Deep-learning based detection via reconstruction error
        # Requires tensorflow.keras
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("tensorflow is required for autoencoder-based outlier detection")

        # Prepare data: drop NaNs
        subset = df[columns].dropna()
        data = subset.values.astype(float)
        # Scale data to [0,1] or standardize? Here we standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Default autoencoder params
        params = {
            "hidden_layers": [int(data.shape[1] / 2), int(data.shape[1] / 4), int(data.shape[1] / 2)],
            "epochs": 20,
            "batch_size": 32,
            "threshold_method": "quantile",
            "threshold_param": 1 - contamination  # keep top contamination as outliers
        }
        if autoencoder_params:
            params.update(autoencoder_params)

        # Build simple autoencoder
        input_dim = data_scaled.shape[1]
        inp = Input(shape=(input_dim,))
        x = inp
        # Encoder
        for units in params["hidden_layers"][:len(params["hidden_layers"])//2]:
            x = Dense(units, activation="relu")(x)
        # Bottleneck if odd/even? Here last of first half
        # Decoder
        for units in reversed(params["hidden_layers"][:len(params["hidden_layers"])//2]):
            x = Dense(units, activation="relu")(x)
        x = Dense(input_dim, activation=None)(x)
        autoencoder = Model(inputs=inp, outputs=x)
        autoencoder.compile(optimizer=Adam(), loss="mse")
        # Train
        autoencoder.fit(data_scaled, data_scaled,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        verbose=0)
        # Compute reconstruction errors
        reconstructed = autoencoder.predict(data_scaled, verbose=0)
        mse = np.mean(np.square(data_scaled - reconstructed), axis=1)
        # Determine threshold
        if params["threshold_method"] == "quantile":
            thresh = np.quantile(mse, params["threshold_param"])
        else:
            mean = mse.mean()
            std = mse.std()
            thresh = mean + params.get("threshold_param", 3) * std
        # Outliers where mse > thresh
        out_positions = np.where(mse > thresh)[0]
        out_idx = subset.index[out_positions].tolist()
        outlier_indices.update(out_idx)

    else:
        raise ValueError(
            f"Unknown detection method: {method}. Supported: boxplot, iqr, zscore, "
            "isolation_forest, lof, elliptic, dbscan, autoencoder"
        )

    outlier_list = sorted(outlier_indices)
    info = {
        "method": method,
        "columns": columns.copy(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "outlier_indices": outlier_list,
        "num_outliers": len(outlier_list)
    }
    return info


def remove_outliers(
    df: pd.DataFrame,
    columns: list = None,
    *,
    method: str = "iqr",
    contamination: float = 0.05,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    autoencoder_params: dict = None
) -> pd.DataFrame:
    """
    Detects and removes outliers from the DataFrame using the specified method.
    Signature unchanged except added optional parameters for detection.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to process. If None, uses all numeric columns.
        method (str): Method to use for outlier detection/removal.
                      One of: 'iqr', 'zscore', 'isolation_forest', 'lof', 'elliptic'
                      (same as detection methods but excluding 'boxplot','dbscan','autoencoder')
        contamination (float): Expected proportion of outliers.
        dbscan_eps, dbscan_min_samples: ignored here (dbscan not in removal options).
        autoencoder_params: ignored here (autoencoder not in removal options).

    Returns:
        pd.DataFrame: DataFrame with outliers removed (index reset).
    """
    # Only allow removal methods: IQR, zscore, isolation_forest, lof, elliptic
    # Map synonyms
    method_lower = method.lower()
    allowed = {
        "iqr": "iqr",
        "zscore": "zscore",
        "isolation_forest": "isolation_forest",
        "lof": "lof",
        "elliptic": "elliptic"
    }
    if method_lower not in allowed:
        raise ValueError(
            f"Unknown removal method: {method}. Supported removal methods: {list(allowed.keys())}"
        )

    # Use detect_outliers internally
    info = detect_outliers(df, columns=columns, method=method_lower, contamination=contamination)
    outlier_indices = info["outlier_indices"]

    if not outlier_indices:
        # Nothing to remove
        return df.copy()

    # Drop and reset index
    df_cleaned = df.drop(index=pd.Index(outlier_indices)).reset_index(drop=True)
    # Optionally log: number removed
    # e.g., print or logger; here we simply return cleaned DF
    # print(f"Removed {len(outlier_indices)} outliers out of {len(df)} rows.")
    return df_cleaned
