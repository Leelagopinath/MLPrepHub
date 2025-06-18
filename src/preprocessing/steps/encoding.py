import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def apply_encoding(df: pd.DataFrame, columns: list, method: str = "label") -> pd.DataFrame:
    df_copy = df.copy()

    if method == "label":
        for col in columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    elif method == "onehot":
        df_copy = pd.get_dummies(df_copy, columns=columns)

    elif method == "ordinal":
        oe = OrdinalEncoder()
        df_copy[columns] = oe.fit_transform(df_copy[columns].astype(str))

    else:
        raise ValueError("Unsupported encoding method")

    return df_copy
