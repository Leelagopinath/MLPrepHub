import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def apply_feature_selection(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()]
    return df[selected_columns]
