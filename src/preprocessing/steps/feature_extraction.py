import pandas as pd
from sklearn.decomposition import PCA

def apply_feature_extraction(df: pd.DataFrame, columns: list, n_components: int = 2) -> pd.DataFrame:
    df_copy = df.copy()
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_copy[columns])
    for i in range(components.shape[1]):
        df_copy[f'pca_component_{i+1}'] = components[:, i]
    return df_copy.drop(columns=columns)
