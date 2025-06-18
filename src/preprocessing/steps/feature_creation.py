import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def apply_feature_creation(df: pd.DataFrame, columns: list, degree: int = 2, interaction_only: bool = False) -> pd.DataFrame:
    df_copy = df.copy()
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df_copy[columns])
    feature_names = poly.get_feature_names_out(columns)
    
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_copy.index)
    df_copy = df_copy.drop(columns=columns).join(poly_df)
    return df_copy
