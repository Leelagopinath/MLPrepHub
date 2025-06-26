import pandas as pd
import numpy as np
import streamlit as st

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import FastICA
from scipy.fft import fft
from scipy.stats import boxcox, probplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _find_corr_blocks(df: pd.DataFrame, thresh: float):
    corr = df.corr().abs()
    visited = set()
    blocks = []
    for col in corr.columns:
        if col in visited: 
            continue
        group = set([col] + list(corr.index[corr[col] > thresh]))
        if len(group) > 1:
            blocks.append(list(group))
        visited |= group
    return blocks


def apply_feature_extraction(
    df: pd.DataFrame,
    method: str,
    n_components: int = 2,
    columns: list = None,
    label_col: str = None,
    text_col: str = None,
    autoencoder_model=None,
    **kwargs
) -> pd.DataFrame:
    """
    Apply various feature extraction techniques and visualize results.
    Supported methods: PCA, t-SNE, LDA, Autoencoders, TF-IDF, Word2Vec, Fourier, DWT, Feature Hashing
    """
    df_copy = df.copy()
    result_df = df_copy.copy()
    method = method.lower()

    # Select columns
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        if label_col and label_col in columns:
            columns.remove(label_col)

    # Standardize numeric data for most methods
    X = df_copy[columns].values if columns else df_copy.select_dtypes(include=[np.number]).values
    X_scaled = StandardScaler().fit_transform(X)

    # --- NEW: multicol_pca ---
    if method == "multicol_pca":
        corr_thresh = kwargs.get("corr_thresh", 0.8)
        var_retained = kwargs.get("variance_retained", 0.95)

        blocks = _find_corr_blocks(df_copy[columns], corr_thresh)
        # drop all block columns
        to_drop = [c for block in blocks for c in block]
        result_df = df_copy.drop(columns=to_drop)

        for i, block in enumerate(blocks):
            sub = df_copy[block].values
            sub_scaled = StandardScaler().fit_transform(sub)
            pca = PCA(n_components=var_retained)
            comps = pca.fit_transform(sub_scaled)
            comp_names = [f"MC_PCA_{i+1}_{j+1}" for j in range(comps.shape[1])]
            pcs = pd.DataFrame(comps, columns=comp_names, index=df_copy.index)
            result_df = pd.concat([result_df, pcs], axis=1)

        st.markdown(f"#### Applied block‐wise PCA on {len(blocks)} groups (ρ>{corr_thresh}), kept {var_retained*100:.0f}% variance")
        return result_df

    # --- NEW: vif-based selection ---
    if method == "vif":
        vif_thresh = kwargs.get("vif_thresh", 10.0)
        Xv = df_copy[columns].copy()
        while True:
            Xc = sm.add_constant(Xv)
            vifs = pd.Series(
                [variance_inflation_factor(Xc.values, idx+1) for idx in range(Xv.shape[1])],
                index=Xv.columns
            )
            max_vif = vifs.max()
            if max_vif <= vif_thresh:
                break
            drop = vifs.idxmax()
            Xv = Xv.drop(columns=[drop])
        result_df = pd.concat([df_copy.drop(columns=columns), Xv], axis=1)
        st.markdown(f"#### Dropped features until VIF ≤ {vif_thresh}: kept {len(Xv.columns)}/{len(columns)} columns")
        return result_df


    # --- Feature Extraction ---
    if method == "pca":
        # Add component validation
        n_components = min(n_components, len(columns), len(df))
        model = PCA(n_components=n_components)
        components = model.fit_transform(X_scaled)
        comp_cols = [f"PCA_{i+1}" for i in range(n_components)]
        df_components = pd.DataFrame(components, columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=columns), df_components], axis=1)
        # Scree plot
        st.markdown("#### Scree Plot (PCA Explained Variance)")
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(model.explained_variance_ratio_)+1), model.explained_variance_ratio_, marker='o')
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        st.pyplot(fig)
    elif method == "tsne":
        model = TSNE(n_components=n_components, random_state=42)
        components = model.fit_transform(X_scaled)
        comp_cols = [f"tSNE_{i+1}" for i in range(n_components)]
        df_components = pd.DataFrame(components, columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=columns), df_components], axis=1)
    elif method == "lda":
        if label_col is None or label_col not in df_copy.columns:
            raise ValueError("label_col must be specified for LDA.")
        y = df_copy[label_col]
        model = LDA(n_components=n_components)
        components = model.fit_transform(X_scaled, y)
        comp_cols = [f"LDA_{i+1}" for i in range(n_components)]
        df_components = pd.DataFrame(components, columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=columns), df_components], axis=1)
    elif method == "autoencoder":
        if autoencoder_model is None:
            raise ValueError("Provide a trained autoencoder model.")
        encoded = autoencoder_model.predict(X_scaled)
        comp_cols = [f"AE_{i+1}" for i in range(encoded.shape[1])]
        df_components = pd.DataFrame(encoded, columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=columns), df_components], axis=1)

    elif method == "tfidf":
        if text_col not in df_copy.columns:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame")
        if not pd.api.types.is_string_dtype(df_copy[text_col]):
            df_copy[text_col] = df_copy[text_col].astype(str)
            st.warning(f"Converted '{text_col}' to string for text processing")
        
        vectorizer = TfidfVectorizer(max_features=n_components)
        tfidf_matrix = vectorizer.fit_transform(df_copy[text_col])  # Removed .astype(str)
        comp_cols = [f"TFIDF_{i+1}" for i in range(tfidf_matrix.shape[1])]
        df_components = pd.DataFrame(tfidf_matrix.toarray(), columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=[text_col]), df_components], axis=1)
        

    elif method == "feature_hashing":
        if text_col not in df_copy.columns:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame")
        if not pd.api.types.is_string_dtype(df_copy[text_col]):
            df_copy[text_col] = df_copy[text_col].astype(str)
            st.warning(f"Converted '{text_col}' to string for text processing")
        
        vectorizer = HashingVectorizer(n_features=n_components)
        hashed_matrix = vectorizer.fit_transform(df_copy[text_col])  # Removed .astype(str)
        comp_cols = [f"Hash_{i+1}" for i in range(hashed_matrix.shape[1])]
        df_components = pd.DataFrame(hashed_matrix.toarray(), columns=comp_cols, index=df_copy.index)
        result_df = pd.concat([df_copy.drop(columns=[text_col]), df_components], axis=1)
    elif method == "fourier":
        # Apply FFT to each numeric column
        fft_features = []
        for col in columns:
            fft_vals = np.abs(fft(df_copy[col].values))
            fft_features.append(fft_vals[:n_components])
        fft_features = np.array(fft_features).T
        comp_cols = [f"FFT_{i+1}" for i in range(n_components)]
        df_components = pd.DataFrame(fft_features, columns=comp_cols, index=df_copy.index[:fft_features.shape[0]])
        result_df = pd.concat([df_copy, df_components], axis=1)
    
    elif method == "pca_correlated_groups":
        result_df, reduction_info = reduce_collinearity_pca(
            df_copy, 
            threshold=kwargs.get('correlation_threshold', 0.8),
            explained_variance=kwargs.get('explained_variance', 0.95),
            target_col=label_col
        )
        visualize_collinearity_reduction(reduction_info)
        
    elif method == "vif_reduction":
        result_df, vif_info = reduce_collinearity_vif(
            df_copy,
            threshold=kwargs.get('vif_threshold', 5.0),
            target_col=label_col
        )
        visualize_vif_reduction(vif_info)

    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")

    # --- Visualization Options ---
    st.markdown("#### Feature Extraction Visualizations")
    plot_type = st.selectbox(
        "Choose Visualization",
        [
            "Scree Plot",
            "2D Scatter",
            "3D Scatter",
            "Heatmap",
            "Box Plot",
            "Q-Q Plot"
        ],
        key=f"viz_{method}"
    )

    # Scree Plot (for PCA)
    if plot_type == "Scree Plot" and method == "pca":
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(model.explained_variance_ratio_)+1), model.explained_variance_ratio_, marker='o')
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        st.pyplot(fig)

    # 2D Scatter
    if plot_type == "2D Scatter" and n_components >= 2:
        fig, ax = plt.subplots()
        ax.scatter(result_df.iloc[:, -2], result_df.iloc[:, -1], alpha=0.7)
        ax.set_xlabel(result_df.columns[-2])
        ax.set_ylabel(result_df.columns[-1])
        st.pyplot(fig)

    # 3D Scatter
    if plot_type == "3D Scatter" and n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result_df.iloc[:, -3], result_df.iloc[:, -2], result_df.iloc[:, -1], alpha=0.7)
        ax.set_xlabel(result_df.columns[-3])
        ax.set_ylabel(result_df.columns[-2])
        ax.set_zlabel(result_df.columns[-1])
        st.pyplot(fig)

    # Heatmap (Correlation Matrix)
    if plot_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(result_df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Box Plot
    if plot_type == "Box Plot":
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=result_df.select_dtypes(include=[np.number]), ax=ax)
        st.pyplot(fig)

    # Q-Q Plot (for last component)
    if plot_type == "Q-Q Plot":
        from scipy.stats import probplot
        fig, ax = plt.subplots()
        probplot(result_df.iloc[:, -1], dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {result_df.columns[-1]}")
        st.pyplot(fig)

    return result_df

def visualize_collinearity_reduction(reduction_info: dict):
    """Visualize the collinearity reduction process"""
    st.markdown("#### Collinearity Reduction Report")
    
    # Display correlation matrix heatmap
    st.markdown("**Correlation Matrix (|ρ| > threshold)**")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(reduction_info['correlation_matrix'], annot=True, fmt=".2f", 
                cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Show correlated groups
    st.markdown("**Identified Correlated Groups**")
    for i, group in enumerate(reduction_info['correlated_groups']):
        st.write(f"Group {i+1}: {', '.join(group)}")
    
    # Show transformation summary
    st.markdown("**Transformation Summary**")
    st.write(f"Original features: {len(reduction_info['original_features'])}")
    st.write(f"Removed features: {len(reduction_info['removed_features'])}")
    st.write(f"New PCA components: {len(reduction_info['new_components'])}")
    st.write(f"Net change: {len(reduction_info['new_components']) - len(reduction_info['removed_features'])} features")

def visualize_vif_reduction(vif_info: dict):
    """Visualize the VIF reduction process"""
    st.markdown("#### VIF Reduction Report")
    
    # Show initial vs final VIF distribution
    st.markdown("**VIF Distribution Comparison**")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initial VIF
    ax1.bar(vif_info['initial_vif'].keys(), vif_info['initial_vif'].values())
    ax1.axhline(y=vif_info['threshold'], color='r', linestyle='--')
    ax1.set_title("Initial VIF Scores")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # Final VIF
    ax2.bar(vif_info['final_vif'].keys(), vif_info['final_vif'].values())
    ax2.axhline(y=vif_info['threshold'], color='r', linestyle='--')
    ax2.set_title("Final VIF Scores")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show removal process
    st.markdown("**Removed Features**")
    if vif_info['removed_features']:
        st.write(f"Removed {len(vif_info['removed_features'])} features with VIF > {vif_info['threshold']}:")
        for feature in vif_info['removed_features']:
            st.write(f"- {feature}")
    else:
        st.success("No features needed to be removed!")