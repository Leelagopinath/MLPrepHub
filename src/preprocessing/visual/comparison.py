import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

def plot_boxplot_comparison(before: pd.DataFrame, after: pd.DataFrame, col: str, title: str = None):
    """Create side-by-side boxplots for before/after comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.boxplot(data=before, y=col, ax=ax1, color='skyblue')
    ax1.set_title('Before')
    
    sns.boxplot(data=after, y=col, ax=ax2, color='salmon')
    ax2.set_title('After')
    
    if title:
        fig.suptitle(title)
    st.pyplot(fig)

def plot_heatmap_comparison(before: pd.DataFrame, after: pd.DataFrame, title: str = None):
    """Create side-by-side correlation heatmaps for before/after comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(before.corr(), ax=ax1, cmap='coolwarm', center=0)
    ax1.set_title('Correlations (Before)')
    sns.heatmap(after.corr(), ax=ax2, cmap='coolwarm', center=0)
    ax2.set_title('Correlations (After)')
    if title:
        fig.suptitle(title)
    st.pyplot(fig)

def plot_histogram_comparison(before: pd.DataFrame, after: pd.DataFrame, col: str, title: str = None):
    """Create side-by-side histograms for before/after comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.histplot(data=before, x=col, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Before')
    
    sns.histplot(data=after, x=col, kde=True, ax=ax2, color='salmon')
    ax2.set_title('After')
    
    if title:
        fig.suptitle(title)
    st.pyplot(fig)

def plot_scatter_comparison(before: pd.DataFrame, after: pd.DataFrame, n_components: int = 2, 
                          title: str = None, label_col: str = None):
    """Create scatter plots for comparing before/after (e.g., for PCA/LDA results)."""
    if n_components < 2:
        st.warning("Need at least 2 components for scatter plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before plot
    if label_col and label_col in before.columns:
        scatter = ax1.scatter(before.iloc[:, 0], before.iloc[:, 1], 
                            c=before[label_col], cmap='viridis')
        plt.colorbar(scatter, ax=ax1)
    else:
        ax1.scatter(before.iloc[:, 0], before.iloc[:, 1], alpha=0.5)
    ax1.set_title('Before')
    
    # After plot
    if label_col and label_col in after.columns:
        scatter = ax2.scatter(after.iloc[:, 0], after.iloc[:, 1],
                            c=after[label_col], cmap='viridis')
        plt.colorbar(scatter, ax=ax2)
    else:
        ax2.scatter(after.iloc[:, 0], after.iloc[:, 1], alpha=0.5)
    ax2.set_title('After')
    
    if title:
        fig.suptitle(title)
    st.pyplot(fig)

def show_before_after_plots(before_df: pd.DataFrame, after_df: pd.DataFrame, columns: list):
    st.markdown("### 🔍 Before vs After Comparison")

    if not columns:
        st.warning("⚠️ No columns selected for visualization.")
        return

    for col in columns:
        if col not in before_df.columns or col not in after_df.columns:
            st.warning(f"⚠️ Column '{col}` not found in one of the datasets.")
            continue

        if not pd.api.types.is_numeric_dtype(before_df[col]):
            st.info(f"ℹ️ Skipping non-numeric column: `{col}`")
            continue

        st.markdown(f"## 📊 Column: `{col}`")

        ### 🔸 Histogram Row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📘 Histogram (Before)**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(before_df[col], kde=True, color='salmon', ax=ax)
            ax.set_title("Before")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**📙 Histogram (After)**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(after_df[col], kde=True, color='skyblue', ax=ax)
            ax.set_title("After")
            st.pyplot(fig)
            plt.close(fig)

        ### 🔸 Boxplot Row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📘 Boxplot (Before)**")
            fig, ax = plt.subplots(figsize=(5, 1.5))
            sns.boxplot(x=before_df[col], color='salmon', ax=ax)
            ax.set_title("Before")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**📙 Boxplot (After)**")
            fig, ax = plt.subplots(figsize=(5, 1.5))
            sns.boxplot(x=after_df[col], color='skyblue', ax=ax)
            ax.set_title("After")
            st.pyplot(fig)
            plt.close(fig)

        ### 🔸 Violin Plot Row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📘 Violin Plot (Before)**")
            fig, ax = plt.subplots(figsize=(5, 2))
            sns.violinplot(x=before_df[col], color='salmon', ax=ax)
            ax.set_title("Before")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**📙 Violin Plot (After)**")
            fig, ax = plt.subplots(figsize=(5, 2))
            sns.violinplot(x=after_df[col], color='skyblue', ax=ax)
            ax.set_title("After")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")