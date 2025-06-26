import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np

from src.eda.visualization import (
    plot_histograms,
    plot_boxplots,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_missing_matrix
)

def run_eda1(df: pd.DataFrame, state: dict):
    st.subheader("📊 EDA-1: Exploratory Data Analysis on Raw Dataset")

    # 1. Dataset Overview in columns
    st.markdown("### 🧾 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Dataframe info in expandable section
    with st.expander("Show Detailed Column Info", expanded=False):
        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        }))

    # 2. Data Distributions
    st.markdown("### 📈 Data Distributions")
    
    # Numeric Distributions - 2 columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for i in range(0, len(numeric_cols), 2):
        col1, col2 = st.columns(2)
        with col1:
            if i < len(numeric_cols):
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(df[numeric_cols[i]], kde=True, ax=ax)
                plt.title(f'Distribution: {numeric_cols[i]}')
                st.pyplot(fig)
                plt.close()
        with col2:
            if i + 1 < len(numeric_cols):
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(df[numeric_cols[i + 1]], kde=True, ax=ax)
                plt.title(f'Distribution: {numeric_cols[i + 1]}')
                st.pyplot(fig)
                plt.close()

    # Boxplots - 2 columns
    st.markdown("### 📦 Box Plots")
    for i in range(0, len(numeric_cols), 2):
        col1, col2 = st.columns(2)
        with col1:
            if i < len(numeric_cols):
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(y=df[numeric_cols[i]], ax=ax)
                plt.title(f'{numeric_cols[i]}')
                st.pyplot(fig)
                plt.close()
        with col2:
            if i + 1 < len(numeric_cols):
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(y=df[numeric_cols[i + 1]], ax=ax)
                plt.title(f'{numeric_cols[i + 1]}')
                st.pyplot(fig)
                plt.close()

    # Categorical value counts - 3 columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.markdown("### 📊 Categorical Variables")
        for i in range(0, len(categorical_cols), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(categorical_cols):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        df[categorical_cols[i + j]].value_counts().head().plot(kind='bar')
                        plt.title(f'{categorical_cols[i + j]} (Top 5)')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()

    # 3. Correlation Heatmap
    st.markdown("### 📊 Data Quality Insights")
    
    with st.expander("Correlation Heatmap", expanded=False):
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Dynamic font size calculation
            base_font_size = 10
            # Reduce font size by 0.4 for every column beyond 10
            font_size = max(6, base_font_size - 0.4 * max(0, len(numeric_cols) - 10))
            
            sns.heatmap(df[numeric_cols].corr(), 
                       annot=True, 
                       cmap='coolwarm', 
                       fmt='.2f',
                       annot_kws={'size': font_size},  # Apply dynamic font size
                       ax=ax)
            plt.title("Understanding Multicollinearity between Features")
            st.pyplot(fig)
            plt.close()
    
    # 4. Missing Values Matrix
    with st.expander("Missing Values Pattern", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        msno.matrix(df, ax=ax)
        plt.title("Missing Values Matrix")
        st.pyplot(fig)
        plt.close()

    # Approval section
    st.markdown("---")
    approved = st.checkbox("✅ Approve this raw dataset for preprocessing", 
                         key="eda1_approved_checkbox")
    if approved:
        state["eda1_approved"] = True
        st.success("EDA-1 approved! You can proceed to preprocessing.")