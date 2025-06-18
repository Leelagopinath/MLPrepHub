import streamlit as st
import pandas as pd
from src.eda.visualization import (plot_histograms,plot_boxplots,plot_correlation_heatmap,plot_missing_matrix,plot_pairplot)

def run_eda2(df: pd.DataFrame, state: dict):
    """Main EDA-2 interface for processed data analysis"""
    st.subheader("📊 EDA-2: Pre-Processed Data Visualization")

    # Display dataset overview
    st.markdown("### 🧾 Pre-Processed Dataset Overview")
    st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
    
    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    }))

    # Visualizations section
    st.markdown("---")
    st.markdown("### 📈 Visualizations on Pre-Processed Dataset")

    # Expandable sections for each visualization
    with st.expander("📊 Histograms"):
        plot_histograms(df)

    with st.expander("📦 Boxplots"):
        plot_boxplots(df)

    with st.expander("📉 Missing Value Matrix"):
        plot_missing_matrix(df)

    with st.expander("🔗 Correlation Heatmap"):
        plot_correlation_heatmap(df)

    with st.expander("🧩 Pair Plot"):
        plot_pairplot(df)

    # Approval checkbox and state management
    approved = st.checkbox("✅ Approve this preprocessed dataset for training")
    
    if approved:
        state["eda2_approved"] = True
        st.success("Dataset approved for model training! ✨")
    
    return df if approved else None