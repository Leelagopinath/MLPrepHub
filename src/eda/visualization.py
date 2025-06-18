import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import missingno as msno

def plot_histograms(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            st.pyplot(fig)
            plt.close()

def plot_boxplots(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, y=col)
            plt.title(f'Boxplot of {col}')
            st.pyplot(fig)
            plt.close()

def plot_missing_matrix(df: pd.DataFrame):
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No missing values found in the dataset.")

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)
        plt.close()

def plot_pairplot(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        cols = numeric_df.columns[:5]  # First 5 numeric columns
        fig = sns.pairplot(df[cols])
        st.pyplot(fig)
        plt.close()