import os
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from src.storage.dataset_repository import save_dataset

def read_dataset(file) -> pd.DataFrame:
    filename = file.name
    ext = filename.split('.')[-1].lower()

    if ext == "csv":
        return pd.read_csv(file)
    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(file)
    elif ext == "json":
        return pd.read_json(file)
    elif ext == "parquet":
        return pd.read_parquet(file)
    elif ext == "xml":
        return pd.read_xml(file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def read_from_url(url: str) -> pd.DataFrame:
    parsed = urlparse(url)
    if "drive.google.com" in parsed.netloc:
        file_id = parsed.path.split('/')[-2]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_csv(download_url)
    elif "github.com" in parsed.netloc:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return pd.read_csv(raw_url)
    else:
        return pd.read_csv(url)

def upload_dataset_ui():
    st.subheader("📂 Upload Dataset")

    option = st.radio("Select Data Source", ["Upload from local", "From URL (Google Drive / GitHub)"])
    df = None
    dataset_name = ""

    if option == "Upload from local":
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xls", "xlsx", "json", "parquet", "xml"])
        if uploaded_file:
            df = read_dataset(uploaded_file)
            dataset_name = uploaded_file.name

    elif option == "From URL (Google Drive / GitHub)":
        url = st.text_input("Enter file URL")
        if st.button("Load from URL") and url:
            try:
                df = read_from_url(url)
                dataset_name = os.path.basename(urlparse(url).path)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    if df is not None:
        st.success("✅ Dataset loaded successfully")
        st.write(df.head())
        save_dataset(df, dataset_name)
    
    return df  # <-- This line is essential!

def load_data(file) -> pd.DataFrame:
    """
    Unified function to load data from a file object.
    """
    return read_dataset(file)