# File: src/storage/dataset_repository.py

import os
import pandas as pd
from datetime import datetime

DATA_DIR = "./project_data/processed"

def generate_versioned_filename(base_name: str, ext: str = "csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_name = f"{base_name}_{timestamp}.{ext}"
    return os.path.join(DATA_DIR, versioned_name)

def save_dataset(df: pd.DataFrame, base_name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = generate_versioned_filename(base_name)
    df.to_csv(file_path, index=False)
    return file_path

def list_datasets() -> list:
    os.makedirs(DATA_DIR, exist_ok=True)
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def load_dataset(file_name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(path)
