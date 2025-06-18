# src/data/saver.py
import os
import pandas as pd
from datetime import datetime

PROCESSED_DIR = "project_data/processed"

def save_processed_dataset(df: pd.DataFrame, name: str = "processed") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return path
def save_processed_version(df: pd.DataFrame, state: dict, step: str = None) -> str:
    """
    Save an intermediate version of the processed DataFrame with step information.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        state (dict): Application state containing dataset info
        step (str, optional): Name of the preprocessing step. Defaults to None.
    
    Returns:
        str: Path where the file was saved
    """
    # Get dataset name from state or use default
    dataset_name = state.get("dataset_name", "unnamed")
    
    # Create version string with timestamp and step
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_str = f"{timestamp}"
    if step:
        version_str = f"{step}_{version_str}"
    
    # Create filename
    filename = f"{dataset_name}_{version_str}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    
    # Ensure directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(path, index=False)
    
    # Log saving operation if needed
    print(f"Saved processed version to: {path}")
    
    return path

def get_latest_version(dataset_name: str) -> str:
    """Get the path to the latest version of a processed dataset."""
    if not os.path.exists(PROCESSED_DIR):
        return None
        
    files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith(dataset_name)]
    if not files:
        return None
        
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(PROCESSED_DIR, x)))
    return os.path.join(PROCESSED_DIR, latest)