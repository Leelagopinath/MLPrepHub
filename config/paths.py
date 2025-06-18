# MLPrepHub/config/paths.py

import os

# Root project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'project_data/raw/')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'project_data/processed/')
EDA_REPORTS_DIR = os.path.join(BASE_DIR, 'project_data/eda_reports/')
MODELS_DIR = os.path.join(BASE_DIR, 'project_data/models/')
SPLITS_DIR = os.path.join(BASE_DIR, 'project_data/splits/')
STATE_DIR = os.path.join(BASE_DIR, 'project_data/state/')

# Create directories if not present
def ensure_dirs():
    dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, EDA_REPORTS_DIR,
        MODELS_DIR, SPLITS_DIR, STATE_DIR
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

ensure_dirs()
