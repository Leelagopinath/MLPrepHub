# src/data/version_manager.py
import os

PROCESSED_DIR = "project_data/processed"

def list_versions():
    if not os.path.exists(PROCESSED_DIR):
        return []
    return sorted(os.listdir(PROCESSED_DIR), reverse=True)

def get_latest_version():
    versions = list_versions()
    return os.path.join(PROCESSED_DIR, versions[0]) if versions else None
