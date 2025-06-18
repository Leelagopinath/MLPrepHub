import os
import streamlit as st
from contextlib import contextmanager
import time

@contextmanager
def loading_button(message: str = "Processing..."):
    """
    Context manager to show loading state while processing.
    
    Usage:
        with loading_button("Saving..."):
            # do some work
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(message)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        yield
    finally:
        progress_bar.empty()
        status_text.empty()

def tick_animation(success_message: str):
    """
    Show a completion animation with checkmark and message.
    
    Args:
        success_message (str): Message to show after completion
    """
    with st.spinner('Processing...'):
        time.sleep(0.5)  # Brief pause for effect
    st.success(f"✅ {success_message}")
    time.sleep(0.5)  # Let success message show briefly

def section_header(title: str):
    """Display a section header with styling."""
    st.markdown(f"## {title}")
    st.markdown("---")

# ...existing code...
def get_latest_version_path(base_path: str) -> str:
    """
    Returns the latest versioned folder from a base path like ./project_data/processed/
    """
    versions = [
        int(d.replace("v", ""))
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    if not versions:
        return os.path.join(base_path, "v1")
    latest_version = f"v{max(versions)}"
    return os.path.join(base_path, latest_version)

def increment_version(base_path: str) -> str:
    """
    Creates a new versioned folder by incrementing the current highest version number.
    Returns the path to the new version folder.
    """
    versions = [
        int(d.replace("v", ""))
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    next_version = 1 if not versions else max(versions) + 1
    new_folder = os.path.join(base_path, f"v{next_version}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def list_files_with_extension(directory: str, extensions: list) -> list:
    """
    Returns a list of files in the given directory that have any of the specified extensions.
    
    Args:
        directory (str): Path to directory to search
        extensions (list): List of file extensions to filter by (e.g. ['.pkl', '.joblib'])
    
    Returns:
        list: List of matching filenames
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []
        
    return [
        f for f in os.listdir(directory)
        if any(f.endswith(ext) for ext in extensions)
    ]

def section_header(title: str):
    """Display a section header in Streamlit with consistent styling."""
    import streamlit as st
    st.markdown(f"## {title}")