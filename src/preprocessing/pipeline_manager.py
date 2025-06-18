import pandas as pd
import json
import os
from datetime import datetime
from src.utils.helpers import get_latest_version_path, increment_version
from src.storage.dataset_repository import save_dataset
from src.storage.pipeline_repository import save_pipeline_step
class PreprocessingPipelineManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_data_path = f'project_data/raw/{dataset_name}.csv'
        self.processed_dir = f'project_data/processed/'
        self.pipeline_dir = f'project_data/pipelines/'
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.pipeline_dir, exist_ok=True)

    def load_latest_data(self) -> pd.DataFrame:
        path = get_latest_version_path(self.processed_dir, self.dataset_name)
        return pd.read_csv(path) if path else pd.read_csv(self.raw_data_path)

    def save_processed_data(self, df: pd.DataFrame, step_name: str, params: dict):
        version_path = increment_version(self.processed_dir, self.dataset_name, suffix="_preprocessed")
        df.to_csv(version_path, index=False)

        step_record = {
            "step": step_name,
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "saved_path": version_path
        }

        save_pipeline_step(self.pipeline_dir, self.dataset_name, step_record)

    def get_pipeline_summary(self) -> list:
        summary_file = os.path.join(self.pipeline_dir, f"{self.dataset_name}_pipeline.json")
        if not os.path.exists(summary_file):
            return []
        with open(summary_file, "r") as f:
            return json.load(f)

    def add_step(self, step_info: dict):
        """Add a preprocessing step to the pipeline."""
        summary = self.get_pipeline_summary()
        summary.append(step_info)
        self._save_pipeline_summary(summary)
        return step_info

    def undo_last_step(self) -> dict:
        """Remove and return the last step from the pipeline."""
        summary = self.get_pipeline_summary()
        if not summary:
            return None
        last_step = summary.pop()
        self._save_pipeline_summary(summary)
        return last_step

    def _save_pipeline_summary(self, summary: list):
        """Save the pipeline summary to JSON."""
        summary_file = os.path.join(self.pipeline_dir, f"{self.dataset_name}_pipeline.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

# Create module-level functions that work with a global pipeline manager
_pipeline_manager = None

def init_pipeline_manager(dataset_name: str):
    """Initialize the global pipeline manager."""
    global _pipeline_manager
    _pipeline_manager = PreprocessingPipelineManager(dataset_name)

def add_step(state: dict, step_info: dict) -> dict:
    """Add a preprocessing step to the current pipeline."""
    if _pipeline_manager is None:
        init_pipeline_manager(state.get("dataset_name", "unnamed"))
    return _pipeline_manager.add_step(step_info)

def undo_last_step() -> dict:
    """Remove and return the last step from the current pipeline."""
    if _pipeline_manager is None:
        return None
    return _pipeline_manager.undo_last_step()

def get_applied_steps() -> list:
    """Get all steps applied in the current pipeline."""
    if _pipeline_manager is None:
        return []
    return _pipeline_manager.get_pipeline_summary()