import json
import os
from typing import Dict, List

def save_pipeline_step(pipeline_dir: str, dataset_name: str, step_record: dict) -> str:
    """
    Appends a preprocessing step to a JSON pipeline file for the given dataset.
    If the file does not exist, it creates a new one.
    Returns the path to the pipeline file.
    """
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline_file = os.path.join(pipeline_dir, f"{dataset_name}_pipeline.json")

    if os.path.exists(pipeline_file):
        with open(pipeline_file, "r") as f:
            pipeline = json.load(f)
    else:
        pipeline = []

    pipeline.append(step_record)

    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f, indent=4)
    return pipeline_file

def load_pipeline_steps(pipeline_dir: str, dataset_name: str) -> List[dict]:
    """
    Load the pipeline steps for a given dataset from its JSON file.
    """
    pipeline_file = os.path.join(pipeline_dir, f"{dataset_name}_pipeline.json")
    if os.path.exists(pipeline_file):
        with open(pipeline_file, "r") as f:
            return json.load(f)
    return []