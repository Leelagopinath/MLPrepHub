# File: src/state/persistence.py

import os
import json

STATE_DIR = "./project_data/state/"

def save_state(session_id: str, state: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(f"{STATE_DIR}{session_id}.json", "w") as f:
        json.dump(state, f)

def load_state(session_id: str) -> dict:
    try:
        with open(f"{STATE_DIR}{session_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
