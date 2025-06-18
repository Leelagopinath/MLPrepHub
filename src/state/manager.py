# File: src/state/manager.py

from typing import Any, Dict
import streamlit as st

class SessionStateManager:
    """
    Manages session state variables in Streamlit.
    """

    @staticmethod
    def initialize(defaults: Dict[str, Any]):
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get(key: str, default=None) -> Any:
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        st.session_state[key] = value

    @staticmethod
    def clear():
        st.session_state.clear()


def get_session_state():
    """Returns the Streamlit session state dictionary."""
    return st.session_state