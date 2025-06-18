import streamlit as st
from src.application.navigation import setup_sidebar
from src.model_training.ui import admin_training_ui
from src.inference.ui import client_prediction_ui
from src.data.loader import upload_dataset_ui as load_data_ui
from src.eda.eda1_raw import run_eda1
from src.eda.eda2_processed import run_eda2
from src.preprocessing.visual.step_ui import run_preprocessing_ui
from src.state.manager import get_session_state
from src.utils.helpers import section_header

st.set_page_config(page_title="MLPrepHub", layout="wide")

def main():
    section_header("🤖 MLPrepHub: Universal ML Preprocessing System")
    interface = setup_sidebar()
    state = get_session_state()

    if interface == "Admin Interface":
        run_admin_pipeline(state)
    elif interface == "Client Interface":
        run_client_pipeline(state)

def run_admin_pipeline(state):
    st.subheader("🧑‍💼 Admin Workflow")
    df = load_data_ui()

    if df is not None:
        st.info("Scroll down to see EDA visualizations. Approve at the bottom to proceed.")
        run_eda1(df, state)

    # Only proceed if EDA1 is approved and df is not None
    if state.get("eda1_approved") and df is not None:
        processed_df = run_preprocessing_ui(df, state)
        # Only proceed if preprocessing is done and processed_df is not None
        if processed_df is not None and state.get("preprocessing_done"):
            run_eda2(processed_df, state)
            if state.get("eda2_approved"):
                admin_training_ui(processed_df, state)

def run_client_pipeline(state):
    st.subheader("🙋‍♂️ Client Prediction Workflow")
    df = load_data_ui()

    if df is not None:
        st.info("Scroll down to see EDA visualizations. Approve at the bottom to proceed.")
        run_eda1(df, state)

    if state.get("eda1_approved") and df is not None:
        processed_df = run_preprocessing_ui(df, state)
        if processed_df is not None and state.get("preprocessing_done"):
            run_eda2(processed_df, state)
            if state.get("eda2_approved"):
                client_prediction_ui(processed_df, state)

if __name__ == "__main__":
    main()