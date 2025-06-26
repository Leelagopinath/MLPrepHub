# /home/leelagopinath/MLPrepHub/src/application/app.py

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
import pandas as pd

st.set_page_config(page_title="MLPrepHub", layout="wide")

def main():
    section_header("🤖 MLPrepHub: Universal ML Preprocessing System")
    state = get_session_state()

    # Sidebar: Interface selection
    interface = st.sidebar.selectbox("🔀 Choose Interface", ["Admin Interface", "Client Interface"])

    # Sidebar: Step selection
    steps = ["Data Upload", "EDA1", "Preprocessing", "EDA2", 
             "Training" if interface=="Admin Interface" else "Prediction"]
    current_step = st.sidebar.radio("📑 Workflow Step", steps)

    # Show progress indicator
    step_idx = steps.index(current_step) + 1
    st.sidebar.markdown(f"**Step {step_idx} of {len(steps)}**")

    # Prepare raw_df and processed_df in state
    raw_df = state.get("raw_df")  # may be None
    processed_df = state.get("processed_df")  # may be None

    # --- Step: Data Upload ---
    if current_step == "Data Upload":
        st.header("1️⃣ Data Upload")
        df = load_data_ui()
        if df is not None:
            # If new or changed dataset, reset downstream state
            reset = False
            if raw_df is None:
                reset = True
            else:
                # Compare shape/columns to detect change
                try:
                    if not (df.shape == raw_df.shape and list(df.columns) == list(raw_df.columns)):
                        reset = True
                except:
                    reset = True
            if reset:
                state["raw_df"] = df.copy()
                state["processed_df"] = None
                # Clear any preprocessing state inside run_preprocessing_ui
                # E.g., state.pop("original_df", None), pop target, applied_steps, etc.
                for key in ["original_df", "processed_df", "applied_steps", "target_col", "preprocessing_done"]:
                    state.pop(key, None)
                st.success("📥 New dataset loaded and session state reset.")
            else:
                st.info("📂 Same dataset reloaded; keeping previous preprocessing state.")
            st.dataframe(df.head())
        else:
            st.info("Please upload a dataset to proceed with EDA and preprocessing.")
        return

    # For other steps, ensure raw_df exists
    if raw_df is None:
        st.warning("⚠️ Please upload a dataset first in 'Data Upload' step.")
        return

    # --- Step: EDA1 ---
    if current_step == "EDA1":
        st.header("2️⃣ Exploratory Data Analysis (Raw Data)")
        try:
            run_eda1(raw_df, state)
        except Exception as e:
            st.error(f"Error during EDA1: {e}")
        return

    # --- Step: Preprocessing ---
    if current_step == "Preprocessing":
        st.header("3️⃣ Data Preprocessing")
        try:
            result = run_preprocessing_ui(raw_df, state)
            # If user clicked Finish inside run_preprocessing_ui, it returns processed_df
            if result is not None:
                # Update processed_df in session
                state["processed_df"] = result.copy()
                processed_df = state["processed_df"]
                st.success("✅ Preprocessing result stored in session.")
                st.dataframe(processed_df.head())
            else:
                # If they haven’t finished or are mid-steps, processed_df may still be previous
                if processed_df is not None:
                    st.info("Showing last processed data preview:")
                    st.dataframe(processed_df.head())
                else:
                    st.info("No preprocessing applied yet; raw data will be used downstream.")
        except Exception as e:
            st.error(f"Error during Preprocessing: {e}")
        return

    # Determine which dataframe to use for EDA2 / Training
    df_for_next = processed_df if processed_df is not None else raw_df

    # --- Step: EDA2 ---
    if current_step == "EDA2":
        st.header("4️⃣ Exploratory Data Analysis (Processed Data)")
        st.info("Using processed data (or raw if no preprocessing applied).")
        try:
            run_eda2(df_for_next, state)
        except Exception as e:
            st.error(f"Error during EDA2: {e}")
        return

    # --- Step: Training or Prediction ---
    if current_step == "Training" and interface == "Admin Interface":
        st.header("5️⃣ Model Training & Serialization")
        st.info("Using processed data (or raw if no preprocessing applied).")
        try:
            admin_training_ui(df_for_next, state)
        except Exception as e:
            st.error(f"Error in Training UI: {e}")
        return

    if current_step == "Prediction" and interface == "Client Interface":
        st.header("5️⃣ Client Prediction")
        st.info("Using processed data (or raw if no preprocessing applied).")
        try:
            client_prediction_ui(df_for_next, state)
        except Exception as e:
            st.error(f"Error in Prediction UI: {e}")
        return

    # Fallback (should not reach here)
    st.error("Invalid step or interface configuration.")

if __name__ == "__main__":
    main()
