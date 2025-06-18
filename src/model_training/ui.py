import streamlit as st
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from .trainer import train_model
from .evaluator import evaluate_model
from .serializer import serialize_model
from .model_selector import select_model, select_task_type
import streamlit as st
import pandas as pd
from typing import Dict
import os
from src.data.loader import read_dataset  # Now this import will work
from src.inference.predictor import load_model, predict_with_model
from src.utils.helpers import list_files_with_extension


SPLIT_DIR = "project_data/splits"
MODEL_DIR = "project_data/models"


def model_training_ui(processed_df, target_column):
    st.subheader("\U0001F4BB Model Training")

    if st.button("\U0001F3CB\ufe0f Start Training Pipeline"):
        with st.spinner("Splitting dataset into train and test sets..."):
            X = processed_df.drop(columns=[target_column])
            y = processed_df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pd.DataFrame(X_train).to_csv(f"{SPLIT_DIR}/X_train.csv", index=False)
            pd.DataFrame(X_test).to_csv(f"{SPLIT_DIR}/X_test.csv", index=False)
            pd.DataFrame(y_train).to_csv(f"{SPLIT_DIR}/y_train.csv", index=False)
            pd.DataFrame(y_test).to_csv(f"{SPLIT_DIR}/y_test.csv", index=False)
            time.sleep(1)
            st.success("Splitting has done \u2705")

    task_type = select_task_type()
    model_name = select_model()

    if model_name:
        with st.spinner("Selecting model..."):
            time.sleep(1)
            st.success("Ready for Training \u2705")

    if st.button("\U0001F680 Start Training"):
        with st.spinner("Training model..."):
            X_train = pd.read_csv(f"{SPLIT_DIR}/X_train.csv")
            y_train = pd.read_csv(f"{SPLIT_DIR}/y_train.csv").values.ravel()
            model = train_model(model_name, X_train, y_train)
            joblib.dump(model, f"{MODEL_DIR}/latest_model.pkl")
            time.sleep(1)
            st.success("Training completed \u2705")

    if st.button("\U0001F52C Test Model"):
        model = joblib.load(f"{MODEL_DIR}/latest_model.pkl")
        X_test = pd.read_csv(f"{SPLIT_DIR}/X_test.csv")
        y_test = pd.read_csv(f"{SPLIT_DIR}/y_test.csv").values.ravel()
        results = evaluate_model(model, X_test, y_test, task_type)
        st.json(results)

    formats = st.multiselect("Choose Serialization Formats", ['pkl', 'joblib', 'onnx', 'h5'])
    if st.button("\U0001F4BE Serialize Model"):
        model = joblib.load(f"{MODEL_DIR}/latest_model.pkl")
        X_sample = pd.read_csv(f"{SPLIT_DIR}/X_test.csv").iloc[:1]  # For ONNX input shape
        with st.spinner("Serializing model..."):
            serialize_model(model, MODEL_DIR, formats, X_sample)
            time.sleep(1)
            st.success("Model serialized \u2705")

def admin_training_ui(df: pd.DataFrame, state: Dict):
    st.subheader("🔧 Model Training Interface")
    # Add your training interface code here
    st.write("Training interface coming soon...")

import streamlit as st
import pandas as pd
from typing import Dict

def client_prediction_ui(df: pd.DataFrame, state: Dict):
    st.subheader("🔮 Prediction Interface")
    # Add your prediction interface code here
    st.write("Prediction interface coming soon...")