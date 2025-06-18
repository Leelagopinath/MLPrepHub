import pandas as pd
import joblib
import onnxruntime as rt
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


def load_model(model_path):
    ext = os.path.splitext(model_path)[1]
    if ext == ".pkl" or ext == ".joblib":
        return joblib.load(model_path), ext
    elif ext == ".onnx":
        session = rt.InferenceSession(model_path)
        return session, ext
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def predict_with_model(model, model_format, input_df):
    if model_format in [".pkl", ".joblib"]:
        return model.predict(input_df)

    elif model_format == ".onnx":
        input_name = model.get_inputs()[0].name
        input_data = input_df.astype(np.float32).to_numpy()
        pred = model.run(None, {input_name: input_data})
        return np.argmax(pred[0], axis=1)
    else:
        raise ValueError("Unsupported model format")
