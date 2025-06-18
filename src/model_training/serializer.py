import joblib
import pickle
import onnx
import onnxruntime as ort
import torch
import os

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def serialize_model(model, output_path, formats, X_sample):
    for fmt in formats:
        path = os.path.join(output_path, f"model.{fmt}")
        if fmt == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        elif fmt == 'joblib':
            joblib.dump(model, path)
        elif fmt == 'onnx':
            initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        elif fmt == 'h5':
            # Simplified: placeholder for future neural models
            torch.save(model, path)