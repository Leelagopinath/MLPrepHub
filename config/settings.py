# MLPrepHub/config/settings.py

import os

# General Settings
APP_NAME = "MLPrepHub"
APP_MODE = "admin"  # or "client" – can be controlled via UI

# Preprocessing
DEFAULT_SPLIT_RATIO = 0.8
SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'json', 'parquet', 'sql', 'xml', 'hdf5']
SUPPORTED_SERIALIZATION_FORMATS = ['pkl', 'joblib', 'onnx', 'h5']

# Visualization
MAX_PLOTS = 30  # limit plots for large datasets

# Model Configuration
DEFAULT_RANDOM_STATE = 42
SUPPORTED_CLASSIFIERS = [
    "Logistic Regression", "Decision Trees", "Random Forest",
    "Support Vector Machine", "Naive Bayes", "KNN"
]
SUPPORTED_REGRESSORS = [
    "Linear Regression", "Decision Trees", "Random Forest",
    "Support Vector Machine", "KNN"
]

# UI Settings
SHOW_ANIMATIONS = True
USE_CACHE = True
ENABLE_DEBUG_MODE = False
