# utils.py

import numpy as np

def detect_task_type(X_train, y_train):
    # Enhanced task type detection
    if isinstance(X_train, np.ndarray):
        if len(X_train.shape) == 4:
            return 'image_classification' if len(np.unique(y_train)) > 1 else 'image_regression'
        elif len(X_train.shape) == 3:
            return 'sequence_modeling'
        elif len(X_train.shape) == 2:
            if len(np.unique(y_train)) > 1:
                return 'tabular_classification'
            else:
                return 'tabular_regression'
    elif isinstance(X_train[0], str):
        return 'text_classification'
    else:
        return 'unknown'

def get_input_size(X_train):
    # Determines the input size for models
    if isinstance(X_train, np.ndarray):
        return X_train.shape[1]
    else:
        # For sequences or other types
        return None
