
import numpy as np

def modal_absolute_error(y_true, y_mode_pred):
    return float(np.mean(np.abs(y_true - y_mode_pred)))

def modal_squared_error(y_true, y_mode_pred):
    return float(np.mean((y_true - y_mode_pred)**2))
