"""
SHAP-based explainability for the best binary classifier.
"""

import numpy as np


def create_shap_explainer(model, X_train):
    """
    Create SHAP TreeExplainer for the best binary model.
    Uses subsample of 500 rows for speed.

    Returns:
        (explainer, shap_values) — shap_values is a 2D numpy array
    """
    import shap

    X_sample = X_train.values[:500] if hasattr(X_train, 'values') else X_train[:500]

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)

    # Binary models return list [neg_class, pos_class] — take positive class
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    return explainer, shap_values


def get_shap_for_single_prediction(explainer, X_single):
    """
    SHAP values for a single observation (used in the Streamlit app).

    Args:
        X_single: shape (1, n_features) array or DataFrame

    Returns:
        1D numpy array of SHAP values
    """
    import shap

    X_arr = X_single.values if hasattr(X_single, 'values') else np.array(X_single)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    shap_values_raw = explainer.shap_values(X_arr)

    if isinstance(shap_values_raw, list):
        return shap_values_raw[1][0]
    return shap_values_raw[0]
