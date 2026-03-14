"""
Data preprocessing for Škoda Predictive Maintenance.
Loads, cleans, engineers features and prepares train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV, drop non-informative columns, rename to snake_case.
    """
    df = pd.read_csv(filepath)

    # Drop identifier columns — not useful for the model
    cols_to_drop = [c for c in ['UDI', 'UID', 'Product ID', 'udi', 'uid', 'product_id']
                    if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Rename columns to clean snake_case
    rename_map = {}
    for col in df.columns:
        c = col.strip()
        if c in ('Air temperature [K]', 'Air temperature'):
            rename_map[col] = 'air_temp_k'
        elif c in ('Process temperature [K]', 'Process temperature'):
            rename_map[col] = 'process_temp_k'
        elif c in ('Rotational speed [rpm]', 'Rotational speed'):
            rename_map[col] = 'rotational_speed_rpm'
        elif c in ('Torque [Nm]', 'Torque'):
            rename_map[col] = 'torque_nm'
        elif c in ('Tool wear [min]', 'Tool wear'):
            rename_map[col] = 'tool_wear_min'
        elif c in ('Machine failure', 'machine_failure'):
            rename_map[col] = 'machine_failure'
        elif c in ('Type', 'type'):
            rename_map[col] = 'type'

    df = df.rename(columns=rename_map)

    required = [
        'type', 'air_temp_k', 'process_temp_k', 'rotational_speed_rpm',
        'torque_nm', 'tool_wear_min', 'machine_failure',
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}. Got: {df.columns.tolist()}")

    df = df.dropna(subset=required)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from sensor readings.

    - temp_diff_k: temperature differential (heat dissipation indicator)
    - power_w: mechanical power in watts
    - torque_wear_product: overstrain proxy
    - speed_torque_ratio: efficiency indicator
    - is_low_quality, is_high_quality: product quality dummies
    """
    df = df.copy()

    df['temp_diff_k'] = df['process_temp_k'] - df['air_temp_k']
    df['power_w'] = df['torque_nm'] * df['rotational_speed_rpm'] * 2 * np.pi / 60
    df['torque_wear_product'] = df['torque_nm'] * df['tool_wear_min']
    df['speed_torque_ratio'] = df['rotational_speed_rpm'] / (df['torque_nm'] + 1e-6)
    df['is_low_quality'] = (df['type'] == 'L').astype(int)
    df['is_high_quality'] = (df['type'] == 'H').astype(int)

    df = df.drop(columns=['type'])
    return df


def _get_failure_type(row: pd.Series) -> str:
    for col, label in [('TWF', 'TWF'), ('HDF', 'HDF'), ('PWF', 'PWF'), ('OSF', 'OSF'), ('RNF', 'RNF')]:
        if row[col] == 1:
            return label
    return 'No Failure'


def prepare_datasets(df: pd.DataFrame):
    """
    Split into features/targets, scale, train/test split (80/20, stratified).

    Returns:
        X_train, X_test, y_train_binary, y_test_binary,
        y_train_multi, y_test_multi, scaler, feature_names
    """
    df = df.copy()
    df['failure_type'] = df.apply(_get_failure_type, axis=1)

    y_binary = df['machine_failure'].astype(int)
    y_multiclass = df['failure_type']

    drop_cols = ['machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'failure_type']
    X = df.drop(columns=drop_cols)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train_b, y_test_b, y_train_m, y_test_m = train_test_split(
        X, y_binary, y_multiclass,
        test_size=0.2, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    return X_train_s, X_test_s, y_train_b, y_test_b, y_train_m, y_test_m, scaler, feature_names
