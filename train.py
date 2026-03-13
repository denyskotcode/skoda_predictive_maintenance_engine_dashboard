"""
Training pipeline for Škoda Predictive Maintenance System.
Run: python train.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ai4i2020.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_or_generate_data(filepath: str) -> None:
    """Try UCI download first, fall back to synthetic generation."""
    if os.path.exists(filepath):
        print(f"[DATA] Dataset already at {filepath}")
        return

    print("[DATA] Trying to download AI4I 2020 from UCI ML Repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=601)
        df = dataset.data.original
        df.to_csv(filepath, index=False)
        print(f"[DATA] Downloaded {len(df)} records → {filepath}")
        return
    except Exception as e:
        print(f"[DATA] Download failed ({e}). Generating synthetic data...")

    generate_synthetic_data(filepath)


def generate_synthetic_data(filepath: str) -> None:
    """
    Generate realistic synthetic dataset mimicking AI4I 2020.
    10000 records, ~3.4% failure rate.
    """
    np.random.seed(42)
    n = 10_000

    udi = np.arange(1, n + 1)
    type_choices = np.random.choice(['L', 'M', 'H'], size=n, p=[0.5, 0.3, 0.2])
    product_id = [f"{t}{s}" for t, s in zip(type_choices, np.arange(1, n + 1))]

    air_temp = np.random.normal(300, 2, n).clip(295, 305)
    process_temp = (air_temp + 10 + np.random.normal(0, 1, n)).clip(305, 315)
    torque = np.random.normal(40, 10, n).clip(3, 80)

    power_base = 2860
    rotational_speed = (
        power_base * 60 / (2 * np.pi * torque) + np.random.normal(0, 50, n)
    ).clip(1000, 2900).astype(int)

    tool_wear = np.random.randint(0, 251, n)

    temp_diff = process_temp - air_temp
    power_w = torque * rotational_speed * 2 * np.pi / 60

    twf = ((tool_wear >= 200) & (tool_wear <= 240) &
           (np.random.random(n) < 0.05)).astype(int)
    hdf = ((temp_diff < 8.6) & (rotational_speed < 1380) &
           (np.random.random(n) < 0.25)).astype(int)
    pwf = (((power_w < 3500) | (power_w > 9000)) &
           (np.random.random(n) < 0.20)).astype(int)

    torque_wear = torque * tool_wear
    threshold_osf = np.where(
        type_choices == 'L', 11000,
        np.where(type_choices == 'M', 12000, 13000)
    )
    osf = ((torque_wear > threshold_osf) &
           (np.random.random(n) < 0.20)).astype(int)
    rnf = (np.random.random(n) < 0.001).astype(int)

    machine_failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)

    # boost to ~3.4% if rate is too low
    if machine_failure.mean() < 0.025:
        need = int(0.034 * n) - machine_failure.sum()
        zero_idx = np.where(machine_failure == 0)[0]
        flip_idx = np.random.choice(zero_idx, size=min(need, len(zero_idx)), replace=False)
        machine_failure[flip_idx] = 1
        rnf[flip_idx] = 1

    df = pd.DataFrame({
        'UDI': udi,
        'Product ID': product_id,
        'Type': type_choices,
        'Air temperature [K]': air_temp.round(1),
        'Process temperature [K]': process_temp.round(1),
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque.round(1),
        'Tool wear [min]': tool_wear,
        'Machine failure': machine_failure,
        'TWF': twf,
        'HDF': hdf,
        'PWF': pwf,
        'OSF': osf,
        'RNF': rnf
    })

    df.to_csv(filepath, index=False)
    failures = machine_failure.sum()
    print(f"[DATA] Generated {n} records, {failures} failures ({failures/n*100:.1f}%) → {filepath}")


def main():
    print("=" * 55)
    print("Škoda Predictive Maintenance — Training Pipeline")
    print("=" * 55)

    download_or_generate_data(DATA_PATH)

    # TODO: add preprocessing and training steps next
    df = pd.read_csv(DATA_PATH)
    print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Failure rate: {df['Machine failure'].mean()*100:.2f}%")
    print("\n[INFO] Data acquisition complete. Preprocessing coming next.")


if __name__ == '__main__':
    main()
