"""
Full training pipeline for Škoda Predictive Maintenance System.

Steps:
1. Download/generate AI4I 2020 dataset
2. Preprocess and engineer features
3. Train + compare 4 binary classifiers
4. Train multiclass failure-type classifier
5. Create SHAP explainer for best model
6. Save all artifacts to models/

Run: python train.py
"""

import os
import sys
import pickle
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


def save_pkl(obj, filename: str) -> None:
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {filename} ({size_kb:.1f} KB)")


def main():
    print("=" * 55)
    print("Škoda Predictive Maintenance — Training Pipeline")
    print("=" * 55)

    # 1. Data
    download_or_generate_data(DATA_PATH)

    from src.data_preprocessing import load_and_clean_data, engineer_features, prepare_datasets
    from src.model_training import train_binary_classifier, train_multiclass_classifier
    from src.explainability import create_shap_explainer

    # 2. Preprocess
    print("\n[PREPROCESS] Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"  {len(df)} records")

    print("[PREPROCESS] Feature engineering...")
    df = engineer_features(df)

    print("[PREPROCESS] Splitting datasets...")
    (X_train, X_test, y_train_b, y_test_b,
     y_train_m, y_test_m, scaler, feature_names) = prepare_datasets(df)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Failure rate (train): {y_train_b.mean()*100:.1f}%")
    print(f"  Features: {feature_names}")

    # 3. Binary classifiers
    print("\n[TRAIN] Binary classifiers...")
    binary_results = train_binary_classifier(X_train, y_train_b, X_test, y_test_b)

    # 4. Multiclass classifier
    print("\n[TRAIN] Multiclass failure-type classifier...")
    multi_results = train_multiclass_classifier(X_train, y_train_m, X_test, y_test_m)

    # 5. SHAP
    print("\n[EXPLAIN] Creating SHAP explainer...")
    explainer, shap_values = create_shap_explainer(binary_results['best_model'], X_train)

    # 6. Print comparison table
    best_name = binary_results['best_model_name']
    print("\n" + "=" * 66)
    print("MODEL COMPARISON (Binary Classification)")
    print("=" * 66)
    print(f"{'Model':<22} {'Accuracy':>8} {'Prec(1)':>8} {'Rec(1)':>8} {'F1(1)':>8} {'AUC':>8}")
    print("-" * 66)
    for name, m in binary_results['metrics'].items():
        tag = " ← BEST" if name == best_name else ""
        print(f"{name:<22} {m['accuracy']:>8.4f} {m['precision_failure']:>8.4f} "
              f"{m['recall_failure']:>8.4f} {m['f1_failure']:>8.4f} {m['roc_auc']:>8.4f}{tag}")

    print(f"\nBest model: {best_name}")
    print(f"  Accuracy : {binary_results['metrics'][best_name]['accuracy']:.4f}")
    print(f"  F1(fail) : {binary_results['metrics'][best_name]['f1_failure']:.4f}")
    print(f"  ROC AUC  : {binary_results['metrics'][best_name]['roc_auc']:.4f}")

    print("\n[BINARY] Classification report:")
    print(binary_results['metrics'][best_name]['classification_report'])
    print("\n[MULTI] Classification report:")
    print(multi_results['classification_report'])

    # 7. Save artifacts
    print("\n[SAVE] Saving artifacts...")
    save_pkl(binary_results['best_model'], 'binary_classifier.pkl')
    save_pkl(multi_results['model'], 'multiclass_classifier.pkl')
    save_pkl(multi_results['label_encoder'], 'label_encoder.pkl')
    save_pkl(scaler, 'scaler.pkl')
    save_pkl(feature_names, 'feature_names.pkl')
    save_pkl(explainer, 'shap_explainer.pkl')
    save_pkl({
        'metrics': binary_results['metrics'],
        'roc_curves': binary_results['roc_curves'],
        'pr_curves': binary_results['pr_curves'],
        'best_model_name': best_name,
        'confusion_matrix': binary_results['metrics'][best_name]['confusion_matrix'],
        'multi_class_names': multi_results['class_names'],
        'multi_confusion_matrix': multi_results['confusion_matrix'],
        'feature_names': feature_names
    }, 'model_comparison.pkl')

    print("\n[DONE] All artifacts saved to models/")


if __name__ == '__main__':
    main()
