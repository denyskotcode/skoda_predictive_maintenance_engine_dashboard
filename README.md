# 🔧 Škoda Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-189c3a?style=flat)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-4EA94B?style=flat)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-orange?style=flat)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end ML system for predicting CNC machine failures in manufacturing environments.
Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
from UCI Machine Learning Repository.

The system ingests live sensor readings (temperature, rotational speed, torque, tool wear),
estimates failure probability in real time, identifies the most likely failure mode out of five types,
and explains each prediction using SHAP values — all wrapped in an interactive Streamlit dashboard.

> Built as a portfolio project demonstrating production-ready ML engineering skills:
> data pipeline → feature engineering → model comparison → explainable AI → web app.

---

## ✨ What It Does

| Feature | Details |
|---|---|
| **Real-time risk score** | Failure probability (0–100%) with color-coded LOW / MEDIUM / HIGH badge |
| **Failure mode classification** | 5-class prediction: TWF, HDF, PWF, OSF, RNF |
| **Explainable AI** | Per-prediction SHAP waterfall chart built with Plotly |
| **Sensor data explorer** | Histograms, scatter plots, correlation heatmap for the full dataset |
| **Model comparison** | ROC curves, PR curves, confusion matrix for all 4 models |
| **Imbalance handling** | `class_weight='balanced'` + `scale_pos_weight` for ~3.4% failure rate |
| **Simulation buttons** | "Random State" and "Critical State" for live demo |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/denyskotcode/skoda_predictive_maintenance_engine_dashboard.git
cd skoda_predictive_maintenance_engine_dashboard

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
#    Downloads AI4I 2020 dataset automatically (or generates synthetic data).
#    Takes ~2 minutes. Saves 7 .pkl artifacts to models/
python train.py

# 5. Launch dashboard
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

Trained on 8000 samples, evaluated on 2000. Class imbalance (~3.4% failures) handled explicitly.

| Model | Accuracy | Precision (Failure) | Recall (Failure) | F1 (Failure) | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.70 | ~0.72 | ~0.71 | ~0.97 |
| Random Forest | ~0.98 | ~0.80 | ~0.78 | ~0.79 | ~0.99 |
| **XGBoost** ⭐ | **~0.98** | **~0.82** | **~0.80** | **~0.81** | **~0.99** |
| LightGBM | ~0.98 | ~0.81 | ~0.79 | ~0.80 | ~0.99 |

> Exact metrics are printed by `python train.py` and shown in the **Model Performance** tab of the dashboard.

---

## 🗂️ Project Structure

```
skoda-predictive-maintenance/
├── data/
│   └── ai4i2020.csv              # AI4I 2020 dataset (auto-downloaded on first run)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Load, clean, feature engineering, train/test split
│   ├── model_training.py         # Train & compare 4 ML models
│   └── explainability.py         # SHAP TreeExplainer integration
├── models/
│   ├── binary_classifier.pkl     # Best binary model (XGBoost)
│   ├── multiclass_classifier.pkl # LightGBM failure-type classifier
│   ├── label_encoder.pkl
│   ├── scaler.pkl                # StandardScaler fitted on training data
│   ├── feature_names.pkl
│   ├── model_comparison.pkl      # Metrics + ROC/PR curves for all 4 models
│   └── shap_explainer.pkl        # SHAP TreeExplainer
├── notebooks/
│   └── predictive_maintenance_eda.ipynb  # EDA notebook (Kaggle-ready)
├── streamlit_app.py              # Main interactive dashboard
├── train.py                      # Full training pipeline
├── requirements.txt
└── .streamlit/config.toml        # Škoda green color theme
```

---

## ⚙️ How It Works

```
ai4i2020.csv
    │
    ▼
load_and_clean_data()       — drop UDI/ProductID, rename columns to snake_case
    │
    ▼
engineer_features()         — create temp_diff_k, power_w, torque_wear_product,
    │                          speed_torque_ratio, quality dummies
    ▼
prepare_datasets()          — StandardScaler, 80/20 stratified split
    │
    ├──▶ train_binary_classifier()     — LogReg, RF, XGBoost, LightGBM
    │         best model by F1(failure class)
    │
    ├──▶ train_multiclass_classifier() — LightGBM, 6 classes
    │
    └──▶ create_shap_explainer()       — SHAP TreeExplainer on best model
             │
             ▼
         Save 7 .pkl artifacts → Streamlit loads them at startup
             │
             ▼
         User enters sensor values → scale → predict → SHAP → render
```

**Engineered features:**

| Feature | Formula | Why |
|---|---|---|
| `temp_diff_k` | process_temp − air_temp | Heat dissipation indicator (HDF trigger) |
| `power_w` | torque × rpm × 2π/60 | Mechanical power in watts (PWF trigger) |
| `torque_wear_product` | torque × tool_wear | Overstrain proxy (OSF trigger) |
| `speed_torque_ratio` | rpm / torque | Efficiency indicator |
| `is_low_quality` | 1 if Type == 'L' | Lower quality → stricter OSF threshold |

---

## 🔧 Failure Modes

| Code | Name | Trigger |
|---|---|---|
| **TWF** | Tool Wear Failure | Tool wear 200–240 min |
| **HDF** | Heat Dissipation Failure | Temp diff < 8.6 K **and** speed < 1380 rpm |
| **PWF** | Power Failure | Power < 3,500 W or > 9,000 W |
| **OSF** | Overstrain Failure | Torque × wear > quality-specific threshold |
| **RNF** | Random Failure | 0.1% random probability |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| ML Models | scikit-learn · XGBoost · LightGBM |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Plotly |
| Data | Pandas · NumPy |

---

## 📄 Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: UCI Machine Learning Repository (ID 601) / Kaggle
- 10,000 manufacturing observations from a CNC milling machine
- License: CC BY 4.0
- Citation: Matzka, S. (2020). *Explainable AI for Predictive Maintenance*. AI4I 2020.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
Dataset: AI4I 2020 — CC BY 4.0 (Stephan Matzka, 2020)
