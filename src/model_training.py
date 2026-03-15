"""
Model training module — binary and multiclass failure classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder


def train_binary_classifier(X_train, y_train, X_test, y_test) -> dict:
    """
    Train and compare binary classifiers for failure prediction.
    Both models use class_weight='balanced' for imbalanced data.
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    models_def = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', random_state=42, max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='logloss',
            verbosity=0, use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            is_unbalance=True, n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42, verbose=-1
        ),
    }

    fitted_models, metrics, roc_curves, pr_curves = {}, {}, {}, {}

    for name, model in models_def.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_failure': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall_failure': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'f1_failure': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_curves[name] = (prec, rec)

    best_name = max(metrics, key=lambda n: metrics[n]['f1_failure'])
    return {
        'models': fitted_models,
        'metrics': metrics,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves,
        'best_model_name': best_name,
        'best_model': fitted_models[best_name]
    }


def train_multiclass_classifier(X_train, y_train, X_test, y_test) -> dict:
    """
    LightGBM multiclass classifier for failure type prediction.
    Classes: No Failure, TWF, HDF, PWF, OSF, RNF
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    class_names = le.classes_.tolist()

    model = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, verbose=-1, class_weight='balanced',
        objective='multiclass', num_class=len(class_names)
    )

    print(f"  Training LightGBM multiclass ({len(class_names)} classes)...")
    model.fit(X_train, y_train_enc)

    y_pred = le.inverse_transform(model.predict(X_test))
    y_test_orig = le.inverse_transform(y_test_enc)

    return {
        'model': model,
        'label_encoder': le,
        'classification_report': classification_report(y_test_orig, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_orig, y_pred, labels=class_names),
        'class_names': class_names
    }
