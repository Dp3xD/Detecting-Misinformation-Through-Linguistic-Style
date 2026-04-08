"""
xgboost_model.py
----------------
XGBoost classifier for misinformation detection using linguistic features.

Pipeline:
  1. Load LIAR dataset from HuggingFace and binarize labels
  2. Extract 7 linguistic features per sample
  3. Train XGBoost with early stopping on validation set
  4. Evaluate on test set (accuracy, F1, precision, recall)
  5. Save confusion matrix + feature importance chart to results/

Run independently:
    python models/xgboost_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import requests

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_extractor import extract_features_batch, FEATURE_NAMES, load_spacy_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# Absolute paths relative to this file's location
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
DATA_DIR    = os.path.join(_PROJECT_ROOT, "data")

# Raw TSV URLs (thiagorainmaker77/liar_dataset on GitHub)
LIAR_URLS = {
    "train":      "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    "validation": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
    "test":       "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
}

# String label → binary: credible=0, misinformation=1
CREDIBLE_LABELS = {"true", "mostly-true", "half-true"}
MISINFO_LABELS  = {"barely-true", "false", "pants-fire"}


# ---------------------------------------------------------------------------
# Label binarisation
# ---------------------------------------------------------------------------

def binarize_label(label: str) -> int:
    """Map LIAR string label → binary (0=credible, 1=misinformation)."""
    return 0 if label.strip() in CREDIBLE_LABELS else 1


# ---------------------------------------------------------------------------
# Data loading + feature extraction
# ---------------------------------------------------------------------------

def _load_liar_split(split_name: str) -> pd.DataFrame:
    """
    Download (or load from local cache) one LIAR TSV split.

    TSV columns (no header):
      0=id  1=label  2=statement  3=subject  4=speaker  5=job_title
      6=state  7=party  8=barely_true_count  9=false_count  10=half_true_count
      11=mostly_true_count  12=pants_fire_count  13=context
    """
    url        = LIAR_URLS[split_name]
    local_path = os.path.join(DATA_DIR, f"liar_{split_name}.tsv")

    if os.path.exists(local_path):
        return pd.read_csv(local_path, sep="\t", header=None)

    print(f"  Downloading {url} …")
    os.makedirs(DATA_DIR, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return pd.read_csv(local_path, sep="\t", header=None)


def load_and_prepare_data(force_recompute: bool = False):
    """
    Download LIAR TSVs, binarize string labels, and extract linguistic features.

    Feature matrices are cached as .npy files in data/ after the first run.
    Subsequent calls load from cache unless force_recompute=True.

    Parameters
    ----------
    force_recompute : if True, ignore any cached .npy files and recompute.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test  (np.ndarray pairs)
    """
    # Map internal split name → cache file stem
    cache_stems = {
        "train":      "features_train",
        "validation": "features_val",
        "test":       "features_test",
    }
    cache_paths = {
        name: (
            os.path.join(DATA_DIR, f"{stem}.npy"),
            os.path.join(DATA_DIR, f"{stem}_labels.npy"),
        )
        for name, stem in cache_stems.items()
    }

    # ── Cache hit ────────────────────────────────────────────────────────────
    all_cached = all(
        os.path.exists(x_path) and os.path.exists(y_path)
        for x_path, y_path in cache_paths.values()
    )
    if all_cached and not force_recompute:
        print("Loading cached features from data/")
        splits = {
            name: (np.load(x_path), np.load(y_path))
            for name, (x_path, y_path) in cache_paths.items()
        }
        return (
            splits["train"][0],      splits["train"][1],
            splits["validation"][0], splits["validation"][1],
            splits["test"][0],       splits["test"][1],
        )

    # ── Cache miss: download TSVs + extract features ──────────────────────────
    print("Loading LIAR dataset from GitHub TSVs...")
    os.makedirs(DATA_DIR, exist_ok=True)
    nlp    = load_spacy_model()
    splits = {}

    for split_name in ("train", "validation", "test"):
        print(f"\n--- Processing '{split_name}' split ---")
        df = _load_liar_split(split_name)

        texts  = df[2].fillna("").astype(str).tolist()   # column 2 = statement
        labels = [binarize_label(lbl) for lbl in df[1]]  # column 1 = label string

        features = extract_features_batch(texts, nlp)
        y        = np.array(labels, dtype=np.int32)
        splits[split_name] = (features, y)
        print(f"  Class balance: {sum(labels)} misinfo / {len(labels) - sum(labels)} credible")

        # Save to cache
        x_path, y_path = cache_paths[split_name]
        np.save(x_path, features)
        np.save(y_path, y)
        print(f"  Cached → {x_path}")

    return (
        splits["train"][0],      splits["train"][1],
        splits["validation"][0], splits["validation"][1],
        splits["test"][0],       splits["test"][1],
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with early stopping on the validation set.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels (used for early stopping)

    Returns
    -------
    Fitted XGBClassifier with best iteration loaded.
    """
    print("\nTraining XGBoost classifier...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Training complete. Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test",
) -> tuple[dict, np.ndarray]:
    """
    Evaluate classifier and print a report.

    Returns
    -------
    metrics : dict with accuracy, f1, precision, recall
    y_pred  : np.ndarray of predicted labels
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy":  float(accuracy_score(y, y_pred)),
        "f1":        float(f1_score(y, y_pred, average="binary")),
        "precision": float(precision_score(y, y_pred, average="binary")),
        "recall":    float(recall_score(y, y_pred, average="binary")),
    }

    print(f"\n--- XGBoost Results ({split_name}) ---")
    for name, val in metrics.items():
        print(f"  {name:12s}: {val:.4f}")
    print()
    print(classification_report(y, y_pred, target_names=["Credible", "Misinformation"]))

    return metrics, y_pred


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Save a labelled confusion-matrix heatmap (Blues palette)."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Credible", "Misinformation"],
        yticklabels=["Credible", "Misinformation"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("XGBoost – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(model: xgb.XGBClassifier, save_path: str):
    """Save a horizontal bar chart of XGBoost feature importances (by gain)."""
    importances = model.feature_importances_
    order = np.argsort(importances)            # ascending, so bottom = least important
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_vals  = importances[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_names, sorted_vals, color="steelblue")
    ax.set_xlabel("Feature Importance (relative gain)")
    ax.set_title("XGBoost – Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Full pipeline (entry point)
# ---------------------------------------------------------------------------

def run_xgboost_pipeline(force_recompute: bool = False):
    """
    Execute the complete XGBoost pipeline end-to-end.

    Parameters
    ----------
    force_recompute : passed through to load_and_prepare_data(); set True to
                      ignore cached feature .npy files and recompute from scratch.

    Returns
    -------
    model    : trained XGBClassifier
    metrics  : dict (accuracy, f1, precision, recall) on LIAR test set
    y_test   : true test labels
    y_pred   : predicted test labels
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(force_recompute)

    model = train_xgboost(X_train, y_train, X_val, y_val)

    print("\nEvaluating on test set...")
    metrics, y_pred = evaluate_model(model, X_test, y_test, split_name="test")

    print("\nSaving result artefacts...")
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(RESULTS_DIR, "xgboost_confusion_matrix.png"),
    )
    plot_feature_importance(
        model,
        os.path.join(RESULTS_DIR, "xgboost_feature_importance.png"),
    )

    with open(os.path.join(RESULTS_DIR, "xgboost_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'xgboost_metrics.json')}")

    return model, metrics, y_test, y_pred


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_xgboost_pipeline()
