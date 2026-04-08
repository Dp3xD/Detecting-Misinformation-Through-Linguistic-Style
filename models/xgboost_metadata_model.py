"""
xgboost_metadata_model.py
--------------------------
XGBoost+Metadata classifier: 7 linguistic features (from xgboost_model)
combined with 6 metadata features derived from the LIAR TSV columns.

Metadata features (6):
  is_democrat     – party column contains "democrat"
  is_republican   – party column contains "republican"
  is_none_party   – party is "none" / empty / NaN
  is_other_party  – any other party affiliation
  lie_ratio       – speaker's historical deception rate from credit-history cols
  is_politician   – job_title contains a political role keyword

Total features: 7 linguistic + 6 metadata = 13.

Feature cache is stored separately from the linguistic-only model:
  data/features_meta_train.npy / data/features_meta_train_labels.npy
  data/features_meta_val.npy   / data/features_meta_val_labels.npy
  data/features_meta_test.npy  / data/features_meta_test_labels.npy

If the linguistic cache (data/features_train.npy etc.) already exists from a
previous XGBoost-Linguistic run, it is reused automatically — only the fast
vectorised metadata extraction is re-run.

Run independently:
    python models/xgboost_metadata_model.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-use data-loading utilities from the linguistic model
from models.xgboost_model import (
    _load_liar_split,
    binarize_label,
    DATA_DIR,
    RESULTS_DIR,
)
from features.feature_extractor import (
    extract_features_batch,
    FEATURE_NAMES,
    load_spacy_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# Job-title keywords that indicate a political role
POLITICIAN_TITLES = frozenset([
    "senator", "representative", "governor", "president",
    "congressman", "legislator", "candidate",
])

METADATA_FEATURE_NAMES = [
    "is_democrat",
    "is_republican",
    "is_none_party",
    "is_other_party",
    "lie_ratio",
    "is_politician",
]

# Full 13-feature list in order: linguistic first, then metadata
ALL_FEATURE_NAMES = FEATURE_NAMES + METADATA_FEATURE_NAMES

# Linguistic feature cache paths (shared with xgboost_model, reused if present)
_LING_CACHE = {
    split: (
        os.path.join(DATA_DIR, f"features_{stem}.npy"),
        os.path.join(DATA_DIR, f"features_{stem}_labels.npy"),
    )
    for split, stem in [("train", "train"), ("validation", "val"), ("test", "test")]
}

# Combined (meta) feature cache paths
_META_CACHE = {
    split: (
        os.path.join(DATA_DIR, f"features_meta_{stem}.npy"),
        os.path.join(DATA_DIR, f"features_meta_{stem}_labels.npy"),
    )
    for split, stem in [("train", "train"), ("validation", "val"), ("test", "test")]
}


# ---------------------------------------------------------------------------
# Metadata feature extraction
# ---------------------------------------------------------------------------

def extract_metadata_features(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorised extraction of 6 metadata features from a LIAR TSV DataFrame.

    TSV columns used:
      5  – job_title   → is_politician
      7  – party       → is_democrat / is_republican / is_none_party / is_other_party
      8  – barely_true_count  ┐
      9  – false_count         │→ lie_ratio
      10 – half_true_count     │
      11 – mostly_true_count   │
      12 – pants_fire_count   ┘

    Returns np.ndarray of shape (n_samples, 6).
    """
    n = len(df)
    features = np.zeros((n, 6), dtype=np.float32)

    # ── Party one-hot (col 7) ────────────────────────────────────────────────
    party = df[7].fillna("").astype(str).str.lower().str.strip()

    is_dem  = party.str.contains("democrat",   regex=False)
    is_rep  = party.str.contains("republican", regex=False)
    is_none = party.isin(["", "nan", "none"])

    features[:, 0] = is_dem.astype(np.float32).values
    features[:, 1] = is_rep.astype(np.float32).values
    features[:, 2] = is_none.astype(np.float32).values
    # other = not democrat, not republican, not none/blank
    features[:, 3] = (~is_dem & ~is_rep & ~is_none).astype(np.float32).values

    # ── Lie ratio from credit-history counts (cols 8–12) ────────────────────
    def _to_float(col_idx):
        return pd.to_numeric(df[col_idx], errors="coerce").fillna(0.0)

    barely_true = _to_float(8)
    false_cnt   = _to_float(9)
    half_true   = _to_float(10)
    mostly_true = _to_float(11)
    pants_fire  = _to_float(12)

    total = barely_true + false_cnt + half_true + mostly_true + pants_fire + 1.0
    features[:, 4] = ((barely_true + false_cnt + pants_fire) / total).astype(np.float32).values

    # ── Is-politician flag from job_title (col 5) ────────────────────────────
    job = df[5].fillna("").astype(str).str.lower()
    politician_re = "|".join(POLITICIAN_TITLES)
    features[:, 5] = job.str.contains(politician_re, regex=True).astype(np.float32).values

    return features


# ---------------------------------------------------------------------------
# Data loading with combined feature cache
# ---------------------------------------------------------------------------

def load_and_prepare_data_meta(force_recompute: bool = False):
    """
    Load LIAR TSVs, build 13-feature matrices (linguistic + metadata),
    and cache results.

    Reuses the linguistic feature cache (data/features_*.npy) produced by
    xgboost_model if it exists, avoiding redundant spaCy processing.

    Parameters
    ----------
    force_recompute : if True, ignore existing .npy caches and recompute.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test  (np.ndarray pairs)
    """
    # ── Cache hit ─────────────────────────────────────────────────────────────
    all_cached = all(
        os.path.exists(xp) and os.path.exists(yp)
        for xp, yp in _META_CACHE.values()
    )
    if all_cached and not force_recompute:
        print("Loading cached metadata features from data/")
        splits = {
            name: (np.load(xp), np.load(yp))
            for name, (xp, yp) in _META_CACHE.items()
        }
        return (
            splits["train"][0],      splits["train"][1],
            splits["validation"][0], splits["validation"][1],
            splits["test"][0],       splits["test"][1],
        )

    # ── Cache miss ────────────────────────────────────────────────────────────
    print("Loading LIAR dataset for XGBoost+Metadata...")
    os.makedirs(DATA_DIR, exist_ok=True)
    nlp    = None
    splits = {}

    for split_name in ("train", "validation", "test"):
        print(f"\n--- Processing '{split_name}' split (metadata model) ---")
        df     = _load_liar_split(split_name)
        labels = [binarize_label(lbl) for lbl in df[1]]
        y      = np.array(labels, dtype=np.int32)

        # ── Linguistic features: reuse cache if available ─────────────────
        ling_xp, _ = _LING_CACHE[split_name]
        if os.path.exists(ling_xp) and not force_recompute:
            print(f"  Reusing linguistic cache: {os.path.basename(ling_xp)}")
            X_ling = np.load(ling_xp)
        else:
            if nlp is None:
                nlp = load_spacy_model()
            texts  = df[2].fillna("").astype(str).tolist()
            X_ling = extract_features_batch(texts, nlp)

        # ── Metadata features (fast, vectorised) ─────────────────────────
        X_meta = extract_metadata_features(df)

        # ── Concatenate → 13 features ─────────────────────────────────────
        X = np.concatenate([X_ling, X_meta], axis=1).astype(np.float32)
        splits[split_name] = (X, y)

        n_mis = sum(labels)
        print(f"  Feature matrix shape: {X.shape}  "
              f"({n_mis} misinfo / {len(labels) - n_mis} credible)")

        # Save combined cache
        xp, yp = _META_CACHE[split_name]
        np.save(xp, X)
        np.save(yp, y)
        print(f"  Cached → {os.path.basename(xp)}")

    return (
        splits["train"][0],      splits["train"][1],
        splits["validation"][0], splits["validation"][1],
        splits["test"][0],       splits["test"][1],
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgboost_meta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost on 13 combined features with early stopping."""
    print("\nTraining XGBoost+Metadata classifier...")

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
    """Evaluate and print a classification report."""
    y_pred = model.predict(X)

    metrics = {
        "accuracy":  float(accuracy_score(y, y_pred)),
        "f1":        float(f1_score(y, y_pred, average="binary")),
        "precision": float(precision_score(y, y_pred, average="binary")),
        "recall":    float(recall_score(y, y_pred, average="binary")),
    }

    print(f"\n--- XGBoost+Metadata Results ({split_name}) ---")
    for name, val in metrics.items():
        print(f"  {name:12s}: {val:.4f}")
    print()
    print(classification_report(y, y_pred, target_names=["Credible", "Misinformation"]))

    return metrics, y_pred


# ---------------------------------------------------------------------------
# Full pipeline (entry point)
# ---------------------------------------------------------------------------

def run_xgboost_meta_pipeline(force_recompute: bool = False):
    """
    Execute the complete XGBoost+Metadata pipeline end-to-end.

    Returns
    -------
    model   : trained XGBClassifier (13 features)
    metrics : dict (accuracy, f1, precision, recall) on LIAR test set
    y_test  : true test labels
    y_pred  : predicted test labels
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_and_prepare_data_meta(force_recompute)

    model = train_xgboost_meta(X_train, y_train, X_val, y_val)

    print("\nEvaluating on test set...")
    metrics, y_pred = evaluate_model(model, X_test, y_test, split_name="test")

    with open(os.path.join(RESULTS_DIR, "xgboost_meta_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics, y_test, y_pred


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_xgboost_meta_pipeline()
