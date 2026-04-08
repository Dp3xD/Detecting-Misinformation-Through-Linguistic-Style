"""
shap_analysis.py
----------------
SHAP explainability analysis for the XGBoost+Metadata model.

Generates three plots saved to results/:

  shap_summary.png
      Beeswarm summary plot over all test samples and all 13 features.
      Shows which features push predictions toward or away from misinformation.

  shap_waterfall_misclassified.png
      Waterfall explanation for the first false-negative test sample
      (true label = misinformation, predicted = credible).
      Also prints the statement text and labels to console.

  shap_dependence_lie_ratio.png
      Dependence plot for lie_ratio coloured by is_republican interaction.
      Reveals how the speaker's deception history interacts with party affiliation.

Run independently (requires the metadata model cache to exist):
    python models/shap_analysis.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xgboost_metadata_model import (
    ALL_FEATURE_NAMES,
    load_and_prepare_data_meta,
    run_xgboost_meta_pipeline,
)
from models.xgboost_model import _load_liar_split

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "results")

# Feature indices (derived from ALL_FEATURE_NAMES; verified at module load)
_LIE_RATIO_IDX  = ALL_FEATURE_NAMES.index("lie_ratio")
_IS_REP_IDX     = ALL_FEATURE_NAMES.index("is_republican")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_shap_values(explainer, X: np.ndarray):
    """
    Return a 2-D SHAP value array of shape (n_samples, n_features) for class 1
    (misinformation), compatible with both shap < 0.40 and >= 0.40 APIs.
    """
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        # Older shap: returns [class_0_array, class_1_array]
        return np.array(raw[1])

    arr = np.array(raw)
    if arr.ndim == 3:
        # Intermediate API: shape (n, features, n_classes)
        return arr[:, :, 1]

    return arr  # already (n, features)


def _get_explanation_obj(explainer, X: np.ndarray, feature_names: list):
    """
    Return a shap.Explanation object containing class-1 SHAP values,
    regardless of shap version.
    """
    import shap

    exp = explainer(X)

    # Normalise to 2-D values (n, features) for class 1
    vals = exp.values
    base = exp.base_values

    if vals.ndim == 3:
        vals = vals[:, :, 1]
        base = base[:, 1] if base.ndim > 1 else base

    return shap.Explanation(
        values        = vals,
        base_values   = base,
        data          = exp.data,
        feature_names = feature_names,
    )


# ---------------------------------------------------------------------------
# Plot 1: Beeswarm summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_vals: np.ndarray,
    X_test:    np.ndarray,
    save_path: str,
):
    """
    Beeswarm (dot) summary plot: each row = one feature, each dot = one sample.
    Colour encodes feature value; x-position encodes SHAP impact.
    """
    import shap

    shap.summary_plot(
        shap_vals,
        X_test,
        feature_names = ALL_FEATURE_NAMES,
        plot_type     = "dot",
        show          = False,
        max_display   = len(ALL_FEATURE_NAMES),
    )

    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.title("SHAP Feature Impact — XGBoost+Metadata", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Waterfall for the first false-negative (missed misinformation)
# ---------------------------------------------------------------------------

def plot_shap_waterfall_misclassified(
    explanation,          # shap.Explanation (class 1, 2-D values)
    y_test:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: str,
):
    """
    Locate the first false negative (true=1, pred=0) in the test set and draw
    a waterfall plot showing which features pushed the model toward the wrong
    decision.  Prints the sample's statement text to console.
    """
    import shap

    # Locate first false negative
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    if len(fn_indices) == 0:
        print("  WARNING: no false negatives found in test set — "
              "skipping waterfall plot.")
        return None

    idx = int(fn_indices[0])

    # Load the raw statement text from the LIAR test TSV (col 2)
    try:
        df_test = _load_liar_split("test")
        statement = df_test.iloc[idx, 2]
        true_label_str = df_test.iloc[idx, 1]
    except Exception:
        statement      = "(text unavailable)"
        true_label_str = "(unknown)"

    print("\n  ── Misclassified misinformation example (waterfall) ──────────────")
    print(f"  Test index   : {idx}")
    print(f"  Statement    : {statement}")
    print(f"  True label   : {true_label_str}  (→ 1 / misinformation)")
    print(f"  Predicted    : credible (0)  [false negative]")
    print("  ──────────────────────────────────────────────────────────────────\n")

    # Draw waterfall for this one sample
    shap.plots.waterfall(explanation[idx], show=False, max_display=13)

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.suptitle(
        "SHAP Explanation — Misclassified Misinformation Example",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

    return idx


# ---------------------------------------------------------------------------
# Plot 3: Dependence plot — lie_ratio coloured by is_republican
# ---------------------------------------------------------------------------

def plot_shap_dependence_lie_ratio(
    shap_vals: np.ndarray,
    X_test:    np.ndarray,
    save_path: str,
):
    """
    Dependence plot for lie_ratio: x = feature value, y = SHAP impact.
    Points coloured by is_republican to reveal the interaction effect.
    """
    import shap

    fig, ax = plt.subplots(figsize=(8, 5))

    shap.dependence_plot(
        _LIE_RATIO_IDX,
        shap_vals,
        X_test,
        feature_names      = ALL_FEATURE_NAMES,
        interaction_index  = _IS_REP_IDX,
        ax                 = ax,
        show               = False,
    )

    ax.set_title(
        "SHAP Dependence: lie_ratio vs Prediction Impact",
        fontsize=13, pad=10,
    )
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_shap_analysis(
    model,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    y_pred:    np.ndarray,
    results_dir: str = RESULTS_DIR,
):
    """
    Run the complete SHAP explainability suite for the XGBoost+Metadata model.

    Parameters
    ----------
    model        : fitted XGBClassifier (13 features)
    X_test       : test feature matrix, shape (n, 13)
    y_test       : true binary labels
    y_pred       : model predictions
    results_dir  : directory to save the three PNG files
    """
    import shap

    os.makedirs(results_dir, exist_ok=True)

    print("\nRunning SHAP analysis on XGBoost+Metadata test set "
          f"({len(X_test)} samples, {X_test.shape[1]} features)...")

    # ── Build explainer ───────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)

    # 2-D array (n, 13) — used by summary_plot and dependence_plot
    shap_vals  = _get_shap_values(explainer, X_test)

    # Explanation object (n, 13) — used by waterfall
    explanation = _get_explanation_obj(explainer, X_test, ALL_FEATURE_NAMES)

    print(f"  SHAP value matrix shape: {shap_vals.shape}")

    # ── Plot 1: Beeswarm summary ──────────────────────────────────────────────
    print("\n  [SHAP 1/3] Beeswarm summary plot...")
    plot_shap_summary(
        shap_vals, X_test,
        os.path.join(results_dir, "shap_summary.png"),
    )

    # ── Plot 2: Waterfall for first false negative ────────────────────────────
    print("\n  [SHAP 2/3] Waterfall plot (first false negative)...")
    plot_shap_waterfall_misclassified(
        explanation, y_test, y_pred,
        os.path.join(results_dir, "shap_waterfall_misclassified.png"),
    )

    # ── Plot 3: Dependence — lie_ratio × is_republican ────────────────────────
    print("\n  [SHAP 3/3] Dependence plot (lie_ratio × is_republican)...")
    plot_shap_dependence_lie_ratio(
        shap_vals, X_test,
        os.path.join(results_dir, "shap_dependence_lie_ratio.png"),
    )

    print("\n  SHAP analysis complete.")


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading XGBoost+Metadata model and test data...")
    model, _, y_test, y_pred = run_xgboost_meta_pipeline()

    # Reload test features from cache / recompute
    _, _, _, _, X_test, y_test = load_and_prepare_data_meta()

    run_shap_analysis(model, X_test, y_test, y_pred)
