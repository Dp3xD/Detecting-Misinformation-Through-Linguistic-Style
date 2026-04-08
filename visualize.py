"""
visualize.py
------------
Full results visualization suite for the misinformation detection project.

Generates and saves to results/:

  1. model_comparison_bar.png          – grouped bar chart, 4 metrics × 3 models
  2. xgboost_feature_importance_linguistic.png  – 7-feature importance (linguistic only)
  3. xgboost_feature_importance_metadata.png    – 13-feature importance, two-color
  4. confusion_matrices_grid.png        – 1×3 grid, percentages, seaborn heatmaps
  5. performance_radar.png              – spider/radar chart across all 4 metrics
  6. cross_domain_drop.png             – in-domain vs cross-domain, per-model drop %

Color palette (consistent across all plots):
  XGBoost-Linguistic : #4C72B0  (blue)
  XGBoost-Metadata   : #55A868  (green)
  DistilBERT         : #C44E52  (red)

Run independently only after main.py has produced results JSONs:
    python visualize.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

from features.feature_extractor import FEATURE_NAMES
from models.xgboost_metadata_model import ALL_FEATURE_NAMES, METADATA_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "XGBoost-Linguistic": "#4C72B0",
    "XGBoost-Metadata":   "#55A868",
    "DistilBERT":         "#C44E52",
}
MODEL_LABELS = list(MODEL_COLORS.keys())

_LING_COLOR  = "#4C72B0"
_META_COLOR  = "#55A868"

DPI          = 150
LABEL_FS     = 12      # axis / tick label font size
TITLE_FS     = 13      # subplot title font size
SUPTITLE_FS  = 14      # figure suptitle font size


def _apply_base_style(ax, grid_axis="y"):
    """Apply consistent white-background, readable-font style to an axes."""
    ax.set_facecolor("white")
    if grid_axis == "y":
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    elif grid_axis == "x":
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=LABEL_FS)


# ---------------------------------------------------------------------------
# 1. Grouped bar chart – metric comparison across all 3 models
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    metrics_dict: dict,
    save_path: str,
):
    """
    Grouped bar chart: 4 metric groups (Accuracy, F1, Precision, Recall),
    3 bars per group (one per model), value labels on top.

    Parameters
    ----------
    metrics_dict : {"XGBoost-Linguistic": {...}, "XGBoost-Metadata": {...}, "DistilBERT": {...}}
    save_path    : full path to output PNG
    """
    metric_keys   = ["accuracy", "f1", "precision", "recall"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall"]
    models        = [m for m in MODEL_LABELS if m in metrics_dict]

    n_groups  = len(metric_keys)
    n_bars    = len(models)
    bar_w     = 0.22
    x         = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    for i, model in enumerate(models):
        vals   = [metrics_dict[model].get(k, 0.0) for k in metric_keys]
        offset = (i - (n_bars - 1) / 2) * (bar_w + 0.03)
        bars   = ax.bar(
            x + offset, vals, bar_w,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.8,
        )
        # Value labels on top of each bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=LABEL_FS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=LABEL_FS)
    ax.set_title("Model Comparison – LIAR Test Set", fontsize=TITLE_FS, pad=10)
    ax.legend(fontsize=LABEL_FS, loc="lower right")
    _apply_base_style(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Feature importance – linguistic model (7 features)
# ---------------------------------------------------------------------------

def plot_feature_importance_linguistic(
    xgb_ling_model,
    save_path: str,
):
    """
    Horizontal bar chart for the 7 linguistic feature importances,
    sorted ascending so the most important appears at the top.
    """
    importances = xgb_ling_model.feature_importances_
    order       = np.argsort(importances)          # ascending → top = most important
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_vals  = importances[order]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.barh(sorted_names, sorted_vals, color=_LING_COLOR, edgecolor="white")
    ax.set_xlabel("Feature Importance (relative gain)", fontsize=LABEL_FS)
    ax.set_title("XGBoost Linguistic Features — Importance Scores", fontsize=TITLE_FS)
    _apply_base_style(ax, grid_axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Feature importance – metadata model (13 features, two-color)
# ---------------------------------------------------------------------------

def plot_feature_importance_metadata(
    xgb_meta_model,
    save_path: str,
):
    """
    Horizontal bar chart for all 13 XGBoost+Metadata feature importances.
    Linguistic features → blue, metadata features → green, with a legend.
    """
    importances = xgb_meta_model.feature_importances_
    order        = np.argsort(importances)
    sorted_names = [ALL_FEATURE_NAMES[i] for i in order]
    sorted_vals  = importances[order]

    colors = [
        _LING_COLOR if name in FEATURE_NAMES else _META_COLOR
        for name in sorted_names
    ]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (relative gain)", fontsize=LABEL_FS)
    ax.set_title("XGBoost+Metadata — Feature Importance (13 Features)", fontsize=TITLE_FS)
    _apply_base_style(ax, grid_axis="x")

    legend_handles = [
        mpatches.Patch(color=_LING_COLOR, label="Linguistic (7)"),
        mpatches.Patch(color=_META_COLOR, label="Metadata (6)"),
    ]
    ax.legend(handles=legend_handles, fontsize=LABEL_FS, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 4. Confusion matrices grid (1 row × 3 cols, percentages)
# ---------------------------------------------------------------------------

def plot_confusion_matrices_grid(
    preds_dict: dict,
    save_path: str,
):
    """
    Single figure with one confusion matrix per model, showing row-normalised
    percentages (i.e. recall-style: what % of each true class was predicted
    as each class).

    Parameters
    ----------
    preds_dict : {"XGBoost-Linguistic": (y_true, y_pred), ...}
    """
    models  = [m for m in MODEL_LABELS if m in preds_dict]
    n_cols  = len(models)
    cmaps   = ["Blues", "Greens", "Oranges"]

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), facecolor="white")
    if n_cols == 1:
        axes = [axes]

    for ax, model, cmap in zip(axes, models, cmaps):
        y_true, y_pred = preds_dict[model]
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Build annotation strings: "NN\n(XX.X%)"
        annot = np.array([
            [f"{int(cm[r, c])}\n({cm_norm[r, c]:.1f}%)" for c in range(2)]
            for r in range(2)
        ])

        sns.heatmap(
            cm_norm,
            annot=annot, fmt="",
            cmap=cmap,
            vmin=0, vmax=100,
            xticklabels=["Credible", "Misinfo"],
            yticklabels=["Credible", "Misinfo"],
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Row %"},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=LABEL_FS)
        ax.set_ylabel("True", fontsize=LABEL_FS)
        ax.set_title(model, fontsize=TITLE_FS, color=MODEL_COLORS[model], fontweight="bold")
        ax.tick_params(labelsize=LABEL_FS - 1)

    fig.suptitle("Confusion Matrices — All Models", fontsize=SUPTITLE_FS, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 5. Radar / spider chart
# ---------------------------------------------------------------------------

def plot_radar(
    metrics_dict: dict,
    save_path: str,
):
    """
    Radar chart with 4 axes (Accuracy, F1, Precision, Recall).
    One semi-transparent polygon per model.
    """
    categories  = ["Accuracy", "F1", "Precision", "Recall"]
    metric_keys = ["accuracy", "f1", "precision", "recall"]
    N           = len(categories)

    # Evenly-spaced angles, close the polygon by appending the first angle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, facecolor="white")
    fig.patch.set_facecolor("white")

    models = [m for m in MODEL_LABELS if m in metrics_dict]

    for model in models:
        values = [metrics_dict[model].get(k, 0.0) for k in metric_keys]
        values += values[:1]          # close the polygon
        color  = MODEL_COLORS[model]

        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    # Axis labels at each spoke
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=LABEL_FS, fontweight="bold")

    # Radial grid
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_title("Model Performance Radar", fontsize=SUPTITLE_FS, pad=20)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=LABEL_FS,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 6. Cross-domain performance drop
# ---------------------------------------------------------------------------

def plot_cross_domain_drop(
    in_domain_metrics: dict,
    cross_domain_metrics: dict,
    save_path: str,
):
    """
    Side-by-side bars (Accuracy left, F1 right) showing in-domain vs
    cross-domain performance for each model.  The percentage drop is
    annotated above the cross-domain bar.

    Parameters
    ----------
    in_domain_metrics    : {"XGBoost-Linguistic": {"accuracy": …, "f1": …}, …}
    cross_domain_metrics : same structure, for FakeNewsNet results
    """
    metrics_to_plot = [("accuracy", "Accuracy"), ("f1", "F1 Score")]
    models = [m for m in MODEL_LABELS
              if m in in_domain_metrics and m in cross_domain_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        x        = np.arange(len(models))
        bar_w    = 0.32

        in_vals    = [in_domain_metrics[m].get(metric_key, 0.0)    for m in models]
        cross_vals = [cross_domain_metrics[m].get(metric_key, 0.0) for m in models]

        # Solid bars = in-domain; hatched bars = cross-domain
        ax.bar(
            x - bar_w / 2, in_vals, bar_w,
            color=[MODEL_COLORS[m] for m in models],
            edgecolor="white", label="In-domain (LIAR)", alpha=0.92,
        )
        ax.bar(
            x + bar_w / 2, cross_vals, bar_w,
            color=[MODEL_COLORS[m] for m in models],
            edgecolor="white", label="Cross-domain (FakeNewsNet)",
            alpha=0.50, hatch="//",
        )

        # Drop % labels above the cross-domain bar
        for xi, (iv, cv) in enumerate(zip(in_vals, cross_vals)):
            drop = (iv - cv) / iv * 100 if iv > 0 else 0.0
            sign = "-" if drop >= 0 else "+"
            ax.annotate(
                f"{sign}{abs(drop):.1f}%",
                xy=(xi + bar_w / 2, cv),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="darkred" if drop >= 0 else "darkgreen",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=LABEL_FS)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_label, fontsize=LABEL_FS)
        ax.set_title(f"{metric_label}: In-domain vs Cross-domain", fontsize=TITLE_FS)
        ax.legend(fontsize=10, loc="lower left")
        _apply_base_style(ax)

    fig.suptitle("Cross-Domain Performance Drop", fontsize=SUPTITLE_FS)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 7. Feature progression line chart
# ---------------------------------------------------------------------------

def plot_feature_progression(
    metrics_all: dict,
    save_path: str,
):
    """
    Line chart telling the "research story" at a glance:
    X axis = model in complexity order, Y axis = score (0–1).
    Two lines: Accuracy (solid) and F1 (dashed).
    Reference baselines: random chance (0.50) and BERT literature (0.62).
    Each point is coloured by its model colour and annotated with its value.
    """
    model_order = ["XGBoost-Linguistic", "XGBoost-Metadata", "DistilBERT"]
    models      = [m for m in model_order if m in metrics_all]
    if not models:
        return

    x         = np.arange(len(models))
    acc_vals  = [metrics_all[m]["accuracy"] for m in models]
    f1_vals   = [metrics_all[m]["f1"]       for m in models]
    colors    = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    # ── Accuracy line (solid, black spine, colored markers) ──────────────────
    ax.plot(x, acc_vals, "-", linewidth=2.5, color="#333333", zorder=4)
    for xi, yi, c in zip(x, acc_vals, colors):
        ax.scatter(xi, yi, s=160, color=c, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=11, fontweight="bold", color=c)

    # ── F1 line (dashed, gray spine, colored markers) ────────────────────────
    ax.plot(x, f1_vals, "--", linewidth=2, color="#888888", zorder=3)
    for xi, yi, c in zip(x, f1_vals, colors):
        ax.scatter(xi, yi, s=110, color=c, marker="s", zorder=4,
                   edgecolors="white", linewidths=1.5, alpha=0.75)
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=10, color="#555555")

    # Dummy handles for legend
    import matplotlib.lines as mlines
    acc_handle = mlines.Line2D([], [], color="#333333", linewidth=2.5,
                               marker="o", markersize=9, label="Accuracy")
    f1_handle  = mlines.Line2D([], [], color="#888888", linewidth=2,
                               marker="s", markersize=8, linestyle="--",
                               label="F1 Score")

    # ── Baselines ─────────────────────────────────────────────────────────────
    ax.axhline(0.50, linestyle=":", color="#888888", linewidth=1.5, alpha=0.8)
    ax.text(len(models) - 1, 0.50 + 0.015, "Random Baseline (0.50)",
            ha="right", va="bottom", fontsize=10, color="#888888",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.axhline(0.62, linestyle=":", color="darkorange", linewidth=1.5, alpha=0.8)
    ax.text(len(models) - 1, 0.62 + 0.015, "BERT Baseline — literature (0.62)",
            ha="right", va="bottom", fontsize=10, color="darkorange",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=LABEL_FS)
    ax.set_xlim(-0.4, len(models) - 0.6)
    y_min = min(min(acc_vals), min(f1_vals)) - 0.08
    ax.set_ylim(max(0.0, y_min), 1.02)
    ax.set_ylabel("Score", fontsize=LABEL_FS)
    ax.set_title(
        "Model Progression: From Linguistic Features to Neural Classification",
        fontsize=TITLE_FS, pad=10,
    )
    ax.legend(handles=[acc_handle, f1_handle], fontsize=LABEL_FS, loc="lower right")
    _apply_base_style(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 8. Results summary table (paper/slides ready)
# ---------------------------------------------------------------------------

def plot_results_summary_table(
    metrics_all: dict,
    cross_domain_metrics: dict | None,
    save_path: str,
):
    """
    A styled matplotlib table with two sections:
      1. In-Domain Performance (LIAR Test Set) – 4 metrics
      2. Cross-Domain Performance (FakeNewsNet) – accuracy + F1

    Uses alternating row shading and model-coloured first column text.
    Suitable for dropping directly into a paper or slide deck.
    """
    model_order   = ["XGBoost-Linguistic", "XGBoost-Metadata", "DistilBERT"]
    active_models = [m for m in model_order if m in metrics_all]

    # ── Colour palette ────────────────────────────────────────────────────────
    _HDR1   = "#2c3e50"   # dark section header
    _HDR2   = "#3d5166"   # column sub-header
    _ROW_A  = "#f4f6f8"   # alternating row A (light blue-gray)
    _ROW_B  = "#ffffff"   # alternating row B (white)

    def _fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else "—"

    # Build row data and per-cell background colours
    cell_text   = []
    cell_colors = []
    n_cols      = 5

    def _add_section_header(label):
        cell_text.append([label, "", "", "", ""])
        cell_colors.append([_HDR1] * n_cols)

    def _add_col_header(cols):
        row = (cols + [""] * n_cols)[:n_cols]
        cell_text.append(row)
        cell_colors.append([_HDR2] * n_cols)

    # ── Section 1: In-domain ──────────────────────────────────────────────────
    _add_section_header("  In-Domain Performance  —  LIAR Test Set")
    _add_col_header(["Model", "Accuracy", "F1", "Precision", "Recall"])
    for i, m in enumerate(active_models):
        met = metrics_all.get(m, {})
        cell_text.append([
            m,
            _fmt(met.get("accuracy")), _fmt(met.get("f1")),
            _fmt(met.get("precision")), _fmt(met.get("recall")),
        ])
        cell_colors.append([_ROW_A if i % 2 == 0 else _ROW_B] * n_cols)

    # ── Section 2: Cross-domain ────────────────────────────────────────────────
    _add_section_header("  Cross-Domain Performance  —  FakeNewsNet Politifact")
    _add_col_header(["Model", "Accuracy", "F1", "", ""])
    if cross_domain_metrics:
        for i, m in enumerate(active_models):
            cd = cross_domain_metrics.get(m, {})
            cell_text.append([
                m,
                _fmt(cd.get("accuracy")), _fmt(cd.get("f1")), "", "",
            ])
            cell_colors.append([_ROW_A if i % 2 == 0 else _ROW_B] * n_cols)
    else:
        cell_text.append(["No cross-domain results available", "", "", "", ""])
        cell_colors.append([_ROW_A] * n_cols)

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows  = len(cell_text)
    fig_h   = max(3.5, n_rows * 0.52 + 0.6)
    fig, ax = plt.subplots(figsize=(13, fig_h), facecolor="white")
    ax.set_axis_off()

    tbl = ax.table(
        cellText   = cell_text,
        cellColours= cell_colors,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # ── Per-cell styling ──────────────────────────────────────────────────────
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        bg = cell_colors[row_idx][col_idx]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#dee2e6")
        cell.set_linewidth(0.8)

        if bg in (_HDR1, _HDR2):
            # Section / column headers: white bold text
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            if bg == _HDR1:
                cell.get_text().set_fontsize(11.5)
                cell.get_text().set_ha("left")
        elif col_idx == 0:
            # Model name: coloured and bold
            model_name = cell_text[row_idx][0].strip()
            fc = MODEL_COLORS.get(model_name, "#333333")
            cell.get_text().set_color(fc)
            cell.get_text().set_fontweight("bold")

    # ── Column widths ─────────────────────────────────────────────────────────
    col_widths = [0.30, 0.155, 0.13, 0.155, 0.13]
    for c_idx, w in enumerate(col_widths):
        for r_idx in range(n_rows):
            tbl[r_idx, c_idx].set_width(w)

    ax.set_title("Results Summary", fontsize=SUPTITLE_FS,
                 fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Master function – generate all 6 plots at once
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    metrics_all: dict,
    preds_all: dict,
    xgb_ling_model,
    xgb_meta_model,
    cross_domain_metrics: dict | None,
    results_dir: str,
):
    """
    Generate and save all 6 visualization files to *results_dir*.

    Parameters
    ----------
    metrics_all          : {"XGBoost-Linguistic": {acc,f1,prec,rec}, …}
    preds_all            : {"XGBoost-Linguistic": (y_true, y_pred), …}
    xgb_ling_model       : fitted 7-feature XGBClassifier
    xgb_meta_model       : fitted 13-feature XGBClassifier
    cross_domain_metrics : {"XGBoost-Linguistic": {accuracy, f1}, …} or None
    results_dir          : directory to save plots
    """
    os.makedirs(results_dir, exist_ok=True)
    print("\nGenerating visualization suite...")

    def _path(fname):
        return os.path.join(results_dir, fname)

    # 1. Comparison bar chart
    if metrics_all:
        plot_comparison_bar(
            metrics_all,
            _path("model_comparison_bar.png"),
        )

    # 2. Linguistic feature importance
    if xgb_ling_model is not None:
        plot_feature_importance_linguistic(
            xgb_ling_model,
            _path("xgboost_feature_importance_linguistic.png"),
        )

    # 3. Metadata feature importance
    if xgb_meta_model is not None:
        plot_feature_importance_metadata(
            xgb_meta_model,
            _path("xgboost_feature_importance_metadata.png"),
        )

    # 4. Confusion matrices grid
    if preds_all:
        plot_confusion_matrices_grid(
            preds_all,
            _path("confusion_matrices_grid.png"),
        )

    # 5. Radar chart
    if metrics_all:
        plot_radar(
            metrics_all,
            _path("performance_radar.png"),
        )

    # 6. Cross-domain drop
    if cross_domain_metrics and metrics_all:
        in_domain = {m: v for m, v in metrics_all.items() if m in cross_domain_metrics}
        if in_domain:
            plot_cross_domain_drop(
                in_domain,
                cross_domain_metrics,
                _path("cross_domain_drop.png"),
            )

    # 7. Feature progression
    if metrics_all:
        plot_feature_progression(
            metrics_all,
            _path("feature_progression.png"),
        )

    # 8. Results summary table
    if metrics_all:
        plot_results_summary_table(
            metrics_all,
            cross_domain_metrics,
            _path("results_summary_table.png"),
        )

    print("  Visualization suite complete.")


# ---------------------------------------------------------------------------
# Stand-alone runner (loads saved JSON metrics from results/)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    def _load_json(name):
        path = os.path.join(_RESULTS, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    _metrics_all = {}
    for fname, label in [
        ("xgboost_metrics.json",      "XGBoost-Linguistic"),
        ("xgboost_meta_metrics.json", "XGBoost-Metadata"),
        ("distilbert_metrics.json",   "DistilBERT"),
    ]:
        m = _load_json(fname)
        if m:
            _metrics_all[label] = m

    print("Loaded metrics:", list(_metrics_all.keys()))
    print("Note: confusion matrices and feature importance require model objects.")
    print("Run main.py for the full suite; this script only demos what it can.")

    if _metrics_all:
        plot_comparison_bar(_metrics_all, os.path.join(_RESULTS, "model_comparison_bar.png"))
        plot_radar(_metrics_all, os.path.join(_RESULTS, "performance_radar.png"))
