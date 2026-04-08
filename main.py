"""
main.py
-------
Orchestrates the full misinformation detection pipeline.

  Stage 1 – XGBoost-Linguistic   : 7 linguistic features → train → evaluate
  Stage 2 – XGBoost-Metadata     : 13 features (linguistic + metadata) → train → evaluate
  Stage 3 – DistilBERT           : fine-tune on raw text → evaluate
  Stage 4 – Comparison table     : 3-column CSV + terminal printout
  Stage 5 – Visualization suite  : 6 plots saved to results/
  Stage 6 – Cross-domain test    : FakeNewsNet politifact (500 samples)

Usage:
    python main.py                    # full pipeline
    python main.py --skip-bert        # skip DistilBERT fine-tuning
    python main.py --skip-meta        # skip XGBoost+Metadata
    python main.py --skip-cross-domain
    python main.py --force-recompute  # ignore feature caches, recompute all
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import requests
import torch
from io import StringIO
from sklearn.metrics import accuracy_score, f1_score

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from models.xgboost_model          import run_xgboost_pipeline
from models.xgboost_metadata_model import run_xgboost_meta_pipeline, load_and_prepare_data_meta
from models.distilbert_model       import run_distilbert_pipeline, CKPT_DIR, MODEL_NAME
from models.shap_analysis          import run_shap_analysis
from features.feature_extractor    import extract_features_batch, load_spacy_model
from visualize import (
    generate_all_visualizations,
    plot_cross_domain_drop,
    plot_feature_progression,
    plot_results_summary_table,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# FakeNewsNet raw-CSV URLs
_FNN_BASE = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
FAKENEWSNET_URLS = {
    "fake": f"{_FNN_BASE}/politifact_fake.csv",
    "real": f"{_FNN_BASE}/politifact_real.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_table(metrics_all: dict) -> pd.DataFrame:
    """
    Print and save a comparison of all available models across 4 metrics.
    Columns: Metric | XGBoost-Linguistic | XGBoost-Metadata | DistilBERT
    (columns are omitted if a model was not run).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metric_keys   = ["accuracy", "f1", "precision", "recall"]
    model_order   = ["XGBoost-Linguistic", "XGBoost-Metadata", "DistilBERT"]
    active_models = [m for m in model_order if m in metrics_all]

    rows = []
    for key in metric_keys:
        row = {"Metric": key.capitalize()}
        for model in active_models:
            row[model] = round(metrics_all[model].get(key, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    bar = "=" * (14 + 20 * len(active_models))
    print(f"\n{bar}")
    print("  MODEL COMPARISON  –  LIAR Test Set")
    print(bar)
    print(df.to_string(index=False))
    print(bar)

    save_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\n  Comparison table saved: {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Cross-domain test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_or_load_csv(url: str, local_path: str) -> pd.DataFrame | None:
    """Download from *url* and cache, or load from cache. Returns None on error."""
    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    print(f"    Downloading {url} …")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.to_csv(local_path, index=False)
        return df
    except requests.HTTPError as exc:
        print(f"    WARNING: HTTP {exc.response.status_code} fetching {url} – skipping.")
        return None
    except Exception as exc:
        print(f"    WARNING: could not fetch {url}: {exc} – skipping.")
        return None


def load_fakenewsnet(per_class: int = 250) -> tuple[list[str], list[int]] | tuple[None, None]:
    """
    Load up to *per_class* titles from each FakeNewsNet politifact CSV.
    Columns: id, news_url, title, tweet_ids.  Uses 'title' as text.
    Labels: 1=fake, 0=real.  Returns (None, None) if either CSV fails.
    """
    print("\nLoading FakeNewsNet (politifact) for cross-domain evaluation...")
    os.makedirs(DATA_DIR, exist_ok=True)

    rng           = np.random.default_rng(RANDOM_SEED)
    texts, labels = [], []
    failed        = []

    for split, url in FAKENEWSNET_URLS.items():
        local_path = os.path.join(DATA_DIR, os.path.basename(url))
        df = _download_or_load_csv(url, local_path)

        if df is None:
            failed.append(split)
            continue

        if "title" not in df.columns:
            print(f"    WARNING: 'title' column missing in {os.path.basename(url)} "
                  f"(columns: {list(df.columns)}) – skipping.")
            failed.append(split)
            continue

        col_texts = df["title"].dropna().astype(str).tolist()
        n   = min(per_class, len(col_texts))
        idx = rng.choice(len(col_texts), size=n, replace=False)

        col_label = 1 if split == "fake" else 0
        texts.extend([col_texts[i] for i in idx])
        labels.extend([col_label] * n)

    if failed:
        print(f"  WARNING: failed to load splits: {failed}. Cross-domain test skipped.")
        return None, None

    n_fake = sum(labels)
    print(f"  Loaded {len(texts)} samples  ({n_fake} fake / {len(texts) - n_fake} real)")
    return texts, labels


def _load_bert_from_checkpoint() -> tuple:
    """
    Load the highest-step checkpoint from CKPT_DIR into a predict-only Trainer.
    Returns (trainer, tokenizer) on success, (None, None) if nothing found.
    """
    if not os.path.isdir(CKPT_DIR):
        return None, None

    ckpt_dirs = sorted(
        [d for d in os.listdir(CKPT_DIR)
         if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CKPT_DIR, d))],
        key=lambda d: int(d.split("-")[-1]),
    )
    if not ckpt_dirs:
        return None, None

    latest = os.path.join(CKPT_DIR, ckpt_dirs[-1])
    print(f"  Loading DistilBERT from checkpoint: {latest}")

    model     = DistilBertForSequenceClassification.from_pretrained(latest)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    args      = TrainingArguments(
        output_dir=CKPT_DIR,
        per_device_eval_batch_size=16,
        report_to="none",
    )
    return Trainer(model=model, args=args), tokenizer


def cross_domain_test(
    xgb_ling_model,
    xgb_meta_model,
    bert_trainer,
    bert_tokenizer,
    per_class: int = 250,
) -> dict:
    """
    Evaluate all available models on FakeNewsNet politifact data.

    For XGBoost+Metadata, metadata features are unavailable in FakeNewsNet,
    so the 6 metadata columns are set to zero (only linguistic features are
    informative here).

    Returns
    -------
    dict : {"XGBoost-Linguistic": {"accuracy": …, "f1": …}, …}
           Empty dict if data loading fails.
    """
    texts, labels = load_fakenewsnet(per_class)
    if texts is None:
        print("  Cross-domain test skipped (data unavailable).")
        return {}

    y_true  = np.array(labels)
    results = {}

    # Extract linguistic features once upfront — reused by both XGBoost models
    X_ling = None
    if xgb_ling_model is not None or xgb_meta_model is not None:
        print("\n  Extracting linguistic features for cross-domain data...")
        nlp    = load_spacy_model()
        X_ling = extract_features_batch(texts, nlp, verbose=True)

    # ── XGBoost-Linguistic ────────────────────────────────────────────────────
    if xgb_ling_model is not None:
        print("  Running XGBoost-Linguistic on cross-domain data...")
        y_pred = xgb_ling_model.predict(X_ling)
        results["XGBoost-Linguistic"] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1":       round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        }

    # ── XGBoost-Metadata ──────────────────────────────────────────────────────
    if xgb_meta_model is not None:
        # FakeNewsNet has no speaker metadata; fill the 6 metadata columns with
        # zeros (represents "unknown speaker") so linguistic features still carry signal.
        print("  Running XGBoost-Metadata on cross-domain data...")
        print("  NOTE: Metadata features set to zero for cross-domain "
              "(speaker info unavailable).")
        X_meta_zero = np.zeros((len(texts), 6), dtype=np.float32)
        X_combined  = np.concatenate([X_ling, X_meta_zero], axis=1)
        y_pred = xgb_meta_model.predict(X_combined)
        results["XGBoost-Metadata"] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1":       round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        }

    # ── DistilBERT ────────────────────────────────────────────────────────────
    if bert_trainer is not None:
        print("  Running DistilBERT on cross-domain data...")
        from datasets import Dataset

        def _tokenize(batch):
            return bert_tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        cross_ds = Dataset.from_dict({"text": texts, "labels": labels})
        cross_ds = cross_ds.map(_tokenize, batched=True, remove_columns=["text"])
        cross_ds.set_format("torch")

        pred_out = bert_trainer.predict(cross_ds)
        y_pred   = np.argmax(pred_out.predictions, axis=-1)
        results["DistilBERT"] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1":       round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        }

    # ── Print & save ──────────────────────────────────────────────────────────
    if results:
        df  = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
        bar = "=" * 52
        print(f"\n{bar}")
        print("  CROSS-DOMAIN TEST  –  FakeNewsNet Politifact")
        print(bar)
        print(df.to_string(index=False))
        print(bar)

        save_path = os.path.join(RESULTS_DIR, "cross_domain_results.csv")
        df.to_csv(save_path, index=False)
        print(f"\n  Cross-domain results saved: {save_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Misinformation detection pipeline (LIAR + FakeNewsNet)"
    )
    parser.add_argument("--skip-bert",         action="store_true",
                        help="Skip DistilBERT fine-tuning.")
    parser.add_argument("--skip-xgb",          action="store_true",
                        help="Skip XGBoost-Linguistic training.")
    parser.add_argument("--skip-meta",         action="store_true",
                        help="Skip XGBoost+Metadata training.")
    parser.add_argument("--skip-shap",         action="store_true",
                        help="Skip SHAP explainability analysis.")
    parser.add_argument("--skip-cross-domain", action="store_true",
                        help="Skip cross-domain FakeNewsNet evaluation.")
    parser.add_argument("--skip-viz",          action="store_true",
                        help="Skip visualization suite.")
    parser.add_argument("--force-recompute",   action="store_true",
                        help="Ignore feature caches and recompute from scratch.")
    parser.add_argument("--cross-domain-n", type=int, default=250,
                        help="Max samples per class for cross-domain test (default: 250).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    banner = "=" * 62
    print(f"\n{banner}")
    print("   MISINFORMATION DETECTION PIPELINE")
    print("   LIAR Dataset  |  3 Models  |  6 Visualizations")
    print(f"{banner}\n")

    # Containers for results
    metrics_all = {}        # model_label → {accuracy, f1, precision, recall}
    preds_all   = {}        # model_label → (y_true, y_pred)

    xgb_ling_model  = None
    xgb_meta_model  = None
    bert_trainer    = None
    bert_tokenizer  = None

    # ── Stage 1: XGBoost-Linguistic ──────────────────────────────────────────
    if not args.skip_xgb:
        print(f"{banner}")
        print("  STAGE 1/6  –  XGBoost-Linguistic")
        print(f"{banner}")
        xgb_ling_model, xgb_metrics, y_test, y_pred = \
            run_xgboost_pipeline(force_recompute=args.force_recompute)
        metrics_all["XGBoost-Linguistic"] = xgb_metrics
        preds_all["XGBoost-Linguistic"]   = (y_test, y_pred)
    else:
        print("  [Stage 1] XGBoost-Linguistic skipped (--skip-xgb).")

    # ── Stage 2: XGBoost-Metadata ────────────────────────────────────────────
    if not args.skip_meta:
        print(f"\n{banner}")
        print("  STAGE 2/6  –  XGBoost+Metadata")
        print(f"{banner}")
        xgb_meta_model, meta_metrics, y_test_m, y_pred_m = \
            run_xgboost_meta_pipeline(force_recompute=args.force_recompute)
        metrics_all["XGBoost-Metadata"] = meta_metrics
        preds_all["XGBoost-Metadata"]   = (y_test_m, y_pred_m)

        # ── SHAP explainability (runs on the test split used above) ──────────
        if not args.skip_shap:
            print(f"\n{banner}")
            print("  STAGE 2b   –  SHAP Explainability (XGBoost+Metadata)")
            print(f"{banner}")
            # Reload the 13-feature test matrix from cache (fast, no recompute)
            _, _, _, _, X_test_meta, _ = load_and_prepare_data_meta(force_recompute=False)
            run_shap_analysis(
                model       = xgb_meta_model,
                X_test      = X_test_meta,
                y_test      = y_test_m,
                y_pred      = y_pred_m,
                results_dir = RESULTS_DIR,
            )
        else:
            print("  [Stage 2b] SHAP analysis skipped (--skip-shap).")
    else:
        print("  [Stage 2] XGBoost+Metadata skipped (--skip-meta).")

    # ── Stage 3: DistilBERT ──────────────────────────────────────────────────
    if not args.skip_bert:
        print(f"\n{banner}")
        print("  STAGE 3/6  –  DistilBERT (fine-tuning)")
        print(f"{banner}")
        bert_trainer, _, bert_tokenizer, bert_metrics, y_true_b, y_pred_b = \
            run_distilbert_pipeline()
        metrics_all["DistilBERT"] = bert_metrics
        preds_all["DistilBERT"]   = (y_true_b, y_pred_b)
    else:
        print("  [Stage 3] DistilBERT skipped (--skip-bert).")

    # ── Stage 4: Comparison table ─────────────────────────────────────────────
    print(f"\n{banner}")
    print("  STAGE 4/6  –  Model comparison table")
    print(f"{banner}")
    if metrics_all:
        save_comparison_table(metrics_all)
    else:
        print("  No model results available – skipping.")

    # ── Stage 5: Visualization suite ─────────────────────────────────────────
    if not args.skip_viz:
        print(f"\n{banner}")
        print("  STAGE 5/6  –  Visualization suite")
        print(f"{banner}")

        # Cross-domain metrics aren't available yet; placeholder for generate_all
        # (the drop chart is produced after Stage 6 below)
        generate_all_visualizations(
            metrics_all      = metrics_all,
            preds_all        = preds_all,
            xgb_ling_model   = xgb_ling_model,
            xgb_meta_model   = xgb_meta_model,
            cross_domain_metrics = None,   # updated after Stage 6
            results_dir      = RESULTS_DIR,
        )
    else:
        print("  [Stage 5] Visualization suite skipped (--skip-viz).")

    # ── Stage 6: Cross-domain test ────────────────────────────────────────────
    if args.skip_cross_domain:
        print("  [Stage 6] Cross-domain test skipped (--skip-cross-domain).")
        cross_domain_metrics = {}
    elif not any([xgb_ling_model, xgb_meta_model]):
        print("  [Stage 6] Cross-domain test skipped – no XGBoost models available.")
        cross_domain_metrics = {}
    else:
        print(f"\n{banner}")
        print("  STAGE 6/6  –  Cross-domain test (FakeNewsNet politifact)")
        print(f"{banner}")

        # DistilBERT: use in-memory trainer or fall back to checkpoint
        _bert_trainer_cd  = bert_trainer
        _bert_tokenizer_cd = bert_tokenizer
        if _bert_trainer_cd is None and not args.skip_bert:
            pass   # bert was skipped intentionally
        elif _bert_trainer_cd is None:
            print("  DistilBERT not in memory; attempting to load from checkpoint...")
            _bert_trainer_cd, _bert_tokenizer_cd = _load_bert_from_checkpoint()
            if _bert_trainer_cd is None:
                print("  WARNING: no checkpoint found – DistilBERT excluded from "
                      "cross-domain test.")

        cross_domain_metrics = cross_domain_test(
            xgb_ling_model  = xgb_ling_model,
            xgb_meta_model  = xgb_meta_model,
            bert_trainer    = _bert_trainer_cd,
            bert_tokenizer  = _bert_tokenizer_cd,
            per_class       = args.cross_domain_n,
        )

        # Cross-domain drop plot — needs cross-domain data, generated here
        if cross_domain_metrics and not args.skip_viz:
            in_domain = {m: v for m, v in metrics_all.items()
                         if m in cross_domain_metrics}
            if in_domain:
                plot_cross_domain_drop(
                    in_domain,
                    cross_domain_metrics,
                    os.path.join(RESULTS_DIR, "cross_domain_drop.png"),
                )

    # ── Final plots (need both in-domain + cross-domain data) ────────────────
    if not args.skip_viz and metrics_all:
        print(f"\n{banner}")
        print("  FINAL PLOTS  –  Progression chart + summary table")
        print(f"{banner}")

        plot_feature_progression(
            metrics_all,
            os.path.join(RESULTS_DIR, "feature_progression.png"),
        )
        plot_results_summary_table(
            metrics_all,
            cross_domain_metrics if not args.skip_cross_domain else None,
            os.path.join(RESULTS_DIR, "results_summary_table.png"),
        )

    print(f"\n{banner}")
    print("  Pipeline complete.  All artefacts saved to results/")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
