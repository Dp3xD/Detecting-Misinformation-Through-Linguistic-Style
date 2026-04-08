"""
distilbert_model.py
-------------------
Fine-tunes DistilBERT (distilbert-base-uncased) on LIAR raw text for
binary misinformation classification.

Pipeline:
  1. Load LIAR from HuggingFace, binarize labels
  2. Tokenise with DistilBertTokenizerFast (max_length=128)
  3. Fine-tune for 3 epochs using HuggingFace Trainer
  4. Evaluate on test set (accuracy, F1, precision, recall)
  5. Save confusion matrix to results/

Run independently:
    python models/distilbert_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "results")
CKPT_DIR      = os.path.join(_PROJECT_ROOT, "data", "distilbert_checkpoints")
DATA_DIR      = os.path.join(_PROJECT_ROOT, "data")

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
NUM_EPOCHS  = 3
LEARN_RATE  = 2e-5

# Raw TSV URLs (thiagorainmaker77/liar_dataset on GitHub)
LIAR_URLS = {
    "train":      "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    "validation": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
    "test":       "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
}

# String label → binary: credible=0, misinformation=1
CREDIBLE_LABELS = {"true", "mostly-true", "half-true"}


# ---------------------------------------------------------------------------
# Label binarisation
# ---------------------------------------------------------------------------

def binarize_label(label: str) -> int:
    """Map LIAR string label → binary (0=credible, 1=misinformation)."""
    return 0 if label.strip() in CREDIBLE_LABELS else 1


# ---------------------------------------------------------------------------
# Data loading + tokenisation
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


def load_and_tokenize():
    """
    Download LIAR TSVs, binarize string labels, and tokenise for DistilBERT.

    Returns
    -------
    tokenized_dataset : DatasetDict with 'train', 'validation', 'test' splits
                        containing input_ids, attention_mask, labels tensors
    tokenizer         : DistilBertTokenizerFast (needed for inference later)
    """
    print("Loading LIAR dataset from GitHub TSVs...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    split_datasets = {}
    for split_name in ("train", "validation", "test"):
        df     = _load_liar_split(split_name)
        texts  = df[2].fillna("").astype(str).tolist()   # column 2 = statement
        labels = [binarize_label(lbl) for lbl in df[1]]  # column 1 = label string
        split_datasets[split_name] = Dataset.from_dict({"statement": texts, "labels": labels})

    dataset = DatasetDict(split_datasets)

    def preprocess_batch(batch):
        """Tokenise statements; labels column is already present."""
        encoding = tokenizer(
            batch["statement"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoding["labels"] = batch["labels"]
        return encoding

    print("Tokenising splits (this may take a minute)...")
    tokenized = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=["statement"],
        desc="Tokenising",
    )
    tokenized.set_format("torch")

    for split_name, split_data in tokenized.items():
        labels = list(split_data["labels"])
        n_mis  = sum(labels)
        print(f"  {split_name:12s}: {len(labels)} samples  "
              f"({n_mis} misinfo / {len(labels) - n_mis} credible)")

    return tokenized, tokenizer


# ---------------------------------------------------------------------------
# Metric computation (called by Trainer)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall from Trainer eval output."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy":  float(accuracy_score(labels, predictions)),
        "f1":        float(f1_score(labels, predictions, average="binary")),
        "precision": float(precision_score(labels, predictions, average="binary", zero_division=0)),
        "recall":    float(recall_score(labels, predictions, average="binary")),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_distilbert(tokenized_dataset) -> tuple:
    """
    Fine-tune DistilBERT using HuggingFace Trainer.

    Trains for NUM_EPOCHS epochs, evaluates each epoch on validation set,
    and reloads the best checkpoint (by F1) at the end.

    Returns
    -------
    trainer : fitted Trainer object (use for prediction)
    model   : best DistilBertForSequenceClassification
    """
    print(f"\nFine-tuning {MODEL_NAME} for {NUM_EPOCHS} epochs...")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARN_RATE,
        # 'eval_strategy' is the current name (transformers >= 4.41);
        # fall back to 'evaluation_strategy' is handled by the same kwarg below
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=RANDOM_SEED,
        logging_steps=100,
        logging_dir=os.path.join(RESULTS_DIR, "logs"),
        report_to="none",           # disable W&B / TensorBoard
        dataloader_num_workers=0,   # avoids multiprocessing issues on some systems
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("  Training complete.")

    return trainer, model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_distilbert(
    trainer: Trainer,
    test_dataset,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Run inference on the test split and compute metrics.

    Returns
    -------
    metrics : dict (accuracy, f1, precision, recall)
    y_true  : np.ndarray of ground-truth labels
    y_pred  : np.ndarray of predicted labels
    """
    print("\nEvaluating DistilBERT on test set...")
    pred_output = trainer.predict(test_dataset)

    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=-1)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="binary")),
    }

    print("\n--- DistilBERT Results (test) ---")
    for name, val in metrics.items():
        print(f"  {name:12s}: {val:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Credible", "Misinformation"]))

    return metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Save a labelled confusion-matrix heatmap (Oranges palette)."""
    cm = confusion_matrix(y_true, y_pred)

    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=["Credible", "Misinformation"],
        yticklabels=["Credible", "Misinformation"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("DistilBERT – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Full pipeline (entry point)
# ---------------------------------------------------------------------------

def run_distilbert_pipeline():
    """
    Execute the complete DistilBERT fine-tuning pipeline end-to-end.

    Returns
    -------
    trainer       : Trainer (used for cross-domain prediction in main.py)
    model         : best DistilBertForSequenceClassification
    tokenizer     : DistilBertTokenizerFast (needed for new data inference)
    metrics       : dict (accuracy, f1, precision, recall) on LIAR test set
    y_true, y_pred: np.ndarray ground-truth and predicted labels
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    tokenized_dataset, tokenizer = load_and_tokenize()

    trainer, model = train_distilbert(tokenized_dataset)

    metrics, y_true, y_pred = evaluate_distilbert(trainer, tokenized_dataset["test"])

    print("\nSaving result artefacts...")
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RESULTS_DIR, "distilbert_confusion_matrix.png"),
    )

    with open(os.path.join(RESULTS_DIR, "distilbert_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'distilbert_metrics.json')}")

    return trainer, model, tokenizer, metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_distilbert_pipeline()
