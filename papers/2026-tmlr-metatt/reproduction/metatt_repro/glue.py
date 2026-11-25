"""GLUE data (Hugging Face `nyu-mll/glue`) and metrics, without scipy/sklearn.

The GLUE test labels are hidden (-1), so, as in LoRA / LoTR / the paper, models are evaluated on the validation split.
"""
from __future__ import annotations

import numpy as np
import torch

# task -> (text keys, number of labels, validation split, main metric)
TASKS = {
    "mrpc": (("sentence1", "sentence2"), 2, "validation", "accuracy"),
    "rte": (("sentence1", "sentence2"), 2, "validation", "accuracy"),
    "cola": (("sentence", None), 2, "validation", "mcc"),
    "sst2": (("sentence", None), 2, "validation", "accuracy"),
    "qnli": (("question", "sentence"), 2, "validation", "accuracy"),
    "qqp": (("question1", "question2"), 2, "validation", "accuracy"),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched", "accuracy"),
    "stsb": (("sentence1", "sentence2"), 1, "validation", "spearman"),
}


def num_labels(task: str) -> int:
    return TASKS[task][1]


def main_metric(task: str) -> str:
    return TASKS[task][3]


def load(task: str, tokenizer, max_len: int, limit_train: int | None = None, limit_eval: int | None = None):
    """Tokenised train and validation sets as dicts of python lists (input_ids, labels)."""
    from datasets import load_dataset

    (k1, k2), _, val_split, _ = TASKS[task]
    ds = load_dataset("nyu-mll/glue", task)
    out = {}
    for split, limit in (("train", limit_train), (val_split, limit_eval)):
        d = ds[split]
        if limit:
            d = d.select(range(min(limit, len(d))))
        a = list(d[k1])
        b = list(d[k2]) if k2 else None
        enc = tokenizer(a, b, truncation=True, max_length=max_len)
        out["train" if split == "train" else "eval"] = {"input_ids": enc["input_ids"], "labels": list(d["label"])}
    return out


def collate(examples: dict, idx, pad_id: int, regression: bool = False) -> dict:
    """Pad the selected examples to the longest one (dynamic padding, as HF's DataCollatorWithPadding)."""
    seqs = [examples["input_ids"][i] for i in idx]
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[j, :len(s)] = 1
    lab = [examples["labels"][i] for i in idx]
    labels = torch.tensor(lab, dtype=torch.float if regression else torch.long)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                         # average ranks of ties
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def metrics(task: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    """Percentages, as in the paper's tables. preds are class ids (or scores for STS-B)."""
    if task == "stsb":
        p = float(np.corrcoef(preds, labels)[0, 1])
        s = float(np.corrcoef(_rank(preds), _rank(labels))[0, 1])
        return {"pearson": 100 * p, "spearman": 100 * s}
    out = {"accuracy": 100 * float((preds == labels).mean())}
    if task in ("mrpc", "qqp"):
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        out["f1"] = 100 * (2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    if task == "cola":
        tp = float(((preds == 1) & (labels == 1)).sum()); tn = float(((preds == 0) & (labels == 0)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum()); fn = float(((preds == 0) & (labels == 1)).sum())
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out["mcc"] = 100 * ((tp * tn - fp * fn) / den if den else 0.0)
    return out
