"""GLUE data, the FedPETuning train/dev split used by the paper (Table 8) and the client partitions.

Splits (FedPETuning, Zhang et al. 2023, which the paper follows; Table 8 train sizes):
  'train'   GLUE train minus a held-out part, so that |train| matches Table 8
            (MRPC 3,301 / SST-2 66,675 / QNLI 103,695 / QQP 360,210 / MNLI 388,774; RTE 2,241);
  'heldout' the rest of GLUE train (FedPETuning's dev set; 367 / 674 / 1,048 / 3,636 / 3,928 / 249);
  'val'     the GLUE validation set (MNLI: validation_matched) -- the set whose best score over rounds
            the paper reports ("we record the best validation results"). For DeBERTa on QQP the paper
            uses 1,000 random validation samples (cfg val_subsample=1000).
The split uses a fixed seed (42), independent of the run seed, so every run sees the same data.
"""
import os

import numpy as np

TASKS = {
    "mrpc": dict(keys=("sentence1", "sentence2"), n_cls=2, metric="f1", train_size=3301, val="validation"),
    "sst2": dict(keys=("sentence", None), n_cls=2, metric="acc", train_size=66675, val="validation"),
    "qnli": dict(keys=("question", "sentence"), n_cls=2, metric="acc", train_size=103695, val="validation"),
    "qqp": dict(keys=("question1", "question2"), n_cls=2, metric="acc", train_size=360210, val="validation"),
    "mnli": dict(keys=("premise", "hypothesis"), n_cls=3, metric="acc", train_size=388774,
                 val="validation_matched"),
    "rte": dict(keys=("sentence1", "sentence2"), n_cls=2, metric="acc", train_size=2241, val="validation"),
}

# Per-client label distributions (rows = clients) for the 3-client label-skew settings of Appendix B /
# Table 13 (copied from FFA-LoRA). 'severe' is the FedTT text ([0.05, 0.95]); FFA-LoRA's own paper uses
# [0.1, 0.9] for binary tasks, available as 'severe_ffa'.
RATIOS = {
    "mild": {2: [[0.15, 0.85], [0.85, 0.15], [0.5, 0.5]],
             3: [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]},
    "severe": {2: [[0.05, 0.95], [0.95, 0.05], [0.5, 0.5]],
               3: [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]},
    "severe_ffa": {2: [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]],
                   3: [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]},
}


class TokData:
    """Tokenized examples stored as one flat int32 array plus offsets; `idx` selects a subset."""

    def __init__(self, ids, off, labels, idx=None):
        self.ids, self.off, self.labels_all = ids, off, labels
        self.idx = np.arange(len(labels)) if idx is None else np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    @property
    def labels(self):
        return self.labels_all[self.idx]

    def lengths(self):
        return (self.off[self.idx + 1] - self.off[self.idx]).astype(np.int64)

    def subset(self, rel):
        """Subset by positions relative to this view."""
        return TokData(self.ids, self.off, self.labels_all, self.idx[np.asarray(rel)])

    def batch(self, rel, pad_id):
        """Padded (input_ids, attention_mask, labels) as numpy arrays for positions `rel` of this view."""
        j = self.idx[np.asarray(rel)]
        lens = self.off[j + 1] - self.off[j]
        L = int(lens.max())
        x = np.full((len(j), L), pad_id, dtype=np.int64)
        m = np.zeros((len(j), L), dtype=np.int64)
        for i, (a, b) in enumerate(zip(self.off[j], self.off[j + 1])):
            x[i, :b - a] = self.ids[a:b]
            m[i, :b - a] = 1
        return x, m, self.labels_all[j].astype(np.int64)


def _slug(name):
    return name.replace("/", "__")


def _tokenize(texts_a, texts_b, tok, max_len, chunk=20000):
    ids, lens = [], []
    for s in range(0, len(texts_a), chunk):
        a = texts_a[s:s + chunk]
        b = None if texts_b is None else texts_b[s:s + chunk]
        enc = tok(a, b, truncation=True, max_length=max_len)["input_ids"]
        for e in enc:
            ids.append(np.asarray(e, dtype=np.int32))
            lens.append(len(e))
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    return np.concatenate(ids) if ids else np.zeros(0, np.int32), off


def load_glue(task, tok, tok_name, max_len, cache_dir):
    """Returns {'train_full': TokData, 'val': TokData} (cached as .npz under cache_dir)."""
    spec = TASKS[task]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{_slug(tok_name)}_{task}_L{max_len}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {s: TokData(z[f"{s}_ids"], z[f"{s}_off"], z[f"{s}_y"]) for s in ("train_full", "val")}
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", task)
    out, arrays = {}, {}
    for s, split in (("train_full", "train"), ("val", spec["val"])):
        d = ds[split]
        ka, kb = spec["keys"]
        ids, off = _tokenize(list(d[ka]), None if kb is None else list(d[kb]), tok, max_len)
        y = np.asarray(d["label"], dtype=np.int64)
        arrays.update({f"{s}_ids": ids, f"{s}_off": off, f"{s}_y": y})
        out[s] = TokData(ids, off, y)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    _replace(tmp, path)
    return out


def _replace(src, dst, tries=20):
    import time
    for k in range(tries):          # OneDrive / Google Drive may hold a lock for a moment
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    os.replace(src, dst)


def fedpet_split(task, train_full, split_seed=42):
    """(train, heldout) with |train| = Table 8 size, rest held out (FedPETuning)."""
    n = len(train_full)
    n_train = min(TASKS[task]["train_size"], n)
    perm = np.random.default_rng(split_seed).permutation(n)
    return train_full.subset(perm[n - n_train:]), train_full.subset(perm[:n - n_train])


def subsample(data, k, seed=42):
    if k is None or k >= len(data):
        return data
    return data.subset(np.sort(np.random.default_rng(seed).choice(len(data), int(k), replace=False)))


# ---------------------------------------------------------------------------------------------- partitions
def partition(labels, n_clients, kind, rng, alpha=1.0, n_cls=None, min_size=None):
    """List of index arrays (positions in `labels`), one per client.

    'iid'                 uniform random split into equal parts;
    'dir'                 label skew, per class p_c ~ Dir(alpha * 1_N) over clients with the NIID-Bench / FedLab
                          balancing rule, redrawn until every client has >= min_size samples (FedLab's
                          hetero_dir_partition default: min_size = number of classes; FedPETuning uses FedLab
                          with alpha = 1.0);
    'mild' | 'severe' | 'severe_ffa'   3-client label-ratio splits of Appendix B (RATIOS above).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if kind == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), n_clients)]
    classes = np.unique(labels) if n_cls is None else np.arange(n_cls)
    if kind == "dir":
        min_size = len(classes) if min_size is None else min_size
        for _ in range(1000):
            batches = [[] for _ in range(n_clients)]
            for c in classes:
                idx_c = np.flatnonzero(labels == c)
                rng.shuffle(idx_c)
                p = rng.dirichlet(alpha * np.ones(n_clients))
                sizes = np.array([len(b) for b in batches])
                p = p * (sizes < n / n_clients)
                p = p / p.sum()
                cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
                for b, part in zip(batches, np.split(idx_c, cuts)):
                    b.extend(part.tolist())
            if min(len(b) for b in batches) >= min_size:
                return [np.sort(np.asarray(b, dtype=np.int64)) for b in batches]
        raise RuntimeError("Dirichlet partition: could not reach the minimum client size")
    if kind in RATIOS:
        P = np.asarray(RATIOS[kind][len(classes)], dtype=float)
        if P.shape[0] != n_clients:
            raise ValueError(f"'{kind}' is defined for {P.shape[0]} clients")
        out = [[] for _ in range(n_clients)]
        for ci, c in enumerate(classes):
            idx_c = np.flatnonzero(labels == c)
            rng.shuffle(idx_c)
            w = P[:, ci] / P[:, ci].sum()      # share of class c sent to each client
            cuts = np.round(np.cumsum(w) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                out[k].extend(part.tolist())
        return [np.sort(np.asarray(b, dtype=np.int64)) for b in out]
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- metrics
def metrics(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    acc = float((pred == y).mean()) * 100
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y != 1)).sum())
    fn = int(((pred != 1) & (y == 1)).sum())
    f1 = 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)
    return {"acc": acc, "f1": f1}
