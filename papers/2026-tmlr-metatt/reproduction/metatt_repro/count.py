"""Checks the trainable-parameter counts of the MetaTT-4D rows of Table 2 without training or downloading weights:
builds RoBERTa from its config (on the meta device, so no memory is used), applies MetaTT-4D and counts.

  python -m metatt_repro.count [--out param_check.md]
"""
from __future__ import annotations

import argparse
import contextlib

import torch

from . import paper
from metatt import MetaTT4DConfig, apply_metatt, count_trainable, metatt4d_num_params


def build(model_name: str):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(model_name, num_labels=2)
    for ctx in (torch.device("meta"), contextlib.nullcontext()):   # fall back to CPU if meta is not supported
        try:
            with ctx:
                return AutoModelForSequenceClassification.from_config(cfg), cfg
        except Exception:
            continue
    raise RuntimeError(f"cannot build {model_name}")


def check() -> list[dict]:
    out = []
    for (name, rank), printed_k in paper.TABLE2_PARAMS_K.items():
        model, hf_cfg = build(name)                    # fresh model per row (apply_metatt modifies it)
        apply_metatt(model, MetaTT4DConfig(rank=rank, target_modules=paper.COMMON["target_modules"]))
        n = count_trainable(model)
        d, L = hf_cfg.hidden_size, hf_cfg.num_hidden_layers
        f = metatt4d_num_params(d, d, L, 2, rank)
        out.append({"model": name.split("/")[-1], "rank": rank, "counted": n, "formula": f, "paper_k": printed_k,
                    "match": n == f and round(n / 1000) == printed_k})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = check()
    lines = ["| モデル | ランク | 数えた値 | 式 2Dr+(L+M)r² | 論文（×10³） | 一致 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['rank']} | {r['counted']:,} | {r['formula']:,} | {r['paper_k']} | "
              f"{'✓' if r['match'] else '✗'} |" for r in rows]
    text = "# 表 2 の MetaTT-4D のパラメータ数（Q・V に適用、M = 2）\n\n" + "\n".join(lines) + "\n"
    print(text)
    if a.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if all(r["match"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
