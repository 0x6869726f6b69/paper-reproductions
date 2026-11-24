"""Exact trainable-parameter and uplink counts of our implementation vs. the paper (Tables 1, 2, 7, 12, 13-15).

No training and no weights: the models are built from their Hugging Face configs (vocabulary shrunk, which
does not change any counted parameter). A count "matches" when it rounds to the paper's printed value
(0.06M = [0.055M, 0.065M), 39K = [38.5K, 39.5K)). Uplink KB = params * 4 B / 1024 (fp32), as in Table 14.

  python -m fedtt_repro.count [--out OUT/report/param_check.md]
"""
import argparse
import os
import re

from fedtt import model as M
from .suites import DEB, RB, RL

PAPER = [
    # (label, model, n_cls, method, rank, layers, paper value, source)
    ("DeBERTa-base FedTT r5 (binary)", DEB, 2, "fedtt", 5, "all", "0.06M", "Table 1 / 7"),
    ("DeBERTa-base FedTT r5 (MNLI)", DEB, 3, "fedtt", 5, "all", "0.06M", "Table 1"),
    ("DeBERTa-base FedTT+ r5 (binary)", DEB, 2, "fedttplus", 5, "all", "0.02M", "Table 1"),
    ("DeBERTa-base FedTT r2", DEB, 2, "fedtt", 2, "all", "0.03M", "Table 7"),
    ("DeBERTa-base FedTT r10", DEB, 2, "fedtt", 10, "all", "0.17M", "Table 7"),
    ("DeBERTa-base LoRA r8 (head not counted)", DEB, 2, "lora", 8, "all", "0.30M", "Table 1"),
    ("DeBERTa-base LoRA r4 (head not counted)", DEB, 2, "lora", 4, "all", "0.15M", "Table 1"),
    ("RoBERTa-base FedTT r5", RB, 2, "fedtt", 5, "all", "0.06M", "Table 2"),
    ("RoBERTa-large FedTT r5, last 6 (SST-2)", RL, 2, "fedtt", 5, "last6", "39K", "Table 12"),
    ("RoBERTa-large FedTT r5, last 9 (QNLI)", RL, 2, "fedtt", 5, "last9", "55K", "Table 12"),
    ("RoBERTa-large FedTT r5, last 9 (MNLI)", RL, 3, "fedtt", 5, "last9", "55K", "Table 12"),
    ("RoBERTa-large FedTT+ r5, all (SST-2)", RL, 2, "fedttplus", 5, "all", "28K", "Table 12"),
    ("RoBERTa-large FedTT+ r5, all (MNLI)", RL, 3, "fedttplus", 5, "all", "28K", "Table 12"),
    ("RoBERTa-large FedTT+ r5, all (binary)", RL, 2, "fedttplus", 5, "all", "0.03M", "Table 13"),
    ("RoBERTa-large LoRA r2, last 9 (head not counted)", RL, 2, "lora", 2, "last9", "74K", "Table 12"),
]


def parse(v):
    num, unit = float(v[:-1]), v[-1]
    return num * {"M": 1e6, "K": 1e3}[unit], unit, len(v.split(".")[1][:-1]) if "." in v else 0


def matches(n, v):
    x, unit, dec = parse(v)
    scale = {"M": 1e6, "K": 1e3}[unit]
    return round(n / scale, dec) == round(x / scale, dec)


def fmt(n, v):
    _, unit, dec = parse(v)
    scale = {"M": 1e6, "K": 1e3}[unit]
    return f"{n / scale:.{max(dec, 1) + 1}f}{unit}"


def config(name, n_cls):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(name, num_labels=n_cls)
    c.vocab_size = 8           # embeddings are frozen and never counted; keeps the build fast
    return c


def round_counts(fm, method, head=True):
    """Counts for rounds 1..6 (FedTT+ cycles through r; RoLoRA alternates)."""
    out = []
    for t in range(1, 7):
        fm.set_round(t)
        n = 0
        for name, p in fm.trainable():
            if not head and (name in fm.head_names or ".modules_to_save." in name):
                continue
            n += p.numel()
        out.append(n)
    return out


def build(name, n_cls, method, rank, layers):
    cfg = dict(model=name, method=method, rank=rank, adapter_layers=layers)
    return M.build_model(cfg, n_cls, hf_config=config(name, n_cls), pretrained=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    lines = ["# Trainable parameters / uplink per round: our implementation vs. the paper", "",
             "| setting | source | paper | ours (per round) | ours KB/round | match | breakdown (round 1) |",
             "|---|---|---|---|---|---|---|"]
    for label, name, n_cls, method, rank, layers, paper, src in PAPER:
        try:
            fm = build(name, n_cls, method, rank, layers)
        except Exception as e:   # e.g. peft refusing Colab's old torchao for the LoRA rows; keep the other rows
            err = f"{type(e).__name__}: {e}".replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {label} | {src} | {paper} | not built ({err}) | | | |")
            print(lines[-1], flush=True)
            continue
        head = "head not counted" not in label
        cnt = round_counts(fm, method, head)
        lo, hi = min(cnt), max(cnt)
        mean = sum(cnt[:6]) / 6
        shown = f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,} (mean {mean:,.0f})"
        ok = matches(mean, paper) or matches(hi, paper) or matches(lo, paper)
        okstr = "yes" if matches(mean, paper) else ("partly (min/max)" if ok else "**no**")
        bd = M.param_breakdown(fm, 1)
        bds = ", ".join(f"{k} {v:,}" for k, v in bd.items() if v)
        lines.append(f"| {label} | {src} | {paper} | {shown} = {fmt(mean, paper)} | {mean * 4 / 1024:.0f} | "
                     f"{okstr} | {bds} |")
        print(lines[-1], flush=True)
    # what FedTT+ would count if the adapter biases were trained too (they are frozen by default; see README)
    lines += ["", "FedTT+ variants (DeBERTa-base, binary):", ""]
    for tb in (False, True):
        fm = M.build_model(dict(model=DEB, method="fedttplus", rank=5, plus_train_bias=tb), 2,
                           hf_config=config(DEB, 2), pretrained=False)
        cnt = round_counts(fm, "fedttplus")
        lines.append(f"- plus_train_bias={tb}: rounds 1-6 {cnt} -> mean {sum(cnt) / 6:,.0f}")
    # Table 14/15 KB lines
    lines += ["", "Paper uplink KB (Table 14/15) are 4 B x the rounded counts / 1024: 0.06M -> 234, 0.02M -> 78, "
              "51K -> 199, 28K -> 109.", ""]
    text = "\n".join(lines) + "\n"
    print("\n".join(lines[-6:]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return text


if __name__ == "__main__":
    main()
