"""Collects OUT/runs/**/result.json and compares them with Table 2.

  python -m metatt_repro.report --out OUT
writes OUT/report/summary.csv (one row per run), OUT/report/report.md (mean +- standard error over seeds against
the paper) and OUT/report/curves.png (validation accuracy per epoch).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

from . import paper


def load_results(out: str) -> list[dict]:
    res = []
    for p in sorted(glob.glob(os.path.join(out, "runs", "*", "*", "result.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r["_dir"] = os.path.basename(os.path.dirname(p))
        res.append(r)
    return res


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def is_main(r: dict) -> bool:
    c = r["cfg"]
    return not (c.get("limit_train") or c.get("limit_eval")) and c["epochs"] == paper.COMMON["epochs"]


def rows(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        c, s = r["cfg"], r["summary"]
        out.append({"task": c["task"], "warmup": c["warmup_ratio"], "seed": c["seed"], "precision": c["precision"],
                    "smoke": not is_main(r), "best_acc": round(s["best"], 2), "best_epoch": s["best_epoch"],
                    "final_acc": round(s["final"], 2), "before_training": round(s["before_training"], 2),
                    "f1_at_best": round(s.get("f1_at_best", float("nan")), 2), "trainable": r["params"]["trainable"],
                    "head_frozen": r["train"]["frozen_unchanged"], "train_min": round(r["train"]["train_sec"] / 60, 1),
                    "gpu": r["env"].get("gpu", r["env"].get("device", "")), "torch": r["env"]["torch"],
                    "transformers": r["env"]["transformers"], "finished": r.get("finished", "")})
    return sorted(out, key=lambda x: (x["smoke"], x["warmup"], x["task"], x["seed"]))


def table(results: list[dict]) -> str:
    groups = defaultdict(list)
    for r in results:
        if is_main(r):
            c = r["cfg"]
            groups[(c["task"], c["warmup_ratio"], c["precision"])].append(r)
    lines = ["| タスク | warmup | 精度 | シード数 | 最良エポックの正解率（平均 ± 標準誤差） | 最終エポック | 論文 | 差（最良 − 論文） | 各シード（最良） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (task, w, prec), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        best = [r["summary"]["best"] for r in rs]
        final = [r["summary"]["final"] for r in rs]
        mb, sb = mean_se(best)
        mf, sf = mean_se(final)
        p = paper.PAPER_RESULTS.get(task)
        ptxt = f"{p['value']} ± {p['se']}" if p else "–"
        diff = f"{mb - p['value']:+.2f}" if p else "–"
        seeds = ", ".join(f"{r['cfg']['seed']}: {r['summary']['best']:.2f}（{r['summary']['best_epoch']}）" for r in
                          sorted(rs, key=lambda r: r["cfg"]["seed"]))
        lines.append(f"| {task.upper()} | {w:g} | {prec} | {len(rs)} | {mb:.2f} ± {sb:.2f} | {mf:.2f} ± {sf:.2f} | "
                     f"{ptxt} | {diff} | {seeds} |")
    return "\n".join(lines)


def plot(results: list[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    main = [r for r in results if is_main(r)]
    if not main:
        return False
    tasks = sorted({r["cfg"]["task"] for r in main})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 3.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for r in sorted((r for r in main if r["cfg"]["task"] == task), key=lambda r: (r["cfg"]["warmup_ratio"],
                                                                                         r["cfg"]["seed"])):
            h = [x for x in r["history"] if x["epoch"] > 0]
            ls = "-" if r["cfg"]["warmup_ratio"] < 0.3 else "--"
            ax.plot([x["epoch"] for x in h], [x["accuracy"] for x in h], ls,
                    label=f"warmup {r['cfg']['warmup_ratio']:g}, seed {r['cfg']['seed']}")
        p = paper.PAPER_RESULTS.get(task)
        if p:
            ax.axhline(p["value"], color="k", lw=0.8)
            ax.axhspan(p["value"] - p["se"], p["value"] + p["se"], color="k", alpha=0.1)
        ax.set_title(f"{task.upper()} (RoBERTa-base, MetaTT-4D r=8)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    results = load_results(a.out)
    rep = os.path.join(a.out, "report")
    os.makedirs(rep, exist_ok=True)
    if not results:
        print("no finished runs under", a.out)
        return 1
    rs = rows(results)
    with open(os.path.join(rep, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    md = ["# MetaTT-4D（r=8）× RoBERTa-base：論文の表 2 との比較", "",
          "論文の値は表 2（v2・v3 で同じ）。括弧ではなく ± で標準誤差を示す。正解率は GLUE の検証データ。", "",
          table(results), "",
          f"- 終わったラン：{sum(not r['smoke'] for r in rs)} 本（スモーク {sum(r['smoke'] for r in rs)} 本）",
          f"- 学習したパラメータ数：{sorted({r['trainable'] for r in rs})}（論文 {paper.PAPER_PARAMS_K}k）",
          f"- 分類ヘッドが学習で変わっていない：{all(r['head_frozen'] for r in rs)}",
          f"- 参考（論文の値ではない）：多数派クラスだけを答えたときの正解率 "
          + "、".join(f"{t.upper()} {v}" for t, v in paper.MAJORITY_BASELINE.items())
          + "。最良の値がこれと同じランは、何も学習できていない",
          f"- 環境：{sorted({r['gpu'] for r in rs})}、torch {sorted({r['torch'] for r in rs})}、"
          f"transformers {sorted({r['transformers'] for r in rs})}"]
    with open(os.path.join(rep, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    plotted = plot(results, os.path.join(rep, "curves.png"))
    print("\n".join(md))
    print("\nwrote", os.path.join(rep, "summary.csv"), os.path.join(rep, "report.md"),
          os.path.join(rep, "curves.png") if plotted else "(no plot: matplotlib missing or no full runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
