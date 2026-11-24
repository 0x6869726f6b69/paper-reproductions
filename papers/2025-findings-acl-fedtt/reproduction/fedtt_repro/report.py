"""Collects finished runs and compares them with the paper.

For every (suite, setting, method, task):
  best    = best GLUE-validation score over rounds, maximised over the learning-rate / batch-size grid
            (the paper's protocol as written: "we record the best validation results"; optimistic);
  sel     = the GLUE-validation score at the round and grid point with the best held-out (FedPETuning dev)
            score (selection without looking at the evaluation set);
  paper   = the paper's number, delta = best - paper;
  r95     = first round whose validation score reaches 95 % of the paper's score (Tables 14/15 definition).
With several seeds the mean (and std) over seeds is shown.

  python -m fedtt_repro.report --out OUT
"""
import argparse
import glob
import json
import os

import numpy as np

from . import paper

TASKS = paper.TASK_ORDER


def load_runs(out_root):
    rows = []
    for p in glob.glob(os.path.join(out_root, "runs", "*", "*", "*", "result.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        c = r["cfg"]
        rows.append(dict(suite=c["suite"], task=c["task"], method=c["method"], rank=c["rank"],
                         key=f"{c['method']}-r{c['rank']}", level=_level(c), lr=c["lr"], bs=c["bs"], seed=c["seed"],
                         rounds=c["rounds"], best=r["summary"]["best_val"], best_round=r["summary"]["best_val_round"],
                         sel=r["summary"]["val_at_best_heldout"],
                         heldout_best=max([x["heldout"] for x in r["log"] if x.get("heldout") is not None],
                                          default=None),
                         kb=r["summary"]["up_kb_mean"], up=r["summary"]["up_params_mean"],
                         curve=[(x["round"], x["val"]) for x in r["log"] if "val" in x],
                         hours=r["summary"]["train_sec"] / 3600, gpu=r["env"].get("gpu", r["env"].get("cpu", "")),
                         run_id=r["run_id"]))
    return rows


def _level(c):
    if c["suite"] == "t3":
        return c.get("tag") or {3: "iid", 20: "mild", 50: "severe"}.get(c["n_clients"], "")
    if c["suite"] == "t13":
        return c["partition"]
    return ""


def paper_value(suite, level, key, task):
    tab = {"t1": paper.T1, "t2": paper.T2, "t7": paper.T7}.get(suite)
    if suite == "t3":
        tab = paper.T3.get(level, {})
    if suite == "t13":
        tab = paper.T13.get(level, {})
    if tab is None:
        return None
    return tab.get(key, {}).get(task)


def r95(curve, target):
    if target is None:
        return None
    for t, v in curve:
        if v >= 0.95 * target:
            return t
    return None


def summarize(rows):
    """One entry per (suite, level, key, task): grid-maximised and held-out-selected scores over seeds."""
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["suite"], r["level"], r["key"], r["task"], r["rounds"]), []).append(r)
    for (suite, level, key, task, rounds), rs in groups.items():
        seeds = sorted({r["seed"] for r in rs})
        best, sel, rr, kb, hrs = [], [], [], [], []
        target = paper_value(suite, level, key, task)
        for s in seeds:
            g = [r for r in rs if r["seed"] == s]
            b = max(g, key=lambda r: r["best"])
            best.append(b["best"])
            hs = [r for r in g if r["heldout_best"] is not None]
            if hs:
                sel.append(max(hs, key=lambda r: r["heldout_best"])["sel"])
            rr.append(r95(b["curve"], target))
            kb.append(b["kb"])
            hrs.append(sum(r["hours"] for r in g))
        out[(suite, level, key, task, rounds)] = dict(
            best=float(np.mean(best)), best_std=float(np.std(best)) if len(best) > 1 else None,
            sel=float(np.mean(sel)) if sel else None, n_seeds=len(seeds), n_grid=len(rs) // len(seeds),
            paper=target, r95=rr[0] if len(rr) == 1 else rr, kb=float(np.mean(kb)), hours=float(np.sum(hrs)),
            best_cfg=f"lr {max(rs, key=lambda r: r['best'])['lr']:g}, bs {max(rs, key=lambda r: r['best'])['bs']}")
    return out


def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(summary):
    lines = []
    suites = sorted({k[0] for k in summary}, key=lambda s: ["smoke", "t1", "t7", "t2", "t3", "t13"].index(s)
                    if s in ["smoke", "t1", "t7", "t2", "t3", "t13"] else 99)
    titles = dict(t1="Table 1 (DeBERTa-base, 5 clients, IID)", t7="Table 7 (DeBERTa-base, rank analysis, 20 rounds)",
                  t2="Table 2 (RoBERTa-base, FedPETuning setting)", t3="Table 3 (RoBERTa-large, 3/20/50 clients)",
                  t13="Table 13 (RoBERTa-large, 3 clients, label skew)", smoke="smoke (pipeline check only)")
    for suite in suites:
        keys = [k for k in summary if k[0] == suite]
        lines += [f"## {titles.get(suite, suite)}", ""]
        for level in sorted({k[1] for k in keys}, key=lambda x: ["", "iid", "mild", "severe"].index(x)
                            if x in ["", "iid", "mild", "severe"] else 9):
            for rounds in sorted({k[4] for k in keys if k[1] == level}):
                ks = [k for k in keys if k[1] == level and k[4] == rounds]
                tasks = [t for t in TASKS if any(k[3] == t for k in ks)]
                if level:
                    lines += [f"### {level}", ""]
                lines += [f"rounds = {rounds}. Each cell: **ours best** / paper (Δ) ; held-out-selected ; r95.", ""]
                lines += ["| method | " + " | ".join(tasks) + " | avg ours | avg paper |",
                          "|---|" + "---|" * (len(tasks) + 2)]
                for key in sorted({k[2] for k in ks}):
                    cells, ours, pap = [], [], []
                    for t in tasks:
                        s = summary.get((suite, level, key, t, rounds))
                        if s is None:
                            cells.append("–")
                            continue
                        d = None if s["paper"] is None else s["best"] - s["paper"]
                        std = f"±{s['best_std']:.2f}" if s["best_std"] else ""
                        cells.append(f"**{s['best']:.2f}**{std} / {_f(s['paper'])} ({'' if d is None else f'{d:+.2f}'})"
                                     f" ; {_f(s['sel'])} ; {s['r95'] if s['r95'] is not None else '–'}")
                        ours.append(s["best"])
                        if s["paper"] is not None:
                            pap.append(s["paper"])
                    full = len(ours) == len(tasks)
                    lines.append(f"| {key} | " + " | ".join(cells) +
                                 f" | {_f(np.mean(ours)) if full and ours else '–'} | "
                                 f"{_f(np.mean(pap)) if len(pap) == len(tasks) else '–'} |")
                lines.append("")
                lines += ["| method | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
                for key in sorted({k[2] for k in ks}):
                    cells = []
                    for t in tasks:
                        s = summary.get((suite, level, key, t, rounds))
                        cells.append("–" if s is None else f"{s['best_cfg']}; {s['n_grid']} cfg × {s['n_seeds']} seed; "
                                     f"{s['kb']:.0f} KB/round; {s['hours']:.1f} h")
                    lines.append(f"| {key} | " + " | ".join(cells) + " |")
                lines.append("")
    return lines


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    groups = {}
    for r in rows:
        if r["suite"] in ("t3", "t7", "t1") and r["seed"] == 0:
            groups.setdefault((r["suite"], r["level"], r["task"]), []).append(r)
    if not groups:
        return None
    keys = sorted(groups)
    n = len(keys)
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.5 * rows_n), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        best = {}
        for r in groups[k]:
            if r["key"] not in best or r["best"] > best[r["key"]]["best"]:
                best[r["key"]] = r
        for key, r in sorted(best.items()):
            t, v = zip(*r["curve"])
            ax.plot(t, v, label=key, lw=1.2)
        ax.set_title(f"{k[0]} {k[1]} {k[2]}".replace("  ", " "), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = load_runs(args.out)
    os.makedirs(os.path.join(args.out, "report"), exist_ok=True)
    if not rows:
        print("no finished runs yet")
        return
    summ = summarize(rows)
    lines = ["# FedTT / FedTT+ reproduction: our runs vs. the paper", "",
             f"{len(rows)} finished runs. 'best' = best GLUE-validation score over rounds, maximised over the "
             "lr/bs grid (paper protocol); 'held-out-selected' = validation score at the round/grid point with the "
             "best FedPETuning held-out score; r95 = first round reaching 95 % of the paper's score.", ""]
    lines += tables(summ)
    fig = plot_curves(rows, os.path.join(args.out, "report", "curves.png"))
    if fig:
        lines += [f"Curves (validation score per round, best grid point, seed 0): `{os.path.basename(fig)}`", ""]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report", "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    flat = [dict(suite=k[0], level=k[1], method=k[2], task=k[3], rounds=k[4], **{kk: vv for kk, vv in v.items()
                                                                                    if kk != "r95"}, r95=str(v["r95"]))
            for k, v in summ.items()]
    import csv
    with open(os.path.join(args.out, "report", "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(text)


if __name__ == "__main__":
    main()
