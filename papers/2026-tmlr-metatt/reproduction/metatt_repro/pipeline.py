"""Runs the minimal experiment (Table 2, RoBERTa-base, MetaTT-4D r=8, MRPC and RTE, 3 seeds) as parallel worker
processes on one GPU. Finished runs (result.json with the same config) are skipped, so the command can be repeated
after a disconnect.

  python -m metatt_repro.pipeline --out OUT [--tasks mrpc,rte] [--warmups 0.06,0.6] [--parallel 6]
  smoke test: --epochs 1 --limit_train 64 --limit_eval 64 --seeds 42
Layout: OUT/runs/<task>/w<warmup>_s<seed>/{result.json, log.jsonl, error.txt}, OUT/logs/<run>.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from . import paper
from .train import RunConfig


def build_runs(tasks, seeds, warmups, **overrides) -> list[RunConfig]:
    base = {k: v for k, v in paper.COMMON.items()}
    runs = []
    for w in warmups:
        for task in tasks:
            hp = paper.TASK_HPARAMS[task]
            for seed in seeds:
                d = dict(base, task=task, seed=seed, warmup_ratio=w, alpha=hp["alpha"], lr=hp["lr"],
                         batch_size=hp["batch_size"])
                d.update({k: v for k, v in overrides.items() if v is not None})
                runs.append(RunConfig.from_dict(d))
    return runs


def run_name(cfg: RunConfig) -> str:
    name = f"w{cfg.warmup_ratio:g}_s{cfg.seed}"
    if cfg.limit_train or cfg.limit_eval or cfg.epochs != paper.COMMON["epochs"]:
        name += f"_smoke-e{cfg.epochs}-n{cfg.limit_train}"
    if cfg.precision != "fp32":
        name += f"_{cfg.precision}"
    return name


def run_dir(out: str, cfg: RunConfig) -> str:
    return os.path.join(out, "runs", cfg.task, run_name(cfg))


def is_done(out: str, cfg: RunConfig) -> bool:
    p = os.path.join(run_dir(out, cfg), "result.json")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        old = json.load(f)["cfg"]
    return RunConfig.from_dict(old) == cfg


def prefetch(model: str, tasks) -> None:
    """Download the model and the data once, before the workers start."""
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained(model)
    AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
    for t in tasks:
        load_dataset("nyu-mll/glue", t)


def _pythonpath() -> str:
    import metatt
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # folder holding metatt_repro/
    method = os.path.dirname(os.path.dirname(os.path.abspath(metatt.__file__)))  # folder holding metatt/
    parts = [here, method] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    return os.pathsep.join(dict.fromkeys(parts))


def launch(cfg: RunConfig, out: str) -> tuple[RunConfig, int, float]:
    d = run_dir(out, cfg)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    err = os.path.join(d, "error.txt")
    if os.path.exists(err):
        os.remove(err)
    log = os.path.join(out, "logs", f"{cfg.task}_{run_name(cfg)}.log")
    env = dict(os.environ, PYTHONPATH=_pythonpath(), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-m", "metatt_repro.train", "--cfg", json.dumps(asdict(cfg)),
                             "--out", d], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return cfg, rc, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="mrpc,rte")
    ap.add_argument("--seeds", default=None, help="default: the paper's seeds for the model")
    ap.add_argument("--warmups", default=",".join(f"{w:g}" for w in paper.WARMUPS))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--hours", type=float, default=None, help="do not start new runs after this many hours")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_eval", type=int, default=None)
    ap.add_argument("--no_prefetch", action="store_true")
    a = ap.parse_args(argv)

    tasks = [t for t in a.tasks.split(",") if t]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(paper.SEEDS[paper.COMMON["model"]])
    warmups = [float(w) for w in a.warmups.split(",")]
    runs = build_runs(tasks, seeds, warmups, precision=a.precision, epochs=a.epochs,
                      limit_train=a.limit_train, limit_eval=a.limit_eval)
    todo = [r for r in runs if not is_done(a.out, r)]
    print(f"{len(runs)} runs, {len(runs) - len(todo)} already finished, {len(todo)} to run "
          f"({a.parallel} at a time) -> {a.out}", flush=True)
    if not todo:
        return 0
    if not a.no_prefetch:
        prefetch(paper.COMMON["model"], tasks)
        os.environ["HF_HUB_OFFLINE"] = "1"         # workers read the cache only (no parallel hub requests)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    deadline = time.time() + 3600 * a.hours if a.hours is not None else None
    failed = skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as pool:
        futures = []
        for cfg in todo:
            futures.append(pool.submit(lambda c=cfg: launch(c, a.out) if not deadline or time.time() < deadline
                                       else (c, None, 0.0)))
        for fut in as_completed(futures):
            cfg, rc, sec = fut.result()
            tag = f"{cfg.task} warmup={cfg.warmup_ratio:g} seed={cfg.seed}"
            if rc is None:
                print(f"skipped (time budget): {tag}", flush=True)
                skipped += 1
                continue
            if rc == 0:
                with open(os.path.join(run_dir(a.out, cfg), "result.json"), encoding="utf-8") as f:
                    s = json.load(f)["summary"]
                print(f"done {tag}: best {s['metric']} {s['best']:.2f} (epoch {s['best_epoch']}), "
                      f"final {s['final']:.2f}  [{sec / 60:.1f} min]", flush=True)
            else:
                failed += 1
                print(f"FAILED {tag} (exit {rc}); last lines of the log:", flush=True)
                log = os.path.join(a.out, "logs", f"{cfg.task}_{run_name(cfg)}.log")
                with open(log, encoding="utf-8", errors="replace") as f:
                    print("".join(f.readlines()[-15:]), flush=True)
    print(f"finished: {len(todo) - failed - skipped} ok, {failed} failed, {skipped} not started (time budget; "
          f"run again to continue)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
