"""One fine-tuning run: RoBERTa + MetaTT-4D on one GLUE task with one seed.

The paper trains with the Hugging Face Trainer (Appendix D.3 / G.3). This loop reproduces the Trainer's defaults
explicitly so that the run does not depend on the installed transformers version:
  AdamW (betas 0.9/0.999, eps 1e-8), linear decay with linear warm-up over ceil(warmup_ratio * total) steps,
  gradient clipping at global norm 1.0, data reshuffled every epoch, no dropped last batch, dropout on during
  training, evaluation on the validation set after every epoch.

Usage: python -m metatt_repro.train --cfg '<json of RunConfig>' --out <run dir>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import glue
from metatt import MetaTT4DConfig, apply_metatt, count_trainable, metatt4d_num_params


@dataclass
class RunConfig:
    task: str
    seed: int
    alpha: float
    lr: float
    batch_size: int
    warmup_ratio: float
    model: str = "FacebookAI/roberta-base"
    rank: int = 8
    target_modules: tuple = ("query", "value")
    init: str = "ze-id-id-id"
    epochs: int = 20
    max_len: int = 256
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "fp32"            # fp32 (no TF32), tf32 (TF32 matmuls), bf16 (bf16 autocast)
    eval_batch_size: int = 64
    limit_train: int | None = None     # smoke tests only
    limit_eval: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        d["target_modules"] = tuple(d.get("target_modules", cls.target_modules))
        return cls(**d)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_precision(precision: str) -> None:
    if precision not in ("fp32", "tf32", "bf16"):
        raise ValueError(f"unknown precision {precision!r}")
    tf32 = precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def autocast(precision: str, device: torch.device):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def linear_schedule(warmup_steps: int, total_steps: int):
    """Same formula as transformers.get_linear_schedule_with_warmup."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return f


@torch.no_grad()
def frozen_checksum(model: torch.nn.Module) -> float:
    return float(sum(p.detach().double().sum().item() for p in model.parameters() if not p.requires_grad))


@torch.no_grad()
def evaluate(model, data: dict, task: str, device, pad_id: int, batch_size: int, precision: str) -> dict:
    model.eval()
    reg = glue.num_labels(task) == 1
    n = len(data["labels"])
    preds = []
    for s in range(0, n, batch_size):
        batch = glue.collate(data, range(s, min(n, s + batch_size)), pad_id, reg)
        batch.pop("labels")
        with autocast(precision, device):
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float()
        preds.append(logits.squeeze(-1).cpu().numpy() if reg else logits.argmax(-1).cpu().numpy())
    return glue.metrics(task, np.concatenate(preds), np.asarray(data["labels"]))


def fit(model, data: dict, cfg: RunConfig, device, pad_id: int, log=print) -> tuple[list, dict]:
    """Train the trainable parameters of `model` (the MetaTT cores). Returns (per-epoch history, training info)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay)
    n = len(data["train"]["labels"])
    bs = cfg.batch_size
    steps_per_epoch = math.ceil(n / bs)
    total = steps_per_epoch * cfg.epochs
    warmup = math.ceil(total * cfg.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_schedule(warmup, total))
    gen = torch.Generator().manual_seed(cfg.seed)
    reg = glue.num_labels(cfg.task) == 1
    frozen0 = frozen_checksum(model)

    ev = evaluate(model, data["eval"], cfg.task, device, pad_id, cfg.eval_batch_size, cfg.precision)
    history = [{"epoch": 0, "train_loss": None, "lr": 0.0, "sec": 0.0, **ev}]   # before training (frozen random head)
    log(f"epoch 0 (no training): {ev}")
    t_train = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        perm = torch.randperm(n, generator=gen).tolist()
        for s in range(steps_per_epoch):
            batch = glue.collate(data["train"], perm[s * bs:(s + 1) * bs], pad_id, reg)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(cfg.precision, device):
                loss = model(**batch).loss
            loss.backward()
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            loss_sum += loss.detach().float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        t_train += sec
        ev = evaluate(model, data["eval"], cfg.task, device, pad_id, cfg.eval_batch_size, cfg.precision)
        row = {"epoch": epoch, "train_loss": float(loss_sum) / steps_per_epoch, "lr": sched.get_last_lr()[0],
               "sec": sec, **ev}
        history.append(row)
        log(json.dumps(row))
    info = {"steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup,
            "train_sec": t_train, "frozen_unchanged": frozen_checksum(model) == frozen0}
    return history, info


def summarize(history: list, task: str) -> dict:
    key = glue.main_metric(task)
    trained = [h for h in history if h["epoch"] > 0]
    best = max(trained, key=lambda h: h[key])
    out = {"metric": key, "best": best[key], "best_epoch": best["epoch"], "final": trained[-1][key],
           "before_training": history[0][key]}
    if "f1" in best:
        out["f1_at_best"] = best["f1"]
    return out


def environment(device) -> dict:
    import transformers
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "device": str(device)}
    try:
        import datasets
        env["datasets"] = datasets.__version__
    except ImportError:
        pass
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def run(cfg: RunConfig, out_dir: str) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_precision(cfg.precision)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    data = glue.load(cfg.task, tok, cfg.max_len, cfg.limit_train, cfg.limit_eval)
    set_seed(cfg.seed)                       # the (frozen, randomly initialised) classifier head depends on the seed
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=glue.num_labels(cfg.task))
    model = model.float()
    adapter = apply_metatt(model, MetaTT4DConfig(rank=cfg.rank, alpha=cfg.alpha,
                                                 target_modules=tuple(cfg.target_modules), init=cfg.init))
    model.to(device)
    d_in, d_out = adapter.G1.shape[0], adapter.G4.shape[1]
    params = {"trainable": count_trainable(model),
              "formula": metatt4d_num_params(d_in, d_out, adapter.G2.shape[0], adapter.G3.shape[0], cfg.rank),
              "L": adapter.G2.shape[0], "M": adapter.G3.shape[0], "D": d_in}
    assert params["trainable"] == params["formula"], params

    log_path = os.path.join(out_dir, "log.jsonl")
    with open(log_path, "w", encoding="utf-8") as logf:
        def log(msg):
            print(msg, flush=True)
            if msg.startswith("{"):
                logf.write(msg + "\n")
                logf.flush()
        log(f"run {cfg.task} seed={cfg.seed} warmup={cfg.warmup_ratio} trainable={params['trainable']} "
            f"train={len(data['train']['labels'])} eval={len(data['eval']['labels'])} device={device}")
        t0 = time.time()
        history, info = fit(model, data, cfg, device, tok.pad_token_id, log)
    result = {"cfg": asdict(cfg), "summary": summarize(history, cfg.task), "params": params, "train": info,
              "history": history, "env": environment(device), "wall_sec": time.time() - t0,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = os.path.join(out_dir, "result.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "result.json"))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="RunConfig as JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = RunConfig.from_dict(json.loads(a.cfg))
    try:
        r = run(cfg, a.out)
    except Exception:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "error.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        traceback.print_exc()
        return 1
    s = r["summary"]
    print(f"done: best {s['metric']} {s['best']:.2f} (epoch {s['best_epoch']}), final {s['final']:.2f}, "
          f"{r['wall_sec'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
