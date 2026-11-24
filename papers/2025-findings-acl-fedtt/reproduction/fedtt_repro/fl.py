"""Experiment runner for Algorithm 1 (FedTT) and Algorithm 2 (FedTT+): data, rounds, evaluation, logs.

The per-round logic (local updates and server averaging) is in method/fedtt/algorithm.py; this module
feeds it GLUE data, evaluates every round, writes log.jsonl / result.json and resumes from checkpoints.

Per communication round t (1-based):
  FedTT  (Alg. 1)  every selected client starts from the global trainable parameters w^(t), performs K local
                   updates (AdamW) on its data, and sends w_i^(t)+K; the server sets w^(t+1) = mean_i w_i^(t)+K
                   and broadcasts it.
  FedTT+ (Alg. 2)  the same, but in each tensorized adapter layer only {G_1, G_r, G_J} are trainable
                   (r from tt.plus_core_index), the other cores stay frozen and identical on all clients;
                   clients send and the server averages and broadcasts only {G_1, G_r, G_J} (plus the
                   classification head, which is always trainable).
Cross-silo: all N clients take part in every round. Large-scale cross-device (LSCD): `clients_per_round`
clients are drawn uniformly without replacement each round.

One model instance is reused for all clients (sequential simulation); clients are stateless (a fresh AdamW
per client and round), which is what the pseudocode describes.
"""
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from fedtt.algorithm import local_batches, run_round, select_clients  # noqa: F401  (method/fedtt)
from fedtt.model import build_model

DEFAULTS = dict(
    suite="adhoc", task="mrpc", model="microsoft/deberta-base", method="fedtt", rank=5,
    adapter_layers="all", bottleneck=64, act="relu", adapter_init_std=0.01, tt_mode="contract",
    tt_classifier=True, plus_schedule="text", plus_train_bias=False, lora_alpha=None,
    n_clients=5, clients_per_round=None, partition="iid", alpha=1.0,
    rounds=100, local_epochs=1.0, local_steps=None, bs=32, lr=1e-3, weight_decay=0.0, max_grad_norm=0.0,
    agg="uniform", max_len=128, eval_every=1, eval_bs=128, eval_heldout=True,
    val_subsample=None, heldout_subsample=None, train_subsample=None,
    seed=0, amp="auto", ckpt_every=10, threads=None, micro_bs=None, tag="",
)
# Fields shown in the run id; any other field that differs from DEFAULTS adds a short hash.
_ID_FIELDS = ("suite", "task", "model", "method", "rank", "adapter_layers", "n_clients", "clients_per_round",
              "partition", "alpha", "rounds", "local_epochs", "local_steps", "bs", "lr", "seed", "tag")
_NOT_IN_ID = ("eval_bs", "ckpt_every", "threads", "amp", "eval_every", "tt_mode", "micro_bs")


def full_cfg(cfg):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in cfg.items() if v is not None or k in ("clients_per_round", "local_steps",
                                                                       "local_epochs")})
    if out.get("lora_alpha") is None and out["method"] in ("lora", "ffa", "rolora"):
        out["lora_alpha"] = 2 * out["rank"]
    return out


def run_id(cfg):
    c = full_cfg(cfg)
    m = c["model"].split("/")[-1]
    lay = "" if c["adapter_layers"] == "all" else f"-{c['adapter_layers']}"
    who = f"N{c['n_clients']}" + (f"m{c['clients_per_round']}" if c["clients_per_round"] else "")
    part = c["partition"] + (f"{c['alpha']:g}" if c["partition"] == "dir" else "")
    loc = f"K{c['local_steps']}" if c["local_steps"] else f"E{c['local_epochs']:g}"
    rid = (f"{c['task']}/{m}_{c['method']}-r{c['rank']}{lay}_{who}-{part}-{loc}-T{c['rounds']}"
           f"_lr{c['lr']:g}-bs{c['bs']}-s{c['seed']}")
    extra = {k: v for k, v in c.items() if k not in _ID_FIELDS and k not in _NOT_IN_ID and DEFAULTS.get(k) != v}
    if c["tag"]:
        rid += f"-{c['tag']}"
    if extra:
        rid += "-" + hashlib.sha1(json.dumps(extra, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return rid


def run_dir(out_root, cfg):
    c = full_cfg(cfg)
    return os.path.join(out_root, "runs", c["suite"], *run_id(c).split("/"))


def env_info(device):
    d = dict(python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
             device=str(device))
    try:
        import transformers
        d["transformers"] = transformers.__version__
    except Exception:
        pass
    if device.type == "cuda":
        d["gpu"] = torch.cuda.get_device_name(0)
    else:
        d["cpu"] = platform.processor()
        d["threads"] = torch.get_num_threads()
    return d


def _write_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    D._replace(tmp, path)


def _amp_dtype(mode, device):
    if mode == "fp32" or device.type != "cuda":
        return None
    if mode in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def probe_amp(fm, data, device, pad_id, amp_dtype):
    """Some Hugging Face models fail under bf16 autocast (DeBERTa: masked_fill with float32's minimum on bf16
    attention scores -> "value cannot be converted to type BFloat16 without overflow"). Run one small batch;
    if it fails, fall back to fp32 (TF32 tensor-core matmuls on Ampere and newer GPUs).
    Returns (dtype or None, description)."""
    if amp_dtype is None:
        return None, "fp32"
    x, m, _ = data.batch(np.arange(min(4, len(data))), pad_id)
    try:
        fm.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype):
            fm(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
        return amp_dtype, str(amp_dtype).replace("torch.", "")
    except RuntimeError as e:
        return None, f"fp32 (tf32 matmuls); {str(amp_dtype).replace('torch.', '')} autocast failed: {str(e)[:100]}"


@torch.no_grad()
def evaluate(fm, data, bs, device, pad_id, amp_dtype):
    if data is None or len(data) == 0:
        return None
    fm.eval()
    order = np.argsort(data.lengths(), kind="stable")          # length-sorted batches (less padding)
    preds = np.empty(len(data), dtype=np.int64)
    for s in range(0, len(order), bs):
        rel = order[s:s + bs]
        x, m, _ = data.batch(rel, pad_id)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fm(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
        preds[rel] = logits.argmax(-1).cpu().numpy()
    return D.metrics(preds, data.labels)


def prepare_data(cfg, out_root, tok):
    task = cfg["task"]
    raw = D.load_glue(task, tok, cfg["model"], int(cfg["max_len"]), os.path.join(out_root, "cache"))
    train, heldout = D.fedpet_split(task, raw["train_full"])
    train = D.subsample(train, cfg.get("train_subsample"))
    heldout = D.subsample(heldout, cfg.get("heldout_subsample"))
    val = D.subsample(raw["val"], cfg.get("val_subsample"))
    return train, heldout, val


def run(cfg, out_root, deadline=None, verbose=True, device=None):
    """Runs (or resumes) one configuration; returns the result dict, or {'status': 'paused'} at the deadline."""
    cfg = full_cfg(cfg)
    rdir = run_dir(out_root, cfg)
    os.makedirs(rdir, exist_ok=True)
    res_path, ckpt_path = os.path.join(rdir, "result.json"), os.path.join(rdir, "ckpt.pt")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            return json.load(f)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = _amp_dtype(cfg["amp"], device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tok.pad_token_id
    train, heldout, val = prepare_data(cfg, out_root, tok)
    spec = D.TASKS[cfg["task"]]
    parts = D.partition(train.labels, int(cfg["n_clients"]), cfg["partition"],
                        np.random.default_rng([int(cfg["seed"]), 11]), float(cfg["alpha"]), spec["n_cls"])
    clients = [train.subset(p) for p in parts]

    fm = build_model(cfg, spec["n_cls"]).to(device)
    amp_dtype, amp_desc = probe_amp(fm, val, device, pad_id, amp_dtype)
    if verbose:
        print(f"[{run_id(cfg)}] device {device}, precision {amp_desc}", flush=True)
    params = dict(fm.hf.named_parameters())
    gnames = list(fm.global_names)
    gstate = {n: params[n].detach().clone() for n in gnames}
    log, start = [], 1
    log_path = os.path.join(rdir, "log.jsonl")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        gstate = {n: ck["gstate"][n].to(device) for n in gnames}
        log, start = ck["log"], ck["round"] + 1
        if verbose:
            print(f"[{run_id(cfg)}] resuming at round {start}", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:       # drop rows after the last checkpoint
        f.writelines(json.dumps(r) + "\n" for r in log)

    def save_ckpt(t):
        tmp = ckpt_path + f".tmp{os.getpid()}"
        torch.save({"gstate": {n: v.detach().cpu() for n, v in gstate.items()}, "log": log, "round": t}, tmp)
        D._replace(tmp, ckpt_path)

    metric = spec["metric"]
    t0_run = time.time()
    with open(os.path.join(rdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    for t in range(start, int(cfg["rounds"]) + 1):
        t0 = time.time()
        st = run_round(fm, gstate, clients, select_clients(cfg, t), cfg, t, device, pad_id, amp_dtype)
        sched, up_params = st["sched"], st["up_params"]
        train_sec = time.time() - t0
        row = dict(round=t, sched=sched, loss=st["loss"], steps=st["steps"], clients=st["clients"],
                   up_params=up_params, train_sec=round(train_sec, 2))
        if t % int(cfg["eval_every"]) == 0 or t == int(cfg["rounds"]):
            te = time.time()
            vm = evaluate(fm, val, int(cfg["eval_bs"]), device, pad_id, amp_dtype)
            hm = evaluate(fm, heldout, int(cfg["eval_bs"]), device, pad_id, amp_dtype) if cfg["eval_heldout"] else None
            row.update(val=vm[metric], val_acc=vm["acc"], val_f1=vm["f1"],
                       heldout=None if hm is None else hm[metric], eval_sec=round(time.time() - te, 2))
        log.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if verbose:
            ev = f" val {row['val']:.2f}" if "val" in row else ""
            hv = f" heldout {row['heldout']:.2f}" if row.get("heldout") is not None else ""
            print(f"[{run_id(cfg)}] t={t} {sched} loss {row['loss']:.4f}{ev}{hv} "
                  f"({train_sec:.0f}s, up {up_params})", flush=True)
        past = deadline is not None and time.time() > deadline
        if t % int(cfg["ckpt_every"]) == 0 or past:
            save_ckpt(t)
        if past and t < int(cfg["rounds"]):
            return {"status": "paused", "round": t}

    ev = [r for r in log if "val" in r]
    best = max(ev, key=lambda r: r["val"])
    hv = [r for r in ev if r.get("heldout") is not None]
    sel_row = max(hv, key=lambda r: (r["heldout"], -r["round"])) if hv else None
    ups = [r["up_params"] for r in log]
    result = dict(
        status="done", run_id=run_id(cfg), cfg=cfg, metric=metric,
        summary=dict(best_val=best["val"], best_val_round=best["round"], final_val=ev[-1]["val"],
                     val_at_best_heldout=None if sel_row is None else sel_row["val"],
                     best_heldout_round=None if sel_row is None else sel_row["round"],
                     up_params_mean=float(np.mean(ups)), up_params_max=int(max(ups)),
                     up_kb_mean=float(np.mean(ups)) * 4 / 1024, rounds=len(log),
                     train_sec=float(sum(r["train_sec"] for r in log)),
                     steps=int(sum(r["steps"] for r in log))),
        sizes=dict(train=len(train), heldout=len(heldout), val=len(val),
                   clients=[len(p) for p in parts][:50]),
        global_params=int(sum(gstate[n].numel() for n in gnames)),
        log=log, env=dict(env_info(device), precision=amp_desc), wall_sec=time.time() - t0_run,
    )
    _write_json(res_path, result)
    for p in (ckpt_path, os.path.join(rdir, "error.txt")):      # error.txt: left over from a failed earlier try
        if os.path.exists(p):
            os.remove(p)
    return result
