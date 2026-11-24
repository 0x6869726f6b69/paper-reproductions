"""Runs a suite: skips finished runs, resumes interrupted ones from their checkpoint, optionally in parallel
worker processes, and stops starting new work after --hours.

  python -m fedtt_repro.pipeline --suite t7 --out OUT [--parallel 4] [--hours 20]
         [--tasks sst2,qnli] [--methods fedtt] [--lrs 1e-3] [--bss 32] [--seeds 0,1,2] [--rounds 20]
         [--baselines] [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

from . import suites
from .fl import full_cfg, run, run_dir, run_id


def _floats(s):
    return None if not s else [float(x) for x in s.split(",")]


def _ints(s):
    return None if not s else [int(x) for x in s.split(",")]


def _strs(s):
    return None if not s else s.split(",")


def select(args):
    cfgs = suites.SUITES[args.suite](baselines=args.baselines)
    if args.tasks:
        cfgs = [c for c in cfgs if c["task"] in _strs(args.tasks)]
    if args.methods:
        cfgs = [c for c in cfgs if c["method"] in _strs(args.methods)
                or f"{c['method']}-r{c['rank']}" in _strs(args.methods)]
    if args.lrs:
        cfgs = [c for c in cfgs if any(abs(c["lr"] - x) < 1e-12 for x in _floats(args.lrs))]
    if args.bss:
        cfgs = [c for c in cfgs if c["bs"] in _ints(args.bss)]
    if args.levels:
        cfgs = [c for c in cfgs if c.get("tag", "") in _strs(args.levels) or c.get("partition") in _strs(args.levels)]
    seeds = _ints(args.seeds) or [0]
    out = []
    for c in cfgs:
        for s in seeds:
            d = dict(c, seed=s)
            if args.rounds:
                d["rounds"] = int(args.rounds)
            if args.threads:
                d["threads"] = int(args.threads)
            if args.micro_bs:
                d["micro_bs"] = int(args.micro_bs)
            out.append(d)
    return out


def is_done(out_root, cfg):
    return os.path.exists(os.path.join(run_dir(out_root, cfg), "result.json"))


def progress(out_root, cfgs):
    """One line per run in progress: last finished round, its validation score and seconds per round."""
    out = []
    for c in cfgs:
        d = run_dir(out_root, c)
        lp = os.path.join(d, "log.jsonl")
        if os.path.exists(os.path.join(d, "result.json")) or not os.path.exists(lp):
            continue
        try:
            with open(lp, encoding="utf-8") as f:
                rows = [json.loads(x) for x in f if x.strip()]
        except (OSError, ValueError):
            continue
        if not rows:
            continue
        r = rows[-1]
        ev = [x["val"] for x in rows if "val" in x]
        sec = sum(x["train_sec"] + x.get("eval_sec", 0) for x in rows) / len(rows)
        out.append(f"\n    {run_id(c)}: round {r['round']}/{full_cfg(c)['rounds']}, "
                   f"val {ev[-1] if ev else float('nan'):.2f} (best {max(ev) if ev else float('nan'):.2f}), "
                   f"{sec:.0f} s/round")
    return "".join(out)


def worker_main(queue_path, out_root, deadline):
    with open(queue_path, encoding="utf-8") as f:
        cfgs = json.load(f)
    for cfg in cfgs:
        if time.time() > deadline:
            print("deadline reached; stopping", flush=True)
            return
        if is_done(out_root, cfg):
            continue
        try:
            r = run(cfg, out_root, deadline=deadline)
            if r.get("status") == "paused":
                print(f"paused {run_id(cfg)} at round {r['round']}", flush=True)
                return
            s = r["summary"]
            print(f"DONE {run_id(cfg)}: best val {s['best_val']:.2f} (round {s['best_val_round']}), "
                  f"val@best-heldout {s['val_at_best_heldout']}, {s['train_sec'] / 60:.1f} min", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"ERROR {run_id(cfg)}\n{err}", flush=True)
            d = run_dir(out_root, cfg)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "error.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + err + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke", choices=sorted(suites.SUITES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--hours", type=float, default=1e6)
    ap.add_argument("--tasks"), ap.add_argument("--methods"), ap.add_argument("--lrs"), ap.add_argument("--bss")
    ap.add_argument("--levels", help="t3/t13: iid,mild,severe")
    ap.add_argument("--seeds"), ap.add_argument("--rounds", type=int), ap.add_argument("--threads", type=int)
    ap.add_argument("--micro_bs", type=int, help="split each minibatch into chunks (gradient accumulation; GPU memory)")
    ap.add_argument("--baselines", action="store_true", help="also run the LoRA-family references")
    ap.add_argument("--dry", action="store_true", help="list the runs and the estimated work, run nothing")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker_main(args.worker, args.out, args.deadline)

    cfgs = select(args)
    todo = [c for c in cfgs if not is_done(args.out, c)]
    todo.sort(key=lambda c: -suites.est_cost(c))
    tot = sum(suites.est_samples(c) for c in todo)
    print(f"suite {args.suite}: {len(cfgs)} runs, {len(cfgs) - len(todo)} done, {len(todo)} to go "
          f"({tot / 1e6:.1f} M training samples)", flush=True)
    if args.dry:
        for c in todo:
            print(f"  {run_id(c)}  ~{suites.est_samples(c) / 1e6:.2f} M samples")
        return
    if not todo:
        return
    deadline = time.time() + 3600 * args.hours
    os.makedirs(os.path.join(args.out, "queue"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    if args.parallel <= 1:
        qp = os.path.join(args.out, "queue", f"{args.suite}_w0.json")
        with open(qp, "w", encoding="utf-8") as f:
            json.dump([full_cfg(c) for c in todo], f)
        return worker_main(qp, args.out, deadline)
    # greedy balance by estimated cost; each worker runs its queue in order
    shards, load = [[] for _ in range(args.parallel)], [0.0] * args.parallel
    for c in todo:
        j = load.index(min(load))
        shards[j].append(full_cfg(c))
        load[j] += suites.est_cost(c)
    procs, logs = [], []
    for j, sh in enumerate(shards):
        if not sh:
            continue
        qp = os.path.join(args.out, "queue", f"{args.suite}_w{j}.json")
        with open(qp, "w", encoding="utf-8") as f:
            json.dump(sh, f)
        logs.append(os.path.join(args.out, "logs", f"{args.suite}_w{j}.log"))
        lf = open(logs[-1], "w", encoding="utf-8")
        cmd = [sys.executable, "-m", "fedtt_repro.pipeline", "--suite", args.suite, "--out", args.out,
               "--worker", qp, "--deadline", str(deadline)]
        procs.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf))
        print(f"worker {j}: {len(sh)} runs -> logs/{args.suite}_w{j}.log", flush=True)
    while any(p.poll() is None for p, _ in procs):
        time.sleep(60)
        done = sum(is_done(args.out, c) for c in cfgs)
        print(time.strftime("%H:%M:%S"), f"{done}/{len(cfgs)} runs done" + progress(args.out, todo), flush=True)
    for _, lf in procs:
        lf.close()
    done = sum(is_done(args.out, c) for c in cfgs)
    print(f"finished: {done}/{len(cfgs)} runs done", flush=True)
    show_errors(logs)


def show_errors(log_paths, n_lines=60):
    """Print the end of every worker log that contains an error (worker output goes to files, not here)."""
    for lp in log_paths:
        try:
            with open(lp, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if "ERROR" in text or "Traceback" in text:
            start = max(text.rfind("ERROR"), text.rfind("Traceback"))
            tail = text[max(0, start - 200):].splitlines()[-n_lines:]
            print(f"\n----- errors in {lp} -----\n" + "\n".join(tail), flush=True)


if __name__ == "__main__":
    main()
