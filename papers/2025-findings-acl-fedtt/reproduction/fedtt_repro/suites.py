"""Run grids that mirror the experiments of the paper (one suite per table).

Each suite returns a list of run configs (dicts understood by fl.run). The pipeline can filter them
(--tasks/--methods/--lrs/--bss/--seeds) and override the number of rounds (--rounds, which changes the run id).
Anything the paper leaves open is set here and listed in 説明資料.md (§3 "論文に書かれていない設定").
"""
DEB = "microsoft/deberta-base"
RB = "FacebookAI/roberta-base"
RL = "FacebookAI/roberta-large"
GLUE5 = ["mrpc", "sst2", "qnli", "qqp", "mnli"]
GLUE4 = ["sst2", "qnli", "qqp", "mnli"]


def _deb_val(task):
    # Appendix A.1: "For the DeBERTa-Base models on the QQP task, we randomly select 1000 samples from the
    # validation set and report the highest accuracy achieved."
    return dict(val_subsample=1000) if task == "qqp" else {}


def t1(baselines=False):
    """Table 1 (Appendix A.3): DeBERTa-base, cross-silo, IID, 5 clients, 1 local epoch, 100 rounds,
    lr in {5e-3, 1e-3, 5e-4, 1e-4} x batch size in {16, 32}; FedTT and FedTT+ with tensor rank 5."""
    methods = [("fedtt", 5), ("fedttplus", 5)] + ([("lora", 8), ("lora", 4), ("rolora", 4)] if baselines else [])
    out = []
    for task in GLUE5:
        for method, r in methods:
            for lr in (5e-3, 1e-3, 5e-4, 1e-4):
                for bs in (16, 32):
                    out.append(dict(suite="t1", task=task, model=DEB, method=method, rank=r, n_clients=5,
                                    partition="iid", rounds=100, local_epochs=1.0, lr=lr, bs=bs, **_deb_val(task)))
    return out


def t7(baselines=False):
    """Table 7 (Sec. 5.7): DeBERTa-base, 5 clients, one local update (= one local epoch, as in Table 1),
    lr 1e-3, bs 32, best over 20 rounds, tensor rank in {2, 5, 10}. FedTT+ (rank 5) is added in the same
    setting as an extra (not in Table 7)."""
    methods = [("fedtt", 2), ("fedtt", 5), ("fedtt", 10), ("fedttplus", 5)]
    return [dict(suite="t7", task=task, model=DEB, method=m, rank=r, n_clients=5, partition="iid", rounds=20,
                 local_epochs=1.0, lr=1e-3, bs=32, **_deb_val(task))
            for task in GLUE4 for m, r in methods]


def t2(baselines=False):
    """Table 2 (Appendix A.4, FedPETuning setting): RoBERTa-base; MRPC and SST-2 cross-silo with 10 clients,
    QNLI/QQP/MNLI large-scale cross-device with 1000 clients and 10 per round; Dir(1.0) label skew
    (FedPETuning default); 1 local epoch; 100 rounds (cross-silo) / 1000 rounds (LSCD);
    lr in {1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5} x bs in {16, 32}; FedTT rank 5."""
    methods = [("fedtt", 5)] + ([("lora", 8)] if baselines else [])
    out = []
    for task in GLUE5:
        silo = task in ("mrpc", "sst2")
        base = dict(suite="t2", task=task, model=RB, n_clients=10 if silo else 1000,
                    clients_per_round=None if silo else 10, partition="dir", alpha=1.0,
                    rounds=100 if silo else 1000, local_epochs=1.0, eval_every=1 if silo else 10)
        for method, r in methods:
            for lr in (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5):
                for bs in (16, 32):
                    out.append(dict(base, method=method, rank=r, lr=lr, bs=bs))
    return out


def t3(baselines=False):
    """Table 3 (Sec. 5.3, Appendix A.5, RoLoRA setting): RoBERTa-large, 3 / 20 / 50 clients (iid / mild /
    severe heterogeneity, data split uniformly), 20 local updates (steps), lr 1e-3, bs 64 (SST-2) / 32;
    FedTT in the last 6 (SST-2) / 9 (others) layers, FedTT+ in all layers; best over 100 rounds."""
    out = []
    for level, N in (("iid", 3), ("mild", 20), ("severe", 50)):
        for task in GLUE4:
            last = "last6" if task == "sst2" else "last9"
            base = dict(suite="t3", task=task, model=RL, n_clients=N, partition="iid", rounds=100, local_epochs=None,
                        local_steps=20, lr=1e-3, bs=64 if task == "sst2" else 32, tag=level)
            out.append(dict(base, method="fedtt", rank=5, adapter_layers=last))
            out.append(dict(base, method="fedttplus", rank=5, adapter_layers="all"))
            if baselines:
                for m in ("lora", "ffa", "rolora"):
                    out.append(dict(base, method=m, rank=2, adapter_layers=last))
    return out


def t13(baselines=False):
    """Table 13 (Appendix B, FFA-LoRA setting): RoBERTa-large, 3 clients with label-ratio splits
    (iid / mild / severe), 10 local updates, batch size 200, 1000 rounds; FedTT+ in all layers, lr 1e-3."""
    out = []
    for part in ("iid", "mild", "severe"):
        for task in GLUE4:
            base = dict(suite="t13", task=task, model=RL, n_clients=3, partition=part, rounds=1000, local_epochs=None,
                        local_steps=10, bs=200, lr=1e-3, eval_every=10)
            out.append(dict(base, method="fedttplus", rank=5))
            if baselines:
                out.append(dict(base, method="lora", rank=8, lr=1e-3))
                out.append(dict(base, method="ffa", rank=8, lr=1e-3))
    return out


def smoke(baselines=False):
    """Tiny runs on real data/models that exercise every code path (minutes on a GPU)."""
    tiny = dict(suite="smoke", rounds=2, train_subsample=320, val_subsample=96, heldout_subsample=48, ckpt_every=1)
    out = [dict(tiny, task="mrpc", model=DEB, method="fedtt", rank=5, n_clients=5, local_epochs=1.0, bs=32, lr=1e-3),
           dict(tiny, task="mrpc", model=DEB, method="fedttplus", rank=5, n_clients=5, local_epochs=1.0, bs=32, lr=1e-3),
           dict(tiny, task="mnli", model=RB, method="fedtt", rank=5, n_clients=20, clients_per_round=4,
                partition="dir", alpha=1.0, local_epochs=1.0, bs=16, lr=1e-3),
           dict(tiny, task="sst2", model=RL, method="fedttplus", rank=5, n_clients=3, partition="mild",
                local_epochs=None, local_steps=3, bs=16, lr=1e-3),
           dict(tiny, task="qnli", model=RL, method="fedtt", rank=5, adapter_layers="last9", n_clients=3,
                local_epochs=None, local_steps=3, bs=16, lr=1e-3)]
    if baselines:
        out.append(dict(tiny, task="mrpc", model=DEB, method="lora", rank=8, n_clients=5, local_epochs=1.0, bs=32, lr=1e-3))
    return out


SUITES = dict(smoke=smoke, t1=t1, t7=t7, t2=t2, t3=t3, t13=t13)

# Rough training sizes (GLUE train after the FedPETuning split) for cost estimates.
TRAIN_SIZE = dict(mrpc=3301, sst2=66675, qnli=103695, qqp=360210, mnli=388774, rte=2241)
MODEL_COST = {DEB: 1.0, RB: 0.8, RL: 2.8}       # relative time per sample (rough)


def est_samples(cfg):
    """Training samples processed by the whole run (all clients, all rounds)."""
    n = cfg.get("train_subsample") or TRAIN_SIZE[cfg["task"]]
    N = cfg.get("n_clients", 5)
    m = cfg.get("clients_per_round") or N
    if cfg.get("local_steps"):
        per_round = m * cfg["local_steps"] * cfg["bs"]
    else:
        per_round = m * (n / N) * float(cfg.get("local_epochs") or 1.0)
    return per_round * cfg["rounds"]


def est_cost(cfg):
    return est_samples(cfg) * MODEL_COST.get(cfg.get("model"), 1.0)
