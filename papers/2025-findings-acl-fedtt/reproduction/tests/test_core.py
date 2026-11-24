"""CPU tests (no downloads): TT layer, FedTT+ schedule and freezing, aggregation, partitions, counts.

  python tests/test_core.py        (or: python -m pytest tests)
"""
import copy
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedtt_repro import data as D                       # noqa: E402
from fedtt_repro import fl                               # noqa: E402
from fedtt import model as M                             # noqa: E402
from fedtt.tt import TTLinear, plus_core_index, split_modes   # noqa: E402

torch.set_num_threads(4)


# ------------------------------------------------------------------ TT layer
def test_split_modes_table10():
    assert split_modes((8, 8, 12, 8, 8), 768, 64) == ((8, 8, 12), (8, 8))
    assert split_modes((8, 8, 12, 8, 8), 64, 768) == ((8, 8), (12, 8, 8))
    assert split_modes((16, 16, 16, 4, 4, 4), 4096, 64) == ((16, 16, 16), (4, 4, 4))
    assert split_modes((4, 4, 4, 16, 16, 16), 64, 4096) == ((4, 4, 4), (16, 16, 16))
    assert split_modes((12, 8, 8, 8, 8, 12), 768, 768) == ((12, 8, 8), (8, 8, 12))


def test_tt_contract_equals_dense():
    torch.manual_seed(0)
    for shape, (I, O) in [((8, 8, 12, 8, 8), (768, 64)), ((8, 8, 12, 8, 8), (64, 768)),
                          ((12, 8, 8, 8, 8, 12), (768, 768)), ((8, 8, 16, 8, 8), (1024, 64))]:
        L = TTLinear(I, O, shape, 5, bias=True, w_std=0.05)
        with torch.no_grad():
            L.bias.normal_()
        x = torch.randn(3, 7, I)
        y_c = L(x)
        y_ref = x @ L.weight() + L.bias
        assert torch.allclose(y_c, y_ref, atol=1e-5, rtol=1e-4), shape
        L2 = copy.deepcopy(L)
        L2.mode = "dense"
        assert torch.allclose(L2(x), y_c, atol=1e-5, rtol=1e-4)
        # identical gradients for the cores in both modes
        g = torch.randn_like(y_c)
        (y_c * g).sum().backward()
        (L2(x) * g).sum().backward()
        for a, b in zip(L.cores, L2.cores):
            assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-3)


def test_tt_param_count_and_init():
    L = TTLinear(768, 64, (8, 8, 12, 8, 8), 5, bias=False, w_std=0.01)
    assert L.n_core_params() == 40 + 200 + 300 + 200 + 40 == 780
    # (the Sec. 3.2 example "768 x 768 with core dimensions [8,8,8,8,8,8]" cannot be built: 8^6 = 512^2)
    torch.manual_seed(1)
    stds = [TTLinear(768, 64, (8, 8, 12, 8, 8), 5, False, 0.01).weight().std().item() for _ in range(20)]
    assert 0.007 < np.median(stds) < 0.013, np.median(stds)


def test_plus_schedule():
    assert [plus_core_index(t, 5, "text") for t in range(1, 7)] == [2, 3, 4, 2, 3, 4]
    assert [plus_core_index(t, 6, "text") for t in range(1, 6)] == [2, 3, 4, 5, 2]
    assert [plus_core_index(t, 5, "literal") for t in range(1, 7)] == [2, 3, 1, 2, 3, 1]


# ------------------------------------------------------------------ tiny federated model
def tiny_model(method, rank=5, layers="all", model_type="roberta", n_cls=2, **kw):
    cfg = dict(model="tiny", method=method, rank=rank, adapter_layers=layers, adapter_init_std=0.05, **kw)
    hc = M.tiny_config(model_type, vocab_size=60, n_layers=2, num_labels=n_cls)
    return M.build_model(cfg, n_cls, hf_config=hc, pretrained=False), cfg


def tiny_data(n=64, n_cls=2, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(5, 20, size=n)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    ids = rng.integers(3, 60, size=off[-1]).astype(np.int32)
    y = rng.integers(0, n_cls, size=n)
    ids[off[:-1]] = 3 + y                     # make the label learnable from the first token
    return D.TokData(ids, off, y)


def _round_cfg(**kw):
    c = fl.full_cfg(dict(model="tiny", lr=5e-3, bs=8, local_epochs=1.0, seed=0))
    c.update(kw)
    return c


def test_fedtt_round_updates_everything_and_averages():
    torch.manual_seed(0)
    fm, mcfg = tiny_model("fedtt")
    data = tiny_data(64)
    parts = D.partition(data.labels, 2, "iid", np.random.default_rng(0))
    clients = [data.subset(p) for p in parts]
    params = dict(fm.hf.named_parameters())
    g0 = {n: params[n].detach().clone() for n in fm.global_names}
    cfg = _round_cfg(**mcfg)
    dev = torch.device("cpu")
    # each client alone from the same global state ...
    outs = []
    for c in (0, 1):
        g = {n: v.clone() for n, v in g0.items()}
        fl.run_round(fm, g, clients, np.array([c]), cfg, 1, dev, 0, None)
        outs.append(g)
    # ... and both together: the server must hold their mean (Alg. 1, line 8)
    g = {n: v.clone() for n, v in g0.items()}
    st = fl.run_round(fm, g, clients, np.array([0, 1]), cfg, 1, dev, 0, None)
    for n in g:
        assert torch.allclose(g[n], (outs[0][n] + outs[1][n]) / 2, atol=1e-6), n
        assert not torch.equal(g[n], g0[n]), f"{n} did not change"
    assert st["up_params"] == sum(v.numel() for v in g0.values())


def test_fedttplus_only_updates_scheduled_cores():
    torch.manual_seed(0)
    fm, mcfg = tiny_model("fedttplus")
    data = tiny_data(64)
    clients = [data.subset(p) for p in D.partition(data.labels, 2, "iid", np.random.default_rng(0))]
    params = dict(fm.hf.named_parameters())
    g = {n: params[n].detach().clone() for n in fm.global_names}
    cfg = _round_cfg(**mcfg)
    for t in (1, 2, 3, 4):
        before = {n: v.clone() for n, v in g.items()}
        st = fl.run_round(fm, g, clients, np.array([0, 1]), cfg, t, torch.device("cpu"), 0, None)
        r = plus_core_index(t, 5)
        changed = {n for n in g if not torch.equal(g[n], before[n])}
        for L_i, L in enumerate(fm.tt_adapter_layers):
            for j, c in enumerate(L.cores):
                name = next(n for n, p in fm.hf.named_parameters() if p is c)
                assert (name in changed) == (j in (0, r - 1, L.J - 1)), (t, name)
            bname = next(n for n, p in fm.hf.named_parameters() if p is L.bias)
            assert bname not in changed                    # adapter biases frozen in FedTT+ (default)
        assert all(n in changed for n in fm.head_names if n in g)      # head always trained
        assert set(st["uploaded"]) == changed | (set(st["uploaded"]) - changed)
        expected = sum(L.cores[0].numel() + L.cores[r - 1].numel() + L.cores[L.J - 1].numel()
                       for L in fm.tt_adapter_layers) + sum(g[n].numel() for n in fm.head_names if n in g)
        assert st["up_params"] == expected, (st["up_params"], expected)


def test_micro_batches_match_full_batch():
    torch.manual_seed(0)
    fm, mcfg = tiny_model("fedtt")
    data = tiny_data(48)
    clients = [data.subset(p) for p in D.partition(data.labels, 2, "iid", np.random.default_rng(0))]
    params = dict(fm.hf.named_parameters())
    g0 = {n: params[n].detach().clone() for n in fm.global_names}
    outs = []
    for micro in (None, 3):
        g = {n: v.clone() for n, v in g0.items()}
        fl.run_round(fm, g, clients, np.arange(2), _round_cfg(micro_bs=micro, **mcfg), 1, torch.device("cpu"), 0, None)
        outs.append(g)
    for n in g0:
        assert torch.allclose(outs[0][n], outs[1][n], atol=1e-5, rtol=1e-4), n


def test_probe_amp_falls_back_for_deberta():
    """DeBERTa fails under bf16 autocast (seen on Colab A100); the probe must fall back to fp32, RoBERTa keeps bf16."""
    data = tiny_data(8)
    for mt, expect in (("deberta", None), ("roberta", torch.bfloat16)):
        fm, _ = tiny_model("fedtt", model_type=mt)
        dt, desc = fl.probe_amp(fm, data, torch.device("cpu"), 0, torch.bfloat16)
        assert dt == expect, (mt, desc)


def test_fedtt_trains_a_toy_task():
    torch.manual_seed(0)
    fm, mcfg = tiny_model("fedtt")
    data = tiny_data(128)
    clients = [data.subset(p) for p in D.partition(data.labels, 4, "iid", np.random.default_rng(0))]
    params = dict(fm.hf.named_parameters())
    g = {n: params[n].detach().clone() for n in fm.global_names}
    cfg = _round_cfg(**mcfg)
    acc0 = fl.evaluate(fm, data, 32, torch.device("cpu"), 0, None)["acc"]
    for t in range(1, 9):
        fl.run_round(fm, g, clients, np.arange(4), cfg, t, torch.device("cpu"), 0, None)
    acc1 = fl.evaluate(fm, data, 32, torch.device("cpu"), 0, None)["acc"]
    assert acc1 > max(acc0, 80.0), (acc0, acc1)


def test_counts_tiny():
    fm, _ = tiny_model("fedtt")            # 2 blocks, hidden 768, binary
    per_adapter = 2 * 780 + 64 + 768
    head = (60 + 4 * 200 + 60) + 768 + (768 * 2 + 2)     # TT 768x768 [12,8,8,8,8,12] r5 + bias + out_proj
    assert M.count_round_params(fm, 1) == 2 * 2 * per_adapter + head
    fm, _ = tiny_model("fedttplus")
    expect = {1: 40 + 200 + 40, 2: 40 + 300 + 40, 3: 40 + 200 + 40}   # down/up have the same core sizes
    for t in (1, 2, 3):
        assert M.count_round_params(fm, t) == 2 * 2 * 2 * expect[t] + head
    fm, _ = tiny_model("fedtt", layers="last1")
    assert M.count_round_params(fm, 1) == 1 * 2 * per_adapter + head


def test_deberta_head_and_sites():
    fm, _ = tiny_model("fedtt", model_type="deberta", n_cls=3)
    assert isinstance(fm.hf.pooler.dense, TTLinear)
    assert sum(isinstance(m, M.TensorizedAdapter) for m in fm.hf.modules()) == 4
    x = torch.randint(3, 60, (2, 9))
    assert fm(x, torch.ones_like(x)).shape == (2, 3)


# ------------------------------------------------------------------ partitions
def test_partitions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=3000)
    p = D.partition(y, 5, "iid", rng)
    assert sorted(np.concatenate(p).tolist()) == list(range(3000)) and max(map(len, p)) - min(map(len, p)) <= 1
    p = D.partition(y, 100, "dir", np.random.default_rng(1), alpha=1.0)
    assert sorted(np.concatenate(p).tolist()) == list(range(3000)) and min(map(len, p)) >= 2
    y2 = rng.integers(0, 2, size=103695)          # QNLI-sized, 1000 clients (Table 2 LSCD)
    p = D.partition(y2, 1000, "dir", np.random.default_rng(1), alpha=1.0)
    assert len(p) == 1000 and min(map(len, p)) >= 2
    yb = np.repeat([0, 1], 3000)
    p = D.partition(yb, 3, "mild", np.random.default_rng(2))
    fr = [np.mean(yb[q] == 0) for q in p]
    assert np.allclose(fr, [0.15, 0.85, 0.5], atol=0.01), fr
    y3 = np.repeat([0, 1, 2], 1000)
    p = D.partition(y3, 3, "severe", np.random.default_rng(3))
    assert np.allclose([np.mean(y3[p[0]] == 0), np.mean(y3[p[1]] == 1)], [0.9, 0.9], atol=0.01)


def test_local_batches():
    c = _round_cfg(bs=32, local_epochs=1.0)
    b = fl.local_batches(100, c, 1, 0)
    assert len(b) == 4 and sorted(np.concatenate(b).tolist()) == list(range(100))
    c = _round_cfg(bs=32, local_epochs=None, local_steps=20)
    b = fl.local_batches(50, c, 1, 0)
    assert len(b) == 20 and all(len(x) == 32 for x in b)


def test_run_id_distinct():
    a = dict(task="sst2", method="fedtt", rank=5, lr=1e-3, bs=32)
    assert fl.run_id(a) != fl.run_id(dict(a, plus_schedule="literal"))
    assert fl.run_id(a) == fl.run_id(dict(a, eval_bs=64))


if __name__ == "__main__":
    fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            try:
                f()
                print("ok  ", name)
            except Exception as e:
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, e)
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
