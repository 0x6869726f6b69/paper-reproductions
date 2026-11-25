"""CPU tests for the MetaTT-4D implementation (no download, no GPU).

  python reproduction/tests/test_core.py        (or: python -m pytest reproduction/tests)
"""
import math
import os
import sys

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                        # reproduction/ (metatt_repro)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "method"))  # method/ (metatt)

from metatt import MetaTT4D, MetaTT4DConfig, MetaTTLinear, apply_metatt, count_trainable, metatt4d_num_params  # noqa: E402
from metatt_repro import glue  # noqa: E402
from metatt_repro.train import RunConfig, fit, linear_schedule  # noqa: E402


def roberta(hidden=32, layers=3, heads=4, num_labels=2, vocab=100):
    from transformers import RobertaConfig, RobertaForSequenceClassification
    cfg = RobertaConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=2 * hidden, max_position_embeddings=64, num_labels=num_labels)
    torch.manual_seed(0)
    return RobertaForSequenceClassification(cfg)


def batch(n=4, t=10, vocab=100):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, vocab, (n, t), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_param_count_roberta_base():
    """Table 2: RoBERTa-base (D=768, L=12), Q and V (M=2): r=8 -> 13,184 (13k), r=24 -> 44,928 (45k), r=64 -> 155,648 (156k)."""
    from transformers import RobertaConfig, RobertaForSequenceClassification
    cfg = RobertaConfig(num_labels=2)                     # defaults are the roberta-base sizes
    for r, k in ((8, 13), (24, 45), (64, 156)):
        with torch.device("meta"):
            m = RobertaForSequenceClassification(cfg)
        apply_metatt(m, MetaTT4DConfig(rank=r))
        n = count_trainable(m)
        assert n == metatt4d_num_params(768, 768, 12, 2, r) == 2 * 768 * r + (12 + 2) * r * r
        assert round(n / 1000) == k, (r, n)
    assert metatt4d_num_params(768, 768, 12, 2, 4) == 6368      # the hyper-parameter table's "rank 4" cannot be 13k


def test_init_ze_id_id_id_and_zero_update():
    m = roberta()
    m.eval()
    x = batch()
    before = m(**x).logits
    ad = apply_metatt(m, MetaTT4DConfig(rank=8, alpha=4.0))
    assert torch.count_nonzero(ad.G1) == 0
    for s in list(ad.G2) + list(ad.G3):
        assert torch.equal(s, torch.eye(8))
    assert torch.equal(ad.G4, torch.eye(8, 32))           # nn.init.eye_ on the r x D output core
    assert torch.allclose(m(**x).logits, before)          # Delta W = 0 at initialisation


def test_only_cores_train_and_head_stays_frozen():
    m = roberta()
    ad = apply_metatt(m, MetaTT4DConfig(rank=4))
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names == ["metatt.G1", "metatt.G2", "metatt.G3", "metatt.G4"], names
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith("classifier"))
    assert not any("metatt" in k for k in m.state_dict() if not k.startswith("metatt."))  # cores stored once


def test_layer_and_type_slices():
    m = roberta(layers=3)
    apply_metatt(m, MetaTT4DConfig(rank=4, target_modules=("query", "value")))
    seen = {}
    for name, mod in m.named_modules():
        if isinstance(mod, MetaTTLinear):
            seen[name] = (mod.layer, mod.mtype)
    assert len(seen) == 6
    assert seen["roberta.encoder.layer.2.attention.self.value"] == (2, 1)
    assert seen["roberta.encoder.layer.0.attention.self.query"] == (0, 0)
    assert isinstance(m.roberta.encoder.layer[1].attention.self.key, nn.Linear)   # K is not adapted


def test_forward_equals_dense_delta_and_paper_pseudocode():
    """Our adapter term equals x @ Delta W and the paper's Algorithm 3/4 (ParameterList cores, typos fixed)."""
    torch.manual_seed(2)
    cfg = MetaTT4DConfig(rank=(3, 4, 5), alpha=0.5, init="no-no-no-no")
    ad = MetaTT4D(16, 16, 3, 2, cfg)
    base = nn.Linear(16, 16)
    lin = MetaTTLinear(base, ad, layer=2, mtype=1)
    x = torch.randn(7, 16)
    ref_dense = base(x) + x @ ad.delta_weight(2, 1)
    M1, M2, M3, M4 = ad.G1, nn.ParameterList([nn.Parameter(s) for s in ad.G2])[2], \
        nn.ParameterList([nn.Parameter(s) for s in ad.G3])[1], ad.G4
    ref_paper = base(x) + cfg.alpha * (((x @ M1) @ M2) @ M3) @ M4      # Algorithm 4
    assert torch.allclose(lin(x), ref_dense, atol=1e-5)
    assert torch.allclose(lin(x), ref_paper, atol=1e-5)


def test_first_step_moves_only_g1():
    """With G1 = 0 the first gradient reaches only G1 (the other cores' gradients contain x G1 = 0)."""
    m = roberta()
    ad = apply_metatt(m, MetaTT4DConfig(rank=4, alpha=4.0))
    x = batch()
    out = m(**x, labels=torch.tensor([0, 1, 0, 1]))
    out.loss.backward()
    assert ad.G1.grad.abs().sum() > 0
    for g in (ad.G2.grad, ad.G3.grad, ad.G4.grad):
        assert g.abs().sum() == 0


def test_linear_schedule_matches_trainer():
    f = linear_schedule(10, 100)
    assert f(0) == 0 and f(5) == 0.5 and f(10) == 1.0 and math.isclose(f(55), 0.5) and f(100) == 0.0


def test_metrics():
    import numpy as np
    p, y = np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1])
    m = glue.metrics("mrpc", p, y)
    assert math.isclose(m["accuracy"], 60.0) and math.isclose(m["f1"], 100 * 2 * 2 / (2 * 2 + 1 + 1))
    assert math.isclose(glue.metrics("stsb", np.array([1., 2., 3.]), np.array([3., 5., 9.]))["spearman"], 100.0)


def test_fit_smoke():
    """Two epochs of the real training loop on random tokens: runs, changes the cores, leaves the rest frozen."""
    m = roberta()
    ad = apply_metatt(m, MetaTT4DConfig(rank=4, alpha=4.0))
    g = torch.Generator().manual_seed(3)
    ids = [torch.randint(3, 100, (int(torch.randint(4, 12, (1,), generator=g)),), generator=g).tolist() for _ in range(40)]
    labels = [int(sum(s) % 2) for s in ids]
    data = {"train": {"input_ids": ids[:32], "labels": labels[:32]}, "eval": {"input_ids": ids[32:], "labels": labels[32:]}}
    cfg = RunConfig(task="rte", seed=0, alpha=4.0, lr=1e-2, batch_size=8, warmup_ratio=0.06, rank=4, epochs=2)
    g1 = ad.G1.detach().clone()
    hist, info = fit(m, data, cfg, torch.device("cpu"), pad_id=1, log=lambda s: None)
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    assert info["frozen_unchanged"] and info["total_steps"] == 8 and info["warmup_steps"] == 1
    assert not torch.equal(ad.G1.detach(), g1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} tests passed")
