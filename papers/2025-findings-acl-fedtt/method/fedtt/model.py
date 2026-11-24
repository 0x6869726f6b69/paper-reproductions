"""Tensorized adapters and the tensorized classifier of FedTT (Sec. 4.1, Fig. 1, Table 10), injected into a
Hugging Face encoder; optional LoRA-family references (PEFT) for side-by-side checks.

Placement (Fig. 1b): one adapter after the multi-head attention and one after the MLP of every selected
block, i.e. on the outputs of `attention.output.dense` and `output.dense` (before dropout, residual and
LayerNorm, as in Houlsby et al. 2019). Adapter: h -> h + up(act(down(h))), down: H -> 64, up: 64 -> H,
both TT layers (Table 10 shapes, rank 5 by default). Classification (Fig. 1c): the H x H dense layer of the
classification head is replaced by a TT layer; the final H x C projection stays dense and trainable.
The backbone (including LayerNorms) is frozen.
"""
import math
import re

import torch
import torch.nn as nn

from .tt import TTLinear, plus_core_index

# (in, out) -> TT shape. 768/4096 rows are Table 10 of the paper. The paper gives no shapes for hidden size
# 1024 (RoBERTa-large); the 1024 rows are our analogue of the 768 rows (8*8*16 instead of 8*8*12) and are
# checked against the parameter counts of Table 12 by `python -m fedtt_repro.count`.
SHAPES = {
    (768, 64): (8, 8, 12, 8, 8),
    (64, 768): (8, 8, 12, 8, 8),
    (4096, 64): (16, 16, 16, 4, 4, 4),
    (64, 4096): (4, 4, 4, 16, 16, 16),
    (768, 768): (12, 8, 8, 8, 8, 12),
    (1024, 64): (8, 8, 16, 8, 8),
    (64, 1024): (8, 8, 16, 8, 8),
    (1024, 1024): (16, 8, 8, 8, 8, 16),
}

TT_METHODS = ("fedtt", "fedttplus")
LORA_METHODS = ("lora", "ffa", "rolora")


class TensorizedAdapter(nn.Module):
    def __init__(self, hidden, bottleneck, rank, act="relu", w_std=0.01, bias=True, mode="contract",
                 shapes=None):
        super().__init__()
        shapes = shapes or SHAPES
        self.down = TTLinear(hidden, bottleneck, shapes[(hidden, bottleneck)], rank, bias, w_std, mode)
        self.up = TTLinear(bottleneck, hidden, shapes[(bottleneck, hidden)], rank, bias, w_std, mode)
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}[act]

    def forward(self, h):
        return h + self.up(self.act(self.down(h)))


class WithAdapter(nn.Module):
    """x -> adapter(base(x)); `base` is the frozen pretrained projection."""

    def __init__(self, base, adapter):
        super().__init__()
        self.base, self.adapter = base, adapter

    def forward(self, x):
        return self.adapter(self.base(x))


_SITE = re.compile(r"(^|\.)encoder\.layer\.(\d+)\.(attention\.output|output)\.dense$")


def _set_module(root, name, new):
    *path, leaf = name.split(".")
    parent = root
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def select_blocks(n_layers, spec):
    """'all' | 'lastN' | comma list of block indices."""
    if spec in (None, "all"):
        return list(range(n_layers))
    if isinstance(spec, str) and spec.startswith("last"):
        n = int(spec[4:])
        return list(range(n_layers - n, n_layers))
    if isinstance(spec, str):
        return [int(s) for s in spec.split(",")]
    return list(spec)


def adapter_sites(model, blocks):
    names = []
    for n, m in model.named_modules():
        hit = _SITE.search(n)
        if hit and isinstance(m, nn.Linear) and int(hit.group(2)) in blocks:
            names.append(n)
    if len(names) != 2 * len(blocks):
        raise RuntimeError(f"expected {2 * len(blocks)} adapter sites, found {names[:4]}...")
    return names


def head_modules(model):
    """(name of the H x H dense layer of the classification head, name of the H x C output layer)."""
    if hasattr(model, "classifier") and hasattr(model.classifier, "dense"):      # RoBERTa
        return "classifier.dense", "classifier.out_proj"
    if hasattr(model, "pooler") and hasattr(model.pooler, "dense"):              # DeBERTa (v1/v2)
        return "pooler.dense", "classifier"
    raise RuntimeError(f"unknown classification head for {type(model).__name__}")


def _lora_targets(model):
    t = model.config.model_type
    if t == "roberta" or t == "bert":
        return ["query", "value"]
    if t == "deberta":
        return ["in_proj"]                    # fused q/k/v projection of DeBERTa v1
    if t == "deberta-v2":
        return ["query_proj", "value_proj"]
    raise RuntimeError(f"no LoRA targets for model type {t}")


class FedModel(nn.Module):
    """Wraps the HF model and implements the per-round trainable sets of each method."""

    def __init__(self, hf, cfg, tt_adapter_layers, head_names, global_names):
        super().__init__()
        self.hf, self.cfg = hf, cfg
        self.tt_adapter_layers = tt_adapter_layers     # TT layers subject to the FedTT+ freezing schedule
        self.head_names = head_names
        self.global_names = global_names               # parameters kept by the server (trainable in some round)

    def forward(self, input_ids, attention_mask):
        return self.hf(input_ids=input_ids, attention_mask=attention_mask).logits

    def set_round(self, t):
        """Set requires_grad for round t (1-based); returns a short description of the round's schedule."""
        m = self.cfg["method"]
        if m == "fedttplus":
            sched = self.cfg.get("plus_schedule", "text")
            rs = set()
            for L in self.tt_adapter_layers:
                r = plus_core_index(t, L.J, sched)                     # 1-based core index (Alg. 2, line 3)
                L.set_trainable_cores({0, r - 1, L.J - 1}, self.cfg.get("plus_train_bias", False))
                rs.add(r)
            return f"r={'/'.join(map(str, sorted(rs)))}"
        if m == "rolora":                                              # B in odd rounds, A in even rounds
            for n, p in self.hf.named_parameters():
                if "lora_B" in n:
                    p.requires_grad_(t % 2 == 1)
                elif "lora_A" in n:
                    p.requires_grad_(t % 2 == 0)
            return "B" if t % 2 == 1 else "A"
        return ""

    def trainable(self):
        return [(n, p) for n, p in self.hf.named_parameters() if p.requires_grad]


def tiny_config(model_type, vocab_size, n_layers=2, hidden=768, num_labels=2):
    """Small random encoders with a supported hidden size (tests and offline counting only)."""
    from transformers import DebertaConfig, RobertaConfig
    kw = dict(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=n_layers, num_attention_heads=4,
              intermediate_size=256, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
              num_labels=num_labels)
    if model_type == "deberta":
        return DebertaConfig(max_position_embeddings=136, **kw)
    return RobertaConfig(max_position_embeddings=136, **kw)


def build_model(cfg, num_labels, hf_config=None, pretrained=True):
    """cfg keys: model, method, rank, adapter_layers, bottleneck, act, adapter_init_std, tt_mode, tt_classifier,
    lora_alpha. `hf_config` (with pretrained=False) builds an untrained model from a config (counting/tests)."""
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(int(cfg.get("seed", 0)))
    if hf_config is not None and not pretrained:
        hf = AutoModelForSequenceClassification.from_config(hf_config)
    else:
        hf = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=num_labels)
    for p in hf.parameters():
        p.requires_grad_(False)
    method = cfg["method"]
    H = hf.config.hidden_size
    n_layers = hf.config.num_hidden_layers
    blocks = select_blocks(n_layers, cfg.get("adapter_layers", "all"))
    head_dense, head_out = head_modules(hf)
    tt_layers = []
    rank = int(cfg.get("rank", 5))
    if method in TT_METHODS:
        for name in adapter_sites(hf, blocks):
            ad = TensorizedAdapter(H, int(cfg.get("bottleneck", 64)), rank, cfg.get("act", "relu"),
                                   float(cfg.get("adapter_init_std", 0.01)), True, cfg.get("tt_mode", "contract"))
            _set_module(hf, name, WithAdapter(hf.get_submodule(name), ad))
            tt_layers += [ad.down, ad.up]
            for p in ad.parameters():
                p.requires_grad_(True)
        if cfg.get("tt_classifier", True):
            std = getattr(hf.config, "initializer_range", 0.02)
            _set_module(hf, head_dense, TTLinear(H, H, SHAPES[(H, H)], rank, True, std, cfg.get("tt_mode", "contract")))
        for name in (head_dense, head_out):
            for p in hf.get_submodule(name).parameters():
                p.requires_grad_(True)
    elif method in LORA_METHODS:
        from peft import LoraConfig, get_peft_model
        kw = dict(r=rank, lora_alpha=float(cfg.get("lora_alpha", 2 * rank)), lora_dropout=0.0, bias="none",
                  target_modules=_lora_targets(hf), modules_to_save=[head_dense.split(".")[0], head_out.split(".")[0]])
        if cfg.get("adapter_layers", "all") != "all":
            kw.update(layers_to_transform=blocks, layers_pattern="layer")
        hf = get_peft_model(hf, LoraConfig(**kw))
        if method == "ffa":                    # FFA-LoRA: A frozen at its (shared) random init
            for n, p in hf.named_parameters():
                if "lora_A" in n:
                    p.requires_grad_(False)
    elif method == "head":
        for name in (head_dense, head_out):
            for p in hf.get_submodule(name).parameters():
                p.requires_grad_(True)
    else:
        raise ValueError(f"unknown method {method}")
    # Before the first set_round() everything that is trainable in some round still has requires_grad=True
    # (FedTT+ cores and biases, RoLoRA A and B); FFA-LoRA's A is frozen for good and stays out.
    global_names = [n for n, p in hf.named_parameters() if p.requires_grad]
    head_names = [n for n, _ in hf.named_parameters() if n.startswith(head_dense) or n.startswith(head_out)
                  or ".modules_to_save." in n]
    return FedModel(hf, cfg, tt_layers, head_names, global_names)


def count_round_params(fm, t):
    """Parameters a client uploads in round t (= trainable parameters of that round)."""
    fm.set_round(t)
    return sum(p.numel() for _, p in fm.trainable())


def param_breakdown(fm, t=1):
    fm.set_round(t)
    out = {"adapter_cores": 0, "adapter_bias": 0, "head": 0, "lora": 0, "other": 0}
    for n, p in fm.trainable():
        if ".adapter." in n and ".cores." in n:
            out["adapter_cores"] += p.numel()
        elif ".adapter." in n and n.endswith("bias"):
            out["adapter_bias"] += p.numel()
        elif n in fm.head_names or ".modules_to_save." in n:
            out["head"] += p.numel()
        elif "lora_" in n:
            out["lora"] += p.numel()
        else:
            out["other"] += p.numel()
    return out
