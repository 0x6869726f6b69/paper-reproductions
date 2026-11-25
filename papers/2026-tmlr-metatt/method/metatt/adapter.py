"""MetaTT-4D: one tensor train (TT) shared by every adapted linear map of a transformer.

Lopez-Piqueres et al., "MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning",
TMLR 2026 (arXiv:2506.09105). Written from the paper: Sec. 2.3, Eq. (5) and the pseudo-code of
Appendix E (v2) / Appendix H (v3), Algorithms 2-5.

For layer l and projection type m (e.g. m = 0 for Q, 1 for V) the adapted map is (Eq. (5))

    y = x W_{l,m}^T + b + alpha * x G1 G2[l] G3[m] G4

    G1 in R^{D_in x r1}      input core (shared by every layer and type)
    G2 in R^{L x r1 x r2}    layer core (slice G2[l] is an r1 x r2 matrix)
    G3 in R^{M x r2 x r3}    projection-type core
    G4 in R^{r3 x D_out}     output core

Trainable parameters: D_in r1 + L r1 r2 + M r2 r3 + r3 D_out, i.e. 2 D r + (L + M) r^2 for equal ranks.

Differences from the paper's pseudo-code (none change the computation):
  * G2 and G3 are stored as one 3-way tensor each instead of a ParameterList of r x r matrices.
    Element-wise optimisers (AdamW) and global-norm clipping treat both layouts identically.
  * The pseudo-code's typos are fixed (`self.M_2` vs `self.M2`; `ParameterList[...]` for `ParameterList([...])`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

INIT_CODES = ("ze", "id", "no")       # zeros / identity on every matrix slice / normal(0, init_std)


@dataclass
class MetaTT4DConfig:
    """Algorithm 2. `rank` is one int (all three TT bonds equal, as in the paper) or (r1, r2, r3)."""
    rank: int | tuple[int, int, int] = 8
    alpha: float = 1.0                                  # scaling of the adapter output (Eq. (5)); not divided by r
    target_modules: tuple[str, ...] = ("query", "value")  # leaf names of the nn.Linear modules to adapt
    init: str = "ze-id-id-id"                            # init of G1-G2-G3-G4 (Appendix D.4 / G.4, Fig. 5)
    init_std: float = 0.2                                # std of the "no" init (Appendix D.4)

    @property
    def ranks(self) -> tuple[int, int, int]:
        r = (self.rank,) * 3 if isinstance(self.rank, int) else tuple(self.rank)
        if len(r) != 3 or min(r) < 1:
            raise ValueError(f"rank must be a positive int or three positive ints, got {self.rank!r}")
        return r

    @property
    def init_codes(self) -> tuple[str, str, str, str]:
        codes = tuple(self.init.split("-"))
        if len(codes) != 4 or any(c not in INIT_CODES for c in codes):
            raise ValueError(f"init must be four of {INIT_CODES} joined by '-', got {self.init!r}")
        return codes


def metatt4d_num_params(d_in: int, d_out: int, num_layers: int, num_types: int, rank) -> int:
    """Closed-form parameter count (Sec. 2.3, 'Complexity Analysis')."""
    r1, r2, r3 = MetaTT4DConfig(rank=rank).ranks
    return d_in * r1 + num_layers * r1 * r2 + num_types * r2 * r3 + r3 * d_out


class MetaTT4D(nn.Module):
    """The four TT cores (Algorithm 3) and the adapter term alpha * x G1 G2[l] G3[m] G4."""

    def __init__(self, d_in: int, d_out: int, num_layers: int, num_types: int, cfg: MetaTT4DConfig):
        super().__init__()
        r1, r2, r3 = cfg.ranks
        self.cfg = cfg
        self.alpha = float(cfg.alpha)
        self.G1 = nn.Parameter(torch.empty(d_in, r1))
        self.G2 = nn.Parameter(torch.empty(num_layers, r1, r2))
        self.G3 = nn.Parameter(torch.empty(num_types, r2, r3))
        self.G4 = nn.Parameter(torch.empty(r3, d_out))
        self.reset_parameters()

    def cores(self) -> tuple[nn.Parameter, ...]:
        return self.G1, self.G2, self.G3, self.G4

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for core, code in zip(self.cores(), self.cfg.init_codes):
            if code == "ze":
                core.zero_()
            elif code == "no":
                core.normal_(0.0, self.cfg.init_std)
            else:                                        # "id": identity on every matrix slice (nn.init.eye_)
                core.zero_()
                mats = core if core.dim() == 3 else core.unsqueeze(0)
                k = min(mats.shape[-2], mats.shape[-1])
                idx = torch.arange(k, device=core.device)
                mats[:, idx, idx] = 1.0

    def forward(self, x: torch.Tensor, layer: int, mtype: int) -> torch.Tensor:
        # Algorithm 4: (((X M1) M2) M3) M4, cheapest order when the rank is below the number of tokens
        h = x @ self.G1
        h = h @ self.G2[layer]
        h = h @ self.G3[mtype]
        return self.alpha * (h @ self.G4)

    def delta_weight(self, layer: int, mtype: int) -> torch.Tensor:
        """Dense update Delta W_{l,m} (shape D_in x D_out) so that the adapter term equals x @ Delta W."""
        return self.alpha * (self.G1 @ self.G2[layer] @ self.G3[mtype] @ self.G4)

    def extra_repr(self) -> str:
        (d_in, r1), (num_layers, _, r2), (num_types, _, r3), (_, d_out) = (c.shape for c in self.cores())
        return (f"D_in={d_in}, L={num_layers}, M={num_types}, D_out={d_out}, ranks=({r1}, {r2}, {r3}), "
                f"alpha={self.alpha}, init={self.cfg.init}")


class MetaTTLinear(nn.Module):
    """Algorithm 4: frozen nn.Linear plus the shared MetaTT term for its (layer, type) slice."""

    def __init__(self, base: nn.Linear, adapter: MetaTT4D, layer: int, mtype: int):
        super().__init__()
        self.base = base
        # Shared by every wrapped layer and registered once on the model root (as `metatt`);
        # stored outside nn.Module's registry so the cores are not duplicated in state_dict().
        object.__setattr__(self, "_adapter", adapter)
        self.layer = layer
        self.mtype = mtype

    @property
    def adapter(self) -> MetaTT4D:
        return self._adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self._adapter(x, self.layer, self.mtype)

    def extra_repr(self) -> str:
        return f"layer={self.layer}, type={self.mtype}"


_INDEX = re.compile(r"\.(\d+)\.")


def find_targets(model: nn.Module, target_modules) -> list[tuple[str, int, int, nn.Linear]]:
    """(module name, layer number, type index, module) for every nn.Linear named like `...layer.<l>....<target>`.

    The layer number is the last integer component of the module path, which is the transformer block index
    for RoBERTa/BERT (`roberta.encoder.layer.3.attention.self.query`) and Llama (`model.layers.3.self_attn.q_proj`).
    """
    out = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Linear) and leaf in target_modules:
            nums = _INDEX.findall(name + ".")
            if nums:
                out.append((name, int(nums[-1]), list(target_modules).index(leaf), mod))
    return out


def apply_metatt(model: nn.Module, cfg: MetaTT4DConfig) -> MetaTT4D:
    """Algorithm 5: freeze the whole model (classifier head included), build one MetaTT-4D shared by every
    target linear map and swap each of those maps for a MetaTTLinear. Returns the adapter, which is also
    registered on the model as `model.metatt`, so its four cores are the only trainable parameters."""
    if hasattr(model, "metatt"):
        raise ValueError("model already has a MetaTT adapter")
    targets = find_targets(model, cfg.target_modules)
    if not targets:
        raise ValueError(f"no nn.Linear named {cfg.target_modules} inside numbered blocks")
    layers = sorted({t[1] for t in targets})
    types = sorted({t[2] for t in targets})
    if types != list(range(len(cfg.target_modules))):
        missing = [m for i, m in enumerate(cfg.target_modules) if i not in types]
        raise ValueError(f"target modules not found: {missing}")
    seen = {(t[1], t[2]) for t in targets}
    if len(seen) != len(targets) or len(seen) != len(layers) * len(types):
        raise ValueError("every block must contain each target module exactly once")
    d_in = {t[3].in_features for t in targets}
    d_out = {t[3].out_features for t in targets}
    if len(d_in) != 1 or len(d_out) != 1:
        raise ValueError(f"shared G1/G4 need equal shapes, got in={d_in} out={d_out}")

    for p in model.parameters():
        p.requires_grad_(False)
    ref = targets[0][3].weight
    adapter = MetaTT4D(d_in.pop(), d_out.pop(), len(layers), len(types), cfg).to(ref.device)
    model.add_module("metatt", adapter)
    pos = {l: i for i, l in enumerate(layers)}
    for name, layer, mtype, mod in targets:
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, MetaTTLinear(mod, adapter, pos[layer], mtype))
    return adapter


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
