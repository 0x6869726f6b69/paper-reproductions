"""Tensor-train (TT) linear layer of FedTT (Ghiasvand et al., Findings of ACL 2025, Sec. 3.2 and 4.1).

A weight W in R^{P x Q} is reshaped into a tensor in R^{k_1 x ... x k_J} (prod k_j = P*Q) and stored as
J cores G_j in R^{r_{j-1} x k_j x r_j} with r_0 = r_J = 1 (paper Eq. 1):

    W = G_1 x_{3,1} G_2 x_{3,1} ... x_{3,1} G_J .

The first modes k_1..k_m multiply to P (input side), the rest to Q (output side); e.g. Table 10 gives
[8, 8, 12, 8, 8] for 768 x 64 (input 8*8*12, output 8*8) and for 64 x 768 (input 8*8, output 12*8*8).

Forward ("contract", the paper's way): the input activations are contracted with the cores one by one and
W is never formed ("the tensor factors are directly contracted with the vector of activation values").
"dense" builds W from the cores first and calls one matmul; both compute the same function and gradients
(checked in tests/test_core.py) and differ only in speed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_modes(shape, in_features, out_features):
    """Split a TT shape into (input modes, output modes): the shortest prefix whose product is P."""
    p = 1
    for m, k in enumerate(shape):
        p *= int(k)
        if p == in_features:
            in_modes, out_modes = tuple(shape[:m + 1]), tuple(shape[m + 1:])
            if math.prod(out_modes) != out_features:
                break
            return in_modes, out_modes
    raise ValueError(f"TT shape {shape} does not factor {in_features} x {out_features}")


class TTLinear(nn.Module):
    """y = x W + b with W in TT format (rank r between all neighbouring cores)."""

    def __init__(self, in_features, out_features, shape, rank, bias=True, w_std=0.02, mode="contract"):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.shape = tuple(int(k) for k in shape)
        self.in_modes, self.out_modes = split_modes(self.shape, self.in_features, self.out_features)
        self.n_in, self.J = len(self.in_modes), len(self.shape)
        self.rank = int(rank)
        self.ranks = [1] + [self.rank] * (self.J - 1) + [1]
        self.cores = nn.ParameterList(
            [nn.Parameter(torch.empty(self.ranks[j], self.shape[j], self.ranks[j + 1])) for j in range(self.J)])
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.mode = mode
        self.reset_parameters(w_std)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_parameters(self, w_std):
        """i.i.d. N(0, s^2) core entries with s chosen so that every entry of W has std `w_std`:
        W_ij is a sum of prod(r_1..r_{J-1}) products of J core entries, so Var = prod(r) * s^(2J).
        (The paper does not state its initialization; see README.)"""
        n_terms = math.prod(self.ranks[1:-1])
        s = (w_std ** 2 / n_terms) ** (1.0 / (2 * self.J))
        for c in self.cores:
            c.normal_(0.0, s)
        if self.bias is not None:
            self.bias.zero_()

    # ------------------------------------------------------------------ forward
    def weight(self):
        """Dense W (P x Q), rebuilt from the cores (used by mode='dense' and by the tests)."""
        W = self.cores[0].reshape(-1, self.ranks[1])                     # (k_1, r_1)
        for j in range(1, self.J):
            G = self.cores[j]
            W = (W @ G.reshape(self.ranks[j], -1)).reshape(-1, self.ranks[j + 1])
        return W.reshape(self.in_features, self.out_features)

    def _contract(self, x2):
        B = x2.shape[0]
        t = x2.reshape(B, 1, self.in_features)                          # (B, r_0, rest)
        rest = self.in_features
        for j in range(self.n_in):                                       # absorb input modes k_1..k_m
            G = self.cores[j]
            k = G.shape[1]
            rest //= k
            t = torch.einsum("bakn,aks->bsn", t.reshape(B, G.shape[0], k, rest), G)
        t = t.reshape(B, 1, self.ranks[self.n_in])                       # (B, 1, r_m)
        for j in range(self.n_in, self.J):                               # emit output modes
            G = self.cores[j]
            t = torch.einsum("bpa,aks->bpks", t, G).reshape(B, -1, G.shape[2])
        return t.reshape(B, self.out_features)

    def forward(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.mode == "dense":
            y = F.linear(x2, self.weight().t().to(x2.dtype))
        else:
            y = self._contract(x2)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*lead, self.out_features)

    # ------------------------------------------------------------------ FedTT+
    def set_trainable_cores(self, idx, train_bias):
        """Only cores with 0-based index in `idx` get gradients (Alg. 2, lines 5-6)."""
        idx = set(idx)
        for j, c in enumerate(self.cores):
            c.requires_grad_(j in idx)
        if self.bias is not None:
            self.bias.requires_grad_(bool(train_bias))

    def n_core_params(self):
        return sum(c.numel() for c in self.cores)

    def extra_repr(self):
        return (f"{self.in_features}x{self.out_features}, shape={list(self.shape)} "
                f"(in {list(self.in_modes)} / out {list(self.out_modes)}), rank={self.rank}, "
                f"core params={self.n_core_params()}, bias={self.bias is not None}, mode={self.mode}")


def plus_core_index(t, J, schedule="text"):
    """FedTT+ (Alg. 2, line 3): the middle core trained in round t (1-based round; returns a 1-based index).

    'text'    r in {2, ..., J-1}, cycling 2, 3, ..., J-1, 2, ... (Sec. 4.3: "an index r is selected from the
              set r in {2, ..., J-1}", with G_1 and G_J always trained).
    'literal' the pseudocode as printed, mod(t, J-2) = r-1, i.e. r = (t mod (J-2)) + 1 in {1, ..., J-2};
              it never trains G_{J-1} and in some rounds trains only {G_1, G_J}. Kept for comparison.
    """
    if J < 3:
        raise ValueError("FedTT+ needs at least three cores")
    if schedule == "text":
        return 2 + (t - 1) % (J - 2)
    if schedule == "literal":
        return (t % (J - 2)) + 1
    raise ValueError(schedule)
