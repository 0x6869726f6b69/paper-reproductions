"""One communication round of Algorithm 1 (FedTT) and Algorithm 2 (FedTT+), and client selection.

Per communication round t (1-based):
  FedTT  (Alg. 1)  every selected client starts from the global trainable parameters w^(t), performs K local
                   updates (AdamW) on its data, and sends w_i^(t)+K; the server sets w^(t+1) = mean_i w_i^(t)+K
                   and broadcasts it.
  FedTT+ (Alg. 2)  the same, but in each tensorized adapter layer only {G_1, G_r, G_J} are trainable
                   (r from tt.plus_core_index, set by FedModel.set_round), the other cores stay frozen and
                   identical on all clients; clients send and the server averages and broadcasts only
                   {G_1, G_r, G_J} (plus the classification head, which is always trainable).
Cross-silo: all N clients take part in every round. Large-scale cross-device (LSCD): `clients_per_round`
clients are drawn uniformly without replacement each round.

One model instance is reused for all clients (sequential simulation); clients are stateless (a fresh AdamW
per client and round), which is what the pseudocode describes. A client's data is any object with len() and
batch(positions, pad_id) -> (input_ids, attention_mask, labels) as numpy arrays.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def local_batches(n_local, cfg, t, c):
    """Positions (relative to the client's data) of the K minibatches of client c in round t."""
    rng = np.random.default_rng([int(cfg["seed"]), int(t), int(c), 7])
    bs = int(cfg["bs"])
    if cfg.get("local_steps"):
        K = int(cfg["local_steps"])
        need = K * bs
        order = np.concatenate([rng.permutation(n_local) for _ in range(math.ceil(need / n_local))])[:need]
        return [order[i * bs:(i + 1) * bs] for i in range(K)]
    E = float(cfg["local_epochs"])
    order = np.concatenate([rng.permutation(n_local) for _ in range(math.ceil(E))])[:int(round(E * n_local))]
    return [order[i:i + bs] for i in range(0, len(order), bs)]


def run_round(fm, gstate, clients, sel, cfg, t, device, pad_id, amp_dtype):
    """One communication round of Alg. 1 / Alg. 2. Updates `gstate` in place and returns round statistics.

    Every selected client loads the global parameters, runs its K local AdamW updates on the parameters
    that are trainable in round t, and uploads exactly those; the server averages them (uniform 1/|S| as
    in the pseudocode, or weighted by client size with cfg agg='weighted') and keeps all other entries.
    """
    params = dict(fm.hf.named_parameters())
    sched = fm.set_round(t)
    tr = fm.trainable()
    tr_names = [n for n, _ in tr]

    def load_global():
        with torch.no_grad():
            for n, v in gstate.items():
                params[n].copy_(v)

    acc = {n: torch.zeros_like(gstate[n], dtype=torch.float32) for n in tr_names}
    wsum, loss_sum, n_steps = 0.0, torch.zeros((), device=device), 0
    fm.train()
    for c in sel:
        load_global()
        opt = torch.optim.AdamW([p for _, p in tr], lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        cd = clients[c]
        micro = int(cfg.get("micro_bs") or 0)
        for rel in local_batches(len(cd), cfg, t, c):
            opt.zero_grad(set_to_none=True)
            chunks = [rel] if not micro or len(rel) <= micro else [rel[i:i + micro] for i in range(0, len(rel), micro)]
            loss = torch.zeros((), device=device)
            for ch in chunks:            # gradient accumulation: same mean-loss gradient as the full minibatch
                x, msk, y = cd.batch(ch, pad_id)
                x, msk, y = (torch.from_numpy(a).to(device, non_blocking=True) for a in (x, msk, y))
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = fm(x, msk)
                part = F.cross_entropy(logits.float(), y) * (len(ch) / len(rel))
                part.backward()
                loss = loss + part.detach()
            if cfg["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_([p for _, p in tr], float(cfg["max_grad_norm"]))
            opt.step()
            loss_sum += loss.detach()
            n_steps += 1
        w = float(len(cd)) if cfg["agg"] == "weighted" else 1.0
        with torch.no_grad():
            for n in tr_names:
                acc[n] += w * params[n].detach().float()
        wsum += w
    with torch.no_grad():                        # server: average what was uploaded, keep the rest
        for n in tr_names:
            gstate[n] = (acc[n] / wsum).to(gstate[n].dtype)
    load_global()
    return dict(sched=sched, loss=float(loss_sum) / max(n_steps, 1), steps=n_steps, clients=len(sel),
                up_params=int(sum(p.numel() for _, p in tr)), uploaded=tr_names)


def select_clients(cfg, t):
    N, m = int(cfg["n_clients"]), cfg.get("clients_per_round")
    if m and int(m) < N:                           # LSCD: m clients drawn uniformly each round
        return np.sort(np.random.default_rng([int(cfg["seed"]), int(t), 3]).choice(N, int(m), replace=False))
    return np.arange(N)                            # cross-silo: everyone
