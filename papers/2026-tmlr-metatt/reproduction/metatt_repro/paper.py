"""Settings and reported numbers taken from the paper (arXiv:2506.09105; v2 = 2025-11, v3 = TMLR version 2026-07).

Section and table numbers are given for v2 / v3. Values below are identical in both versions.
"""

# Table 2, RoBERTa-base, MetaTT-4D, rank 8 (13k trainable parameters). Accuracy on the GLUE validation set.
# The number in parentheses in the paper is the standard error (rounded up to one significant digit).
PAPER_RESULTS = {
    "mrpc": {"value": 87.6, "se": 0.2, "metric": "accuracy"},
    "rte": {"value": 72.9, "se": 0.5, "metric": "accuracy"},
}
PAPER_PARAMS_K = 13                                   # Table 2 "Param x 10^3"

# Not from the paper: accuracy of always predicting the majority class of the GLUE validation set
# (MRPC 279/408 "equivalent", RTE 146/277 "not_entailment"). A run stuck at this value has learned nothing.
MAJORITY_BASELINE = {"mrpc": 68.38, "rte": 52.71}

# Table 6 (v2) / Table 14 (v3), first RoBERTa-base block. The block is printed with rank "4", but Table 2 has no
# rank-4 row and its 13k parameters require rank 8 (rank 4 gives 6,368), so we read the block as the rank-8 one.
TASK_HPARAMS = {
    "mrpc": {"alpha": 0.5, "lr": 1e-3, "batch_size": 8},
    "rte": {"alpha": 4.0, "lr": 1e-3, "batch_size": 16},
}

SEEDS = {
    "FacebookAI/roberta-base": (33305628, 2025, 42),  # Appendix D.1 / G.1
    "FacebookAI/roberta-large": (56346, 2025, 42),
}

COMMON = {
    "model": "FacebookAI/roberta-base",
    "rank": 8,
    "target_modules": ("query", "value"),   # Appendix D.2 / G.2: every main-text result adapts Q and V
    "init": "ze-id-id-id",                  # Appendix D.4 / G.4: G1 = 0, other cores identity ("majority of experiments")
    "epochs": 20,                           # Appendix D.4 / G.4: final evaluations run 20 epochs, 3 trials
    "max_len": 256,                         # Appendix D.4 / G.4 (grid paragraph)
    "weight_decay": 0.0,                    # same paragraph
    "max_grad_norm": 1.0,                   # not stated for GLUE; HF Trainer default (the paper uses the Trainer, D.3 / G.3)
}

# Warm-up ratio: the GLUE paragraph prints 0.6 (v2 and v3); every other experiment of the paper uses 0.06.
# We run both: 0.06 (likely intended) and 0.6 (as printed).
WARMUPS = (0.06, 0.6)

# Table 2 MetaTT-4D rows, used by count.py: (model, rank) -> parameters x 10^3 as printed.
TABLE2_PARAMS_K = {
    ("FacebookAI/roberta-base", 8): 13, ("FacebookAI/roberta-base", 24): 45, ("FacebookAI/roberta-base", 64): 156,
    ("FacebookAI/roberta-large", 16): 39, ("FacebookAI/roberta-large", 32): 92,
}
