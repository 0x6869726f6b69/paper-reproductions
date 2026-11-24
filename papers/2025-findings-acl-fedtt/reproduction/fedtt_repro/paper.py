"""Numbers reported by Ghiasvand et al., "Communication-Efficient and Tensorized Federated Fine-Tuning of Large
Language Models", Findings of ACL 2025, pp. 24192-24207 (transcribed from the published PDF, 2026-11-24).

Scores: MRPC F1, others accuracy (%). Column order in the paper differs between tables; here keys are tasks.
Rows marked 'ours: no' were copied by the authors from other papers (FedPETuning, RoLoRA, FFA-LoRA).
"""

T1 = {  # Table 1: DeBERTa-base, cross-silo, IID, 5 clients, 1 local epoch, best over 100 rounds
    "lora-r8":    dict(params="0.30M", mrpc=91.87, sst2=94.95, qnli=92.68, qqp=89.2, mnli=87.31, avg=91.10),
    "p-tuning":   dict(params="0.30M", mrpc=82.01, sst2=90.48, qnli=82.12, qqp=84.0, mnli=80.74, avg=83.87),
    "lora-r4":    dict(params="0.15M", mrpc=91.72, sst2=94.95, qnli=92.66, qqp=86.7, mnli=86.91, avg=90.58),
    "bitfit":     dict(params="0.10M", mrpc=91.33, sst2=94.72, qnli=91.89, qqp=88.4, mnli=86.02, avg=90.47),
    "rolora-r4":  dict(params="0.08M", mrpc=91.17, sst2=94.61, qnli=92.40, qqp=87.9, mnli=86.27, avg=90.47),
    "prompt":     dict(params="0.01M", mrpc=82.96, sst2=92.32, qnli=82.13, qqp=80.5, mnli=74.46, avg=82.47),
    "fedtt-r5":     dict(params="0.06M", mrpc=92.68, sst2=94.61, qnli=92.02, qqp=88.4, mnli=85.99, avg=90.74),
    "fedttplus-r5": dict(params="0.02M", mrpc=92.60, sst2=93.58, qnli=90.54, qqp=87.9, mnli=85.33, avg=89.99),
}

T2 = {  # Table 2: RoBERTa-base; MRPC/SST-2 cross-silo 10 clients, others LSCD 1000 clients (10/round)
    "prefix":   dict(params="3.50M", mrpc=88.1, sst2=93.7, qnli=84.6, qqp=81.8, mnli=80.4, avg=85.7),
    "p-tuning": dict(params="0.88M", mrpc=86.8, sst2=92.1, qnli=85.6, qqp=81.5, mnli=79.8, avg=85.2),
    "adapter":  dict(params="0.70M", mrpc=88.5, sst2=94.0, qnli=85.9, qqp=87.0, mnli=84.9, avg=88.1),
    "ia3":      dict(params="0.65M", mrpc=88.0, sst2=93.0, qnli=89.4, qqp=85.4, mnli=82.7, avg=87.7),
    "lora":     dict(params="0.30M", mrpc=89.8, sst2=94.4, qnli=86.0, qqp=86.5, mnli=84.7, avg=88.3),
    "bitfit":   dict(params="0.10M", mrpc=88.6, sst2=92.8, qnli=80.5, qqp=84.0, mnli=80.7, avg=85.3),
    "fedtt-r5": dict(params="0.06M", mrpc=88.9, sst2=93.8, qnli=88.9, qqp=86.2, mnli=84.2, avg=88.4),
}

# Table 3: RoBERTa-large, 3 / 20 / 50 clients (iid / mild / severe), 20 local updates. FedTT adapters in the
# last 6 (SST-2) / 9 (others) layers, FedTT+ in all layers. NOTE: the LoRA/FFA-LoRA/RoLoRA rows equal RoLoRA's
# Table 2 with the QQP and MNLI columns swapped (RoLoRA orders SST-2, QNLI, MNLI, QQP); kept as printed.
T3 = {
    "iid": {
        "lora-r2":      dict(params="68K", sst2=95.64, qnli=92.04, qqp=85.85, mnli=86.16, avg=89.92),
        "ffa-r2":       dict(params="34K", sst2=94.91, qnli=90.11, qqp=84.06, mnli=85.48, avg=88.64),
        "rolora-r2":    dict(params="34K", sst2=95.60, qnli=91.62, qqp=85.66, mnli=86.16, avg=89.76),
        "fedtt-r5":     dict(params="51K", sst2=94.38, qnli=93.01, qqp=88.30, mnli=87.20, avg=90.72),
        "fedttplus-r5": dict(params="28K", sst2=95.64, qnli=94.05, qqp=88.99, mnli=88.27, avg=91.74),
    },
    "mild": {
        "lora-r2":      dict(params="68K", sst2=94.27, qnli=86.91, qqp=81.22, mnli=82.07, avg=86.12),
        "ffa-r2":       dict(params="34K", sst2=93.92, qnli=89.58, qqp=80.51, mnli=82.62, avg=86.66),
        "rolora-r2":    dict(params="34K", sst2=94.84, qnli=90.77, qqp=85.13, mnli=85.10, avg=88.96),
        "fedtt-r5":     dict(params="51K", sst2=94.15, qnli=91.38, qqp=86.25, mnli=86.53, avg=89.58),
        "fedttplus-r5": dict(params="28K", sst2=95.64, qnli=92.60, qqp=87.76, mnli=88.11, avg=91.03),
    },
    "severe": {
        "lora-r2":      dict(params="68K", sst2=93.23, qnli=82.57, qqp=58.96, mnli=76.96, avg=77.93),
        "ffa-r2":       dict(params="34K", sst2=92.32, qnli=85.15, qqp=62.79, mnli=77.78, avg=79.51),
        "rolora-r2":    dict(params="34K", sst2=94.61, qnli=89.83, qqp=85.15, mnli=85.55, avg=88.78),
        "fedtt-r5":     dict(params="51K", sst2=94.38, qnli=90.55, qqp=85.47, mnli=85.27, avg=88.92),
        "fedttplus-r5": dict(params="28K", sst2=94.50, qnli=90.17, qqp=86.65, mnli=86.28, avg=89.40),
    },
}
T3_CLIENTS = {"iid": 3, "mild": 20, "severe": 50}

T7 = {  # Table 7: DeBERTa-base, 5 clients, 1 local update (epoch), lr 1e-3, bs 32, best over 20 rounds
    "fedtt-r2":  dict(params="0.03M", sst2=93.58, qnli=90.76, qqp=86.79, mnli=85.29, avg=89.10),
    "fedtt-r5":  dict(params="0.06M", sst2=93.00, qnli=91.87, qqp=87.65, mnli=85.91, avg=89.61),
    "fedtt-r10": dict(params="0.17M", sst2=93.69, qnli=92.48, qqp=88.22, mnli=87.01, avg=90.35),
}

T13 = {  # Table 13 (Appendix B): RoBERTa-large, 3 clients, label-ratio splits, 10 local updates
    "iid": {"lora-r8": dict(params="1.57M", sst2=94.42, qnli=91.38, qqp=84.47, mnli=86.90, avg=89.29),
            "ffa-r8": dict(params="0.79M", sst2=95.14, qnli=92.64, qqp=86.31, mnli=87.13, avg=90.30),
            "fedttplus-r5": dict(params="0.03M", sst2=95.41, qnli=94.05, qqp=88.15, mnli=88.65, avg=91.56)},
    "mild": {"lora-r8": dict(params="1.57M", sst2=93.55, qnli=91.36, qqp=84.41, mnli=87.01, avg=89.08),
             "ffa-r8": dict(params="0.79M", sst2=94.10, qnli=91.62, qqp=85.33, mnli=87.04, avg=89.52),
             "fedttplus-r5": dict(params="0.03M", sst2=95.53, qnli=93.54, qqp=87.91, mnli=88.45, avg=91.36)},
    "severe": {"lora-r8": dict(params="1.57M", sst2=94.32, qnli=88.95, qqp=83.51, mnli=82.03, avg=87.20),
               "ffa-r8": dict(params="0.79M", sst2=94.32, qnli=90.35, qqp=84.35, mnli=85.05, avg=88.52),
               "fedttplus-r5": dict(params="0.03M", sst2=95.64, qnli=91.67, qqp=86.66, mnli=87.73, avg=90.42)},
}

# Table 12: trainable parameters for Table 3 (RoBERTa-large)
T12 = {"lora-r2": dict(sst2="49K", qnli="74K", qqp="74K", mnli="74K"),
       "ffa-r2": dict(sst2="25K", qnli="37K", qqp="37K", mnli="37K"),
       "rolora-r2": dict(sst2="25K", qnli="37K", qqp="37K", mnli="37K"),
       "fedtt-r5": dict(sst2="39K", qnli="55K", qqp="55K", mnli="55K"),
       "fedttplus-r5": dict(sst2="28K", qnli="28K", qqp="28K", mnli="28K")}

# Table 14 (Table 1 setting) and Table 15 (Table 3 severe): uplink KB per round and rounds to reach 95 % of the
# Table 1 / Table 3 score. KB = params * 4 B / 1024 computed from the rounded counts (0.06M -> 234 KB).
T14 = {"lora-r4": dict(kb=586, rounds=dict(sst2=2, qnli=2, qqp=2, mnli=2)),
       "rolora-r4": dict(kb=312, rounds=dict(sst2=2, qnli=2, qqp=1, mnli=2)),
       "fedtt-r5": dict(kb=234, rounds=dict(sst2=2, qnli=2, qqp=2, mnli=2)),
       "fedttplus-r5": dict(kb=78, rounds=dict(sst2=5, qnli=6, qqp=3, mnli=3))}
T15 = {"rolora-r2": dict(kb=133, rounds=dict(sst2=3, qnli=10, qqp=11, mnli=11)),
       "fedtt-r5": dict(kb=199, rounds=dict(sst2=3, qnli=3, qqp=3, mnli=3)),
       "fedttplus-r5": dict(kb=109, rounds=dict(sst2=3, qnli=3, qqp=3, mnli=3))}

TASK_ORDER = ["mrpc", "sst2", "qnli", "qqp", "mnli"]
