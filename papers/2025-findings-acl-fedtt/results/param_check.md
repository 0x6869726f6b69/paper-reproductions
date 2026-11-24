# Trainable parameters / uplink per round: our implementation vs. the paper

| setting | source | paper | ours (per round) | ours KB/round | match | breakdown (round 1) |
|---|---|---|---|---|---|---|
| DeBERTa-base FedTT r5 (binary) | Table 1 / 7 | 0.06M | 60,634 = 0.061M | 237 | yes | adapter_cores 37,440, adapter_bias 19,968, head 3,226 |
| DeBERTa-base FedTT r5 (MNLI) | Table 1 | 0.06M | 61,403 = 0.061M | 240 | yes | adapter_cores 37,440, adapter_bias 19,968, head 3,995 |
| DeBERTa-base FedTT+ r5 (binary) | Table 1 | 0.02M | 16,666-21,466 (mean 18,266) = 0.018M | 71 | yes | adapter_cores 13,440, head 3,226 |
| DeBERTa-base FedTT r2 | Table 7 | 0.03M | 29,362 = 0.029M | 115 | yes | adapter_cores 6,912, adapter_bias 19,968, head 2,482 |
| DeBERTa-base FedTT r10 | Table 7 | 0.17M | 167,794 = 0.168M | 655 | yes | adapter_cores 142,080, adapter_bias 19,968, head 5,746 |
| DeBERTa-base LoRA r8 (head not counted) | Table 1 | 0.30M | 294,912 = 0.295M | 1152 | **no** | head 592,130, lora 294,912 |
| DeBERTa-base LoRA r4 (head not counted) | Table 1 | 0.15M | 147,456 = 0.147M | 576 | yes | head 592,130, lora 147,456 |
| RoBERTa-base FedTT r5 | Table 2 | 0.06M | 60,634 = 0.061M | 237 | yes | adapter_cores 37,440, adapter_bias 19,968, head 3,226 |
| RoBERTa-large FedTT r5, last 6 (SST-2) | Table 12 | 39K | 38,210 = 38.21K | 149 | **no** | adapter_cores 21,120, adapter_bias 13,056, head 4,034 |
| RoBERTa-large FedTT r5, last 9 (QNLI) | Table 12 | 55K | 55,298 = 55.30K | 216 | yes | adapter_cores 31,680, adapter_bias 19,584, head 4,034 |
| RoBERTa-large FedTT r5, last 9 (MNLI) | Table 12 | 55K | 56,323 = 56.32K | 220 | **no** | adapter_cores 31,680, adapter_bias 19,584, head 5,059 |
| RoBERTa-large FedTT+ r5, all (SST-2) | Table 12 | 28K | 30,914-50,114 (mean 37,314) = 37.31K | 146 | **no** | adapter_cores 26,880, head 4,034 |
| RoBERTa-large FedTT+ r5, all (MNLI) | Table 12 | 28K | 31,939-51,139 (mean 38,339) = 38.34K | 150 | **no** | adapter_cores 26,880, head 5,059 |
| RoBERTa-large FedTT+ r5, all (binary) | Table 13 | 0.03M | 30,914-50,114 (mean 37,314) = 0.037M | 146 | partly (min/max) | adapter_cores 26,880, head 4,034 |
| RoBERTa-large LoRA r2, last 9 (head not counted) | Table 12 | 74K | 73,728 = 73.73K | 288 | yes | head 1,051,650, lora 73,728 |

FedTT+ variants (DeBERTa-base, binary):

- plus_train_bias=False: rounds 1-6 [16666, 21466, 16666, 16666, 21466, 16666] -> mean 18,266
- plus_train_bias=True: rounds 1-6 [36634, 41434, 36634, 36634, 41434, 36634] -> mean 38,234

Paper uplink KB (Table 14/15) are 4 B x the rounded counts / 1024: 0.06M -> 234, 0.02M -> 78, 51K -> 199, 28K -> 109.

