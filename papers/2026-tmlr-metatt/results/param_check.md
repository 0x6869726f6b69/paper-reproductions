# 表 2 の MetaTT-4D のパラメータ数（Q・V に適用、M = 2）

`python -m metatt_repro.count` の出力。RoBERTa を設定ファイルから組み立て（重みは読み込まない）、MetaTT-4D を組み込んで学習するパラメータを数えた。2026-11-25、Windows 11・Python 3.11.9・torch 2.9.1（CPU）・transformers 5.24.0。

| モデル | ランク | 数えた値 | 式 2Dr+(L+M)r² | 論文（×10³） | 一致 |
|---|---|---|---|---|---|
| roberta-base | 8 | 13,184 | 13,184 | 13 | ✓ |
| roberta-base | 24 | 44,928 | 44,928 | 45 | ✓ |
| roberta-base | 64 | 155,648 | 155,648 | 156 | ✓ |
| roberta-large | 16 | 39,424 | 39,424 | 39 | ✓ |
| roberta-large | 32 | 92,160 | 92,160 | 92 | ✓ |
