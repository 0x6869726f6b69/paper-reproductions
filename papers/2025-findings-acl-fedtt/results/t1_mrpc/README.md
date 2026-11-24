# 結果：表 1 の MRPC（16 本）

> **English.** Table 1 / MRPC of the paper: DeBERTa-base, 5 IID clients, 1 local epoch, 100 rounds, the paper's 8-point grid (lr × batch) for FedTT and FedTT+, seed 0. Run on Google Colab (A100 GPU, high-RAM runtime) on 2026-11-24. `summary.csv` has one row per run.

## 中身
`summary.csv` の 1 行が 1 本のランです。値は、ノートブック `FedTT_MRPC_colab.ipynb` の「結果（論文との比較）」のセルが出した表をそのまま写したものです。

| 列 | 意味 |
|---|---|
| `method` | `fedtt` = FedTT、`fedttplus` = FedTT+ |
| `lr`, `bs` | 学習率、バッチサイズ |
| `best_F1` | GLUE 検証データ（408 件）での F1 の、100 ラウンド中の最高値（論文の「best validation results」と同じ取り方） |
| `best_round` | その最高値が出たラウンド |
| `F1_heldout_selected` | held-out（訓練データから取り分けた 367 件）で最高だったラウンドの、検証データでの F1（検証データを見ずにラウンドを選んだ値） |
| `final_F1` | 100 ラウンド目の F1 |
| `uplink_KB` | 1 ラウンド・1 クライアントあたりの送信量（学習パラメータ数 × 4 バイト / 1024 の平均） |
| `sec_per_round` | 1 ラウンドにかかった時間（秒）。同時に走らせた本数で変わる（下） |
| `gpu` | 使った GPU |

## 実験環境
| 項目 | 内容 |
|---|---|
| 計算環境 | Google Colab、GPU ランタイム（**A100 GPU・ハイメモリ**） |
| GPU | NVIDIA A100-SXM4-40GB（1 枚） |
| Python | 3.13 |
| 数値の精度 | fp32（行列積は TF32）。DeBERTa は bf16 の自動混合精度で動かないため（transformers 側の不具合。`説明資料.md` §4） |
| 同時実行 | 1 枚の GPU で、ステップ A（学習率 1e-3・バッチ 32 の 2 本）は 2 本、ステップ B（残りの 14 本）は 6 本を同時に回した |
| 実行日 | 2026-11-24 |
| コードの版 | ノートブックのコードの版 `de7bfd5ed5`。いまのリポジトリの版との違いは、学習に関係しない部分（パラメータ数の照合の例外処理、Colab の torchao の削除、コードを `method/` と `reproduction/` に分けたこと）だけ。分けた前後で、小さなモデルの学習結果がテンソル単位で完全に一致することを確かめた |

`sec_per_round` は、2 本同時のとき 17〜19 秒、6 本同時のときバッチ 32 で約 31 秒、バッチ 16 で 39〜44 秒でした（FedTT+ の学習率 1e-4 の 2 本は、終盤に同時に走る本数が減ったため短い）。

## まだ入っていないもの
ラウンドごとのログ（各ランの `log.jsonl` と `result.json`）は Google ドライブ（`MyDrive/FedTT_repro/results/runs/t1/mrpc/`）にあり、このリポジトリにはまだ入れていません。入れれば、学習曲線と、held-out での最高値（論文の表 8 の Dev 367 件に相当）も載せられます。
