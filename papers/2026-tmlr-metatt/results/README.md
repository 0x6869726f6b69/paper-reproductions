# 結果：表 2 の MRPC・RTE（12 本）

> **English.** Table 2 of the paper: RoBERTa-base, MetaTT-4D rank 8 (13,184 trainable parameters), GLUE MRPC and RTE, warm-up ratio 0.06 and 0.6, seeds 33305628 / 2025 / 42, 20 epochs. Run on Google Colab (A100 GPU, high-RAM runtime) on 2026-11-24 (UTC). The files are copied unchanged from the notebook's output folder. `summary.csv` has one row per run. `runs/<task>/<run>/result.json` holds the full record of each run (configuration, per-epoch validation accuracy and loss, environment).

## 中身
| ファイル | 中身 |
|---|---|
| `report.md` | 論文との比較表（3 シードの平均 ± 標準誤差、シードごとの最良とそのエポック） |
| `curves.png` | 学習曲線（12 本、エポックごとの検証データの正解率。横線と灰色の帯は論文の値 ± 標準誤差） |
| `summary.csv` | 1 行が 1 本のラン（列は下） |
| `runs/<タスク>/w<warmup>_s<シード>/result.json` | 各ランのすべての記録：設定、エポックごとの検証データの正解率（MRPC は F1 も）・学習損失・学習率・時間、学習の前後で分類ヘッドが変わっていないことの確認、実行環境 |
| `param_check.md` | 表 2 の MetaTT-4D の 5 行のパラメータ数の照合（手元の CPU で数えたもの。Colab でも同じ値） |

`report.md`・`curves.png`・`summary.csv`・`result.json` は、ノートブック `MetaTT_min_colab.ipynb` の出力（Google ドライブの `MyDrive/MetaTT_repro/results/`）をそのまま写しました。ランごとのログ（標準出力）は `result.json` と同じ内容なので入れていません。

`summary.csv` の列：

| 列 | 意味 |
|---|---|
| `task`, `warmup`, `seed` | タスク、warmup ratio、シード |
| `precision` | 数値の精度（すべて fp32） |
| `smoke` | 動作確認用の短いランか（すべて False） |
| `best_acc`, `best_epoch` | 20 エポック中で最も高かった検証データの正解率と、そのエポック |
| `final_acc` | 20 エポック目の正解率 |
| `before_training` | 学習前（0 エポック目）の正解率。分類ヘッドが乱数の初期値なので、どちらか一方のクラスだけを答え、MRPC は 68.38 か 31.62、RTE は 52.71 か 47.29 になる |
| `f1_at_best` | 最良のエポックでの F1（MRPC だけ。RTE は nan） |
| `trainable` | 学習したパラメータ数 |
| `head_frozen` | 分類ヘッドが学習の前後で変わっていないか |
| `train_min` | 学習にかかった時間（分）。6 本を同時に回したときの値 |
| `gpu`, `torch`, `transformers` | 実行環境 |
| `finished` | 終わった時刻（Colab の時計、UTC） |

## 実験環境
| 項目 | 内容 |
|---|---|
| 計算環境 | Google Colab、GPU ランタイム（**A100 GPU・ハイメモリ**） |
| GPU | NVIDIA A100-SXM4-80GB（1 枚） |
| Python・ライブラリ | Python 3.13.16、torch 2.14.1+cu130、transformers 5.22.1、datasets 4.8.5 |
| 数値の精度 | fp32（TF32 も使わない） |
| 同時実行 | 1 枚の GPU で 6 本ずつ。warmup 0.06 の 6 本のあとに warmup 0.6 の 6 本 |
| 時間 | 1 本あたりの学習時間は MRPC 約 25 分、RTE 約 22 分（6 本同時）。12 本で約 50 分 |
| 実行日 | 2026-11-24 17:15〜18:06 UTC（日本時間では 2026-11-25） |
| コードの版 | ノートブックのコードの版 `85a22862cd`。このリポジトリの `method/`・`reproduction/` のコードと同じ |
