# FedTT / FedTT+（Findings of ACL 2025）の再現

> **English summary.** Independent re-implementation of FedTT and FedTT+ (Ghiasvand et al., *Findings of ACL 2025*), written from the paper's Algorithms 1–2 and appendix because no official code is available. On **Table 1 / MRPC** (DeBERTa-base, 5 IID clients, 100 rounds, the paper's 8-point learning-rate × batch-size grid) we obtain **F1 92.50 for FedTT (paper: 92.68) and 93.24 for FedTT+ (paper: 92.60)**, within one point, and the paper's trainable-parameter counts are matched exactly (60,634 and 18,266 on average per round). Environment: Google Colab, A100 GPU (high-RAM runtime), fp32. Caveats: one seed; FedTT+ needs learning rate 5e-3; which validation split the paper used is unclear. Details in `説明資料.md` (Japanese).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x6869726f6b69/paper-reproductions/blob/main/papers/2025-findings-acl-fedtt/reproduction/FedTT_MRPC_colab.ipynb)
（表 1 の MRPC を回したノートブックを Colab で開く）

## 論文
- Sajjad Ghiasvand, Yifan Yang, Zhiyu Xue, Mahnoosh Alizadeh, Zheng Zhang, Ramtin Pedarsani. "Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models." *Findings of the Association for Computational Linguistics: ACL 2025*, pp. 24192–24207. [ACL Anthology](https://aclanthology.org/2025.findings-acl.1241/)（doi:[10.18653/v1/2025.findings-acl.1241](https://doi.org/10.18653/v1/2025.findings-acl.1241)）
- **公式コード**：なし（論文と ACL Anthology のページにリンクなし。2026-11 に確認）。ここのコードは、論文の擬似コード（Algorithm 1・2）と本文・付録の記述だけから書いた、独立した再実装です。
- **提案手法**：連合学習で言語モデルを微調整するとき、アダプタの重みをテンソルトレイン（TT）分解の小さなコアで持ち、コアだけを学習・送信する（FedTT）。毎ラウンド一部のコアだけを送る FedTT+ は、送信量をさらに減らす。

## 結果：表 1 の MRPC
**判定：✅ 再現できた（シード 1 つ）**。FedTT・FedTT+ とも論文の値と 1 点以内です。

| 手法 | 本再現（F1、グリッドの最大） | そのときの学習率・バッチ（ラウンド） | 論文（表 1） | 差 | 学習パラメータ数（1 ラウンド） | 送信量（1 ラウンド・1 クライアント） |
|---|---|---|---|---|---|---|
| FedTT | **92.50** | 1e-3・32（64 ラウンド目） | 92.68 | −0.18 | 60,634（論文 0.06M） | 236.9 KB |
| FedTT+ | **93.24** | 5e-3・16（87 ラウンド目） | 92.60 | +0.64 | 平均 18,266（論文 0.02M） | 71.3 KB |

- 指標は GLUE の検証データ（408 件）の F1。各ランの 100 ラウンド中の最高値をとり、学習率 4 通り × バッチ 2 通りの 8 本の中で最大のものを報告しています（論文と同じ取り方）。
- 全 16 本の値は [`results/t1_mrpc/summary.csv`](results/t1_mrpc/summary.csv)、パラメータ数の照合は [`results/param_check.md`](results/param_check.md)、考察は [`説明資料.md`](説明資料.md) の §5 にあります。

**わかったこと**
- **差は数件分**：検証データが 408 件と小さく、1 件の答えが変わると F1 が 0.15〜0.2 点動きます。FedTT の差はおよそ 1 件、FedTT+ の差は 3〜4 件分です。シード 1 つなので、FedTT+ が FedTT を上回ったとは言えません。
- **FedTT+ は大きい学習率が必要**：FedTT は学習率 5e-4〜5e-3 で 90.7〜92.5 と安定していますが、FedTT+ が論文の値に届くのは 5e-3 のときだけです（1e-3 で 88〜90、1e-4 では学習しない）。論文にはこの違いは書かれていません。
- **論文の取り方はやや楽観的**：検証データを見ずに、held-out（訓練データから取り分けた 367 件）でラウンドを選ぶと、FedTT 92.14・FedTT+ 92.44 です。

## 実験環境
| 項目 | 内容 |
|---|---|
| 計算環境 | **Google Colab、A100 GPU・ハイメモリのランタイム** |
| GPU | NVIDIA A100-SXM4-40GB（1 枚）。論文は NVIDIA A6000・V100 |
| Python | 3.13（Colab の既定） |
| 数値の精度 | fp32（行列積は TF32）。DeBERTa は bf16 の自動混合精度で動かないため（transformers 側の不具合） |
| 同時実行 | 1 枚の GPU で 2 本（ステップ A）または 6 本（ステップ B）を同時に |
| 時間 | 1 ラウンド 17〜44 秒（同時に走る本数とバッチによる）。100 ラウンドで 30〜75 分。ステップ A の 2 本は約 30 分 |
| 実行日 | 2026-11-24 |

テストとパラメータ数の照合は、ノート PC の CPU（PyTorch 2.9.1 の CPU 版）で行いました。

## 論文との条件の比較（表 1・MRPC）
✓ = 論文と同じ、△ = 論文の記述を解釈した、? = 論文に書かれていない（こちらで決めた）、**要検討** = 同じかどうか確かめられていない。

| 項目 | 論文の記述 | 本再現 | 一致 |
|---|---|---|---|
| モデル | DeBERTa-Base | `microsoft/deberta-base` | △ チェックポイント名は書かれていない |
| 連合学習の設定 | cross-silo、5 クライアント、i.i.d.（表 1） | 5 クライアント、IID、毎ラウンド全員参加 | ✓ |
| ラウンド数 | 100（付録 A.3） | 100 | ✓ |
| ローカル学習 | 「local training epochs 1」（5.2 節）と「one local update per communication round」（付録 A.3） | 1 ローカルエポック | △ 2 通りの書き方があり、エポックと解釈 |
| 学習率・バッチ | 学習率 {5e-3, 1e-3, 5e-4, 1e-4} × バッチ {16, 32}（付録 A.3） | 同じ 8 通り | ✓ |
| 最適化 | AdamW（5.1 節） | AdamW、重み減衰 0、学習率一定、ラウンドごとに新しい状態 | △ AdamW 以外は書かれていない |
| テンソルの形・ランク | 768×64・64×768 は [8,8,12,8,8]、分類器 768×768 は [12,8,8,8,8,12]、ランク 5（表 10・11） | 同じ | ✓ |
| アダプタ | テンソル化アダプタ、ボトルネック 64（3・4 節、図 1） | 各ブロックの注意と MLP の後ろの 2 か所、ボトルネック 64、ReLU | △ 位置と非線形関数は図からの解釈。パラメータ数が一致することで裏づけ |
| 学習するもの | FedTT：アダプタと分類ヘッド。FedTT+：{G_1, G_r, G_J} と分類ヘッド（4 節、Algorithm 2） | 同じ。FedTT はアダプタのバイアスも学習し、FedTT+ は学習しない | △ バイアスの扱いはパラメータ数（0.06M・0.02M）から逆算 |
| FedTT+ のコアの選び方 | 本文は r ∈ {2, …, J−1}、擬似コードは mod(t, J−2) = r−1 | 本文どおり（r = 2, 3, 4, 2, …） | △ 本文と擬似コードが食い違う |
| 初期化・系列長 | 書かれていない | W の標準偏差 0.01（アダプタ）・0.02（分類器）、系列長 128 | ? |
| 訓練データ | 3,301 件（表 8） | GLUE の訓練データ 3,668 件から 3,301 件を無作為に選ぶ（シード 42）。残り 367 件は held-out | △ 件数は同じ。どの 3,301 件かは不明 |
| 評価データ | 「best validation results」。表 8 の Dev は 367 件 | GLUE の検証データ 408 件（主）。held-out の 367 件でも毎ラウンド評価 | **要検討** |
| 指標・報告のしかた | F1、ラウンド中の最高値（付録 A.1・A.3） | 同じ（グリッドの最大） | ✓ |
| シード | 書かれていない | 0 のみ | ? |
| 計算機・精度 | NVIDIA A6000・V100、精度は書かれていない | A100（Colab、ハイメモリ）、fp32（TF32） | 違う（結果への影響は小さいはず） |

**要検討の点**
1. **評価データ**：論文の表 8 の MRPC の Dev（367 件）は、GLUE の訓練データから訓練用の 3,301 件を除いた件数で、GLUE の検証データ（408 件）ではありません。論文の「best validation」がどちらで測られたかは書かれていません。本再現は 408 件で比べました。ラウンドごとのログを追加すれば、367 件の held-out での最高値とも比べられます。
2. **ローカル学習の量**：「1 エポック」か「1 回の更新」かで、学習量が 21〜42 倍違います（1 クライアント 660 件、バッチ 32 で 21 ステップ、16 で 42 ステップ）。本再現は 1 エポックです（理由は `説明資料.md` §3）。
3. **シード**：論文との差がシード間のばらつきに収まるかは、まだ測っていません。

## このフォルダの中身
| 場所 | 中身 |
|---|---|
| `README.md` | この説明 |
| [`説明資料.md`](説明資料.md) | 詳しい説明：手法のしくみ、論文とコードの対応、論文に書かれていない設定、実験環境、結果の考察、パラメータ数の照合、論文の中の不整合 9 件、回し方 |
| [`method/fedtt/`](method/fedtt/) | **論文で提案されている手法のコード（FedTT・FedTT+）**：TT 線形層（`tt.py`）、テンソル化アダプタ・分類器（`model.py`）、Algorithm 1・2 の 1 ラウンドの処理（`algorithm.py`） |
| [`reproduction/FedTT_MRPC_colab.ipynb`](reproduction/FedTT_MRPC_colab.ipynb) | **再現コード：表 1 の MRPC を回した Google Colab のノートブック**（コードを中に含むので、開いて「すべてのセルを実行」するだけ） |
| [`reproduction/FedTT_repro_colab.ipynb`](reproduction/FedTT_repro_colab.ipynb) | 表 1, 2, 3, 7, 13 の全ステップを回すノートブック（まだ回していない） |
| [`reproduction/fedtt_repro/`](reproduction/fedtt_repro/) | 再現実験の枠組み：GLUE データと分割、実行・評価・記録、論文の表ごとの設定、パラメータ数の照合、集計 |
| [`reproduction/tests/`](reproduction/tests/) | CPU テスト 14 本 |
| [`reproduction/tools/make_notebook.py`](reproduction/tools/make_notebook.py) | ノートブックを作るスクリプト |
| [`results/`](results/) | 結果（`t1_mrpc/summary.csv`）とパラメータ数の照合（`param_check.md`） |
| `requirements.txt` | ローカルで回すときに必要なパッケージ |

## 回し方
**Colab（精度の再現）**
1. 上の「Open in Colab」から `FedTT_MRPC_colab.ipynb` を開きます。
2. 「ランタイム → ランタイムのタイプを変更」で GPU を選びます（本再現は A100・ハイメモリ）。
3. 「ランタイム → すべてのセルを実行」を押します。結果は Google ドライブの `MyDrive/FedTT_repro/results` に保存され、最後のセルに論文との比較表が出ます。

**ローカル（テスト・パラメータ数の照合）**：`reproduction/` フォルダで
```
pip install -r ../requirements.txt
python tests/test_core.py
python -m fedtt_repro.count --out ../results/param_check.md
```

## 残っている確認
- シードを増やす（FedTT 1e-3・32 と FedTT+ 5e-3・16 をシード 1, 2 で）。
- ラウンドごとのログと学習曲線を `results/` に追加し、held-out（367 件）での最高値とも比べる。
- 表 7 の SST-2・QNLI（設定がすべて書かれていて、検証データも大きい）。
- 表 2・3・13（クライアント数・不均一性・RoBERTa）。RoBERTa-large は、テンソルの形が論文に書かれておらず、パラメータ数も一部合わない（`説明資料.md` §6）。

## 引用（元の論文）
```bibtex
@inproceedings{ghiasvand-etal-2025-communication,
  title     = {Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models},
  author    = {Ghiasvand, Sajjad and Yang, Yifan and Xue, Zhiyu and Alizadeh, Mahnoosh and Zhang, Zheng and Pedarsani, Ramtin},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {24192--24207},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.findings-acl.1241},
  url       = {https://aclanthology.org/2025.findings-acl.1241/}
}
```
