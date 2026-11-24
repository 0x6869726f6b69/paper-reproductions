# paper-reproductions

> **English summary.** Independent reproductions of published machine-learning papers. Each case lives in its own folder under `papers/` and contains the paper's proposed method re-implemented from the paper (`method/`), the code that runs the paper's experiments (`reproduction/`, e.g. a Google Colab notebook), the results with the environment they were obtained in (`results/`), a side-by-side comparison of our conditions with the paper's, and a verdict. Documentation is in Japanese with an English summary at the top of each page.

論文の再現実験をまとめるリポジトリです。1 本の論文につき 1 つのフォルダを `papers/` の下に作り、論文で提案された手法を論文の記述から実装し、論文と同じ手順の実験を回して、結果が論文どおりになるかを確かめます。

## 事例一覧

| 論文 | 会議・年 | 公式コード | 再現したもの | 結果（本再現 / 論文） | 判定 | 環境 |
|---|---|---|---|---|---|---|
| [FedTT / FedTT+](papers/2025-findings-acl-fedtt/)：Ghiasvand et al., *Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models* | Findings of ACL 2025 | なし | 表 1 の MRPC（DeBERTa-base、5 クライアント IID、学習率 × バッチの 8 通り） | FedTT 92.50 / 92.68、FedTT+ 93.24 / 92.60（F1） | ✅ 再現（シード 1 つ） | Google Colab、A100・ハイメモリ |

## 判定の基準

| 判定 | 意味 |
|---|---|
| ✅ 再現 | 論文と同じ手順・指標で、論文の値との差がシード間のばらつきの範囲（ばらつきを測っていないときは 1 点以内とし、「シード 1 つ」と書く） |
| 🟡 一部再現 | 一部の手法・設定だけ合う、または論文と違う条件でだけ合う |
| ❌ 再現せず | 手順を合わせても、ばらつきを超える差がある |
| ⏳ 実施中 | 結果がまだない |

判定は、そのフォルダで「再現したもの」に書いた範囲についてのものです。論文全体が再現できたという意味ではありません。

## 各事例のフォルダの中身

```
papers/<年>-<会議>-<短い名前>/
├── README.md        説明：論文、結果と判定、実験環境、論文との条件の比較、ファイルの案内、回し方
├── 説明資料.md       詳しい説明：手法のしくみ、論文とコードの対応、論文に書かれていない設定、結果の考察、論文の中の不整合
├── method/          論文で提案されている手法のコード
├── reproduction/    再現コード（Colab のノートブック、実験の枠組み、テスト）
├── results/         結果（数値・ログ）と、それを得た環境
└── requirements.txt
```

- 各事例のコードは独立しています（ほかの事例のコードに依存しません）。
- 論文に書かれていない設定は、こちらで決めた値と理由を `説明資料.md` に必ず書きます。
- 結果には、計算環境（GPU・ランタイム・精度）と実行日を必ず書きます。

## 事例の追加のしかた
1. [`template/README.md`](template/README.md) を `papers/<年>-<会議>-<短い名前>/README.md` にコピーします（例：`papers/2025-findings-acl-fedtt/`）。
2. 上の構成でコードと結果を置き、`README.md` と `説明資料.md` を書きます。
3. このページの「事例一覧」に 1 行足します。

## 注意
- 論文の著者とは関係のない、独立した再現です。結果や解釈の誤りは、このリポジトリの責任です。
- 論文の数値は、比較のために出典を示して引用しています。論文の PDF や図は含めません。
- コードのライセンスは MIT です（[LICENSE](LICENSE)）。論文の著作権は著者と出版社にあります。
