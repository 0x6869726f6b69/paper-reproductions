"""Writes two self-contained Colab notebooks into the reproduction folder:

  FedTT_MRPC_colab.ipynb   Table 1, MRPC only (the lightest experiment of the paper)
  FedTT_repro_colab.ipynb  all suites, cheapest first

Both packages (../method/fedtt = the method, fedtt_repro = the experiments) are embedded as %%writefile cells
(nothing to upload); results go to Google Drive.
Re-run this script after changing either package so the notebooks carry the current code.
"""
import hashlib
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = os.path.join(os.path.dirname(ROOT), "method")
# (folder, package, file): the method first, then the experiment code
FILES = [(os.path.join(METHOD, "fedtt"), "fedtt", f) for f in ("__init__.py", "tt.py", "model.py", "algorithm.py")] + \
        [(os.path.join(ROOT, "fedtt_repro"), "fedtt_repro", f) for f in
         ("__init__.py", "data.py", "fl.py", "suites.py", "pipeline.py", "count.py", "report.py", "paper.py")]
CODE = "/content/fedtt_code"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def sources():
    out = {}
    for d, pkg, f in FILES:
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            out[f"{pkg}/{f}"] = fh.read()
    h = hashlib.sha1("".join(out[f"{pkg}/{f}"] for _, pkg, f in FILES).encode()).hexdigest()[:10]
    return out, h


def code_cells(src):
    cells = [md("## コード（変更不要）\n論文の擬似コードから書いた実装です。リポジトリ paper-reproductions の "
                "`papers/2025-findings-acl-fedtt/` にあるものと同じで、`fedtt/` が論文の手法（FedTT・FedTT+）、"
                "`fedtt_repro/` が再現実験の枠組みです。下のセルが `/content/fedtt_code/` に書き出します。")]
    for _, pkg, f in FILES:            # keep the file byte-identical (no stripping)
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": (f"%%writefile {CODE}/{pkg}/{f}\n" + src[f"{pkg}/{f}"]).splitlines(True)})
    return cells


def setup_cell(per_run_gb, default_hours, version):
    return f"""
import os, sys, time, subprocess, torch
from google.colab import drive

USE_DRIVE = True                                     # False: 結果を /content に置く（切断すると消える）
OUT   = '/content/drive/MyDrive/FedTT_repro/results' if USE_DRIVE else '/content/results'
HOURS = {default_hours}                                         # このセッションで使う時間（プランの上限より少し短く）
CODE  = '{CODE}'

if USE_DRIVE:
    drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{{CODE}}/fedtt', exist_ok=True)
os.makedirs(f'{{CODE}}/fedtt_repro', exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'          # モデル・データのダウンロード先（ローカルディスク）
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
assert torch.cuda.is_available(), 'GPU がありません。「ランタイム → ランタイムのタイプを変更」で GPU（A100 / L4 など）を選んでください'
gpu = torch.cuda.get_device_name(0)
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
bf16 = torch.cuda.is_bf16_supported()
cpus = os.cpu_count() or 2
PARALLEL = max(1, min(int((gpu_gb - 4) // {per_run_gb}), max(1, cpus // 2), 6))   # 1 つの GPU で同時に流すラン数
T_END = time.time() + 3600 * HOURS
print(f'GPU: {{gpu}}（{{gpu_gb:.0f}} GB）, CPU {{cpus}} 個 -> 同時実行 {{PARALLEL}} 本')
print('数値の精度:', 'bf16 が使える GPU（ただしモデルが bf16 で動かないときは、ランごとに自動で fp32/TF32 に切り替え）' if bf16
      else 'fp32（bf16 非対応の GPU。数倍遅くなります）')
print('結果の保存先:', OUT, '| コードの版:', '{version}')
subprocess.run('pip -q install "datasets>=2.18"', shell=True, check=False)
# Colab に入っている古い torchao（0.10）があると、新しい peft が LoRA 層を作るときに止まる（パラメータ数の照合の LoRA の行）。
# FedTT / FedTT+ は torchao も peft も使わないので、外すだけでよい。
subprocess.run('pip -q uninstall -y torchao', shell=True, check=False)
"""


HELPERS = """
import json, glob
os.chdir(CODE)
sys.path.insert(0, CODE)

def sh(cmd):
    \"\"\"コマンドを実行し、出力をこのセルに流す（ダウンロードの進捗表示などは省く）。\"\"\"
    noise = ('it/s]', 's/it]', 'Loading weights', 'LOAD REPORT', 'UNEXPECTED', 'MISSING', 'Notes:', 'can be ignored',
             'newly initialized', '---+---', 'Key ', 'HF_TOKEN', 'Redirects are currently')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding='utf-8', errors='replace', env=env)
    for line in p.stdout:
        if not any(s in line for s in noise):
            print(line, end='', flush=True)
    return p.wait()

def step(name, args, max_par=6):
    left = (T_END - time.time()) / 3600
    if left < 0.1:
        print(f'時間切れのため「{name}」は始めません（もう一度「すべてのセルを実行」で続きから）'); return False
    print(f'\\n===== {name}（残り {left:.1f} 時間） =====', flush=True)
    sh(f'{sys.executable} -m fedtt_repro.pipeline {args} --out "{OUT}" --parallel {min(PARALLEL, max_par)} --hours {left:.2f}')
    return True

r = sh(f'{sys.executable} -c "import fedtt_repro.fl, fedtt_repro.pipeline; print(\\'code ok\\')"')
assert r == 0, 'コードの読み込みに失敗しました（上のエラーを見てください）'
"""

MRPC_INTRO = """
# FedTT / FedTT+ の再現：表 1 の MRPC（Colab GPU 版）

対象論文：Ghiasvand et al., "Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models", Findings of ACL 2025。
論文の中で一番軽い実験（表 1 の MRPC）で、FedTT と FedTT+ の精度が論文どおりになるかを確かめます。

**使い方**：「ランタイム → ランタイムのタイプを変更」で GPU（A100 か L4 がおすすめ）を選び、「ランタイム → すべてのセルを実行」を押すだけです。コードはこのノートブックの中に入っているので、アップロードは不要です。途中で切れても、もう一度「すべてのセルを実行」を押せば、終わったランは飛ばし、途中のランは 10 ラウンドごとのチェックポイントから続けます（結果は Google ドライブの `MyDrive/FedTT_repro/results` に保存）。

**設定（論文の表 1・付録 A.3）**：DeBERTa-base、5 クライアント、IID、1 ラウンドに 1 ローカルエポック、100 ラウンド、AdamW、テンソルランク 5。評価は MRPC の F1（GLUE の検証データ 408 件）で、100 ラウンド中の最高値を論文の値と比べます。

| ステップ | 中身 | ラン数 | 目安（A100） |
|---|---|---|---|
| 0 | パラメータ数・通信量の照合（学習なし） | – | 1〜2 分 |
| A | 学習率 1e-3・バッチ 32（表 7 で作者が使った設定）の FedTT と FedTT+ | 2 | 約 30 分（実測） |
| B | 論文と同じグリッド（学習率 4 通り × バッチ 2 通り）の残り | 14 | 約 3 時間（見込み） |

L4 ではこの 2 倍程度、T4 ではさらに数倍かかる見込みです。ステップ A だけで済ませたいときは、設定セルの `RUN_FULL_GRID = False` にしてください。

**数値の精度**：DeBERTa は、transformers の DeBERTa が bf16 で動かない不具合（`value cannot be converted to type BFloat16 without overflow`）があるため、各ランの最初に自動で判定して 32 ビット（A100・L4 では TF32 の行列積）で回します。精度を下げる変更ではないので、結果を悪くすることはありません（速さは bf16 より遅くなります）。

**論文の値**：FedTT 92.68、FedTT+ 92.60（F1、学習率 × バッチの 8 通りのうちの最高値）。

**本再現の結果と環境**：Google Colab の A100 GPU・ハイメモリのランタイムで回し（2026-11-24）、FedTT 92.50、FedTT+ 93.24 でした。16 本の値と条件の比較は、リポジトリ paper-reproductions の `papers/2025-findings-acl-fedtt/`（README.md・説明資料.md・results/）にあります。
"""

MRPC_SETTINGS_EXTRA = """
RUN_FULL_GRID = True     # True: ステップ B（論文と同じ 8 通りのグリッド）も回す
"""

MRPC_RUN = """
# ステップ 0：パラメータ数・通信量の照合（学習なし、CPU で 1〜2 分）
sh(f'{sys.executable} -m fedtt_repro.count --out "{OUT}/report/param_check.md"')
# ステップ A：学習率 1e-3・バッチ 32 の FedTT と FedTT+（2 本同時）
step('A：lr 1e-3・bs 32 の FedTT / FedTT+', '--suite t1 --tasks mrpc --lrs 1e-3 --bss 32', 2)
# ステップ B：論文と同じグリッドの残り（終わったランは飛ばす）
if RUN_FULL_GRID:
    step('B：学習率 4 × バッチ 2 のグリッド', '--suite t1 --tasks mrpc')
"""

MRPC_SUMMARY = """
import pandas as pd
from IPython.display import display, Markdown, Image
PAPER = {'fedtt': 92.68, 'fedttplus': 92.60}
rows = []
for p in glob.glob(f'{OUT}/runs/t1/mrpc/*/result.json'):
    r = json.load(open(p, encoding='utf-8')); c, s = r['cfg'], r['summary']
    rows.append(dict(method=c['method'], lr=c['lr'], bs=c['bs'], best_F1=round(s['best_val'], 2),
                     best_round=s['best_val_round'], F1_heldout_selected=round(s['val_at_best_heldout'], 2),
                     final_F1=round(s['final_val'], 2), uplink_KB=round(s['up_kb_mean'], 1),
                     sec_per_round=round(s['train_sec'] / s['rounds'], 1), gpu=r['env'].get('gpu', '')))
if not rows:
    print('まだ終わったランがありません。上の「実行」セルの出力に「errors in ...」が出ていないか、'
          'いちばん下の「うまくいかないとき」のセルで確認してください。')
else:
    df = pd.DataFrame(rows).sort_values(['method', 'lr', 'bs']).reset_index(drop=True)
    display(df)
    lines = ['| 手法 | 終わった設定 | 最高 F1（この実験） | 論文 | 差 | そのときの学習率・バッチ |', '|---|---|---|---|---|---|']
    for m in ('fedtt', 'fedttplus'):
        d = df[df.method == m]
        if len(d):
            b = d.loc[d.best_F1.idxmax()]
            lines.append(f'| {m} | {len(d)}/8 | {b.best_F1:.2f}（{b.best_round} ラウンド目） | {PAPER[m]} | '
                         f'{b.best_F1 - PAPER[m]:+.2f} | {b.lr:g}・{b.bs} |')
    display(Markdown('\\n'.join(lines)))
"""

REPORT = """
sh(f'{sys.executable} -m fedtt_repro.report --out "{OUT}"')
from IPython.display import display, Markdown, Image
if os.path.exists(f'{OUT}/report/curves.png'):
    display(Image(f'{OUT}/report/curves.png'))
print('論文との比較表:', f'{OUT}/report/report.md', '| パラメータ数の照合:', f'{OUT}/report/param_check.md')
"""

DIAG_MD = """
## うまくいかないとき
ランが終わらない・エラーが出るときは、このセルを実行して、出てきた表示をそのまま共有してください。
各ワーカーのログの最後と、失敗したランのエラー（`error.txt`）、各ランの最新のラウンドを表示します。
"""

DIAG = """
import glob, json, os
for f in sorted(glob.glob(f'{OUT}/logs/*.log')):
    print('=====', f)
    print(open(f, encoding='utf-8', errors='replace').read()[-2500:])
errs = glob.glob(f'{OUT}/runs/**/error.txt', recursive=True)
print(f'\\n失敗したランの記録: {len(errs)} 件')
for f in errs[:3]:
    print('===== ERROR FILE', f)
    print(open(f, encoding='utf-8', errors='replace').read()[-2500:])
for f in sorted(glob.glob(f'{OUT}/runs/**/log.jsonl', recursive=True)):
    rows = [json.loads(x) for x in open(f, encoding='utf-8') if x.strip()]
    if rows:
        r = rows[-1]
        print(os.path.basename(os.path.dirname(f)), '| round', r['round'], '| val', round(r.get('val', float('nan')), 2))
"""

FULL_INTRO = """
# FedTT / FedTT+ 再現実験：全ステップ（Colab GPU 版）

対象論文：Ghiasvand et al., Findings of ACL 2025。**「ランタイム → すべてのセルを実行」を押すだけ**で、安いものから順に進みます。コードはこのノートブックの中にあり、アップロードは不要です。途中で止まっても、もう一度「すべてのセルを実行」で続きから進みます。表 1 の MRPC だけなら `FedTT_MRPC_colab.ipynb` のほうが軽いです（同じ結果フォルダを使うので、どちらで回したランも共有されます）。

| 順 | ステップ | 論文 | ラン数 | 目安（A100、見込み） |
|---|---|---|---|---|
| 0 | パラメータ数・通信量の照合（学習なし） | 表 1, 2, 7, 12, 14, 15 | – | 1〜2 分 |
| 1 | スモークテスト | – | 5 | 5〜10 分 |
| 2 | 表 1：MRPC（学習率 4 × バッチ 2 の全グリッド、100 ラウンド） | 表 1 | 16 | 1〜2 時間 |
| 3 | 表 7：SST-2・QNLI（ランク 2/5/10 と FedTT+、20 ラウンド） | 表 7（と表 1） | 8 | 3〜5 時間 |
| 4 | 表 3：3 クライアント（iid） | 表 3 | 8 | 1〜2 時間 |
| 5 | 表 7：QQP・MNLI | 表 7（と表 1） | 8 | 10〜17 時間 |
| 6 | 表 3：20・50 クライアント（mild・severe） | 表 3, 図 2, 表 15 | 16 | 25〜35 時間 |
| 7 | 表 2：RoBERTa-base（FedPETuning の設定、学習率は 2 点に絞る） | 表 2 | 10 | 5〜10 時間 |
| 8 | 表 13：3 クライアント・ラベル偏り（1000 ラウンド） | 表 13 | 12 | 40 時間以上 |

**時間は実測ではなく見込みです**（学習するサンプル数と、1 本あたり 300 サンプル/秒・4〜6 本同時という仮定から）。ステップ 1 のあと、出力の「s/round」で実際の速さがわかります。`STEPS` から不要な行を消せば、そのステップを飛ばせます。

**表 1 の大きな 4 タスク**：論文どおりの全グリッド（8 通り × 100 ラウンド、1 ラウンド 1 エポック）は約 15 億サンプルで現実的でないため、設定がすべて書かれている表 7（学習率 1e-3・バッチ 32・20 ラウンド。ランク 5 は表 1 の FedTT と同じ手法）で比べます。
"""

FULL_RUN = """
# (名前, pipeline の引数, 同時に流す本数の上限)。不要な行を消せば飛ばせます。
STEPS = [
    ('1 スモーク',           '--suite smoke', 6),
    ('2 表1 MRPC',           '--suite t1 --tasks mrpc', 6),
    ('3 表7 SST-2/QNLI',     '--suite t7 --tasks sst2,qnli', 6),
    ('4 表3 iid',            '--suite t3 --levels iid', 6),
    ('5 表7 QQP/MNLI',       '--suite t7 --tasks qqp,mnli', 6),
    ('6 表3 mild/severe',    '--suite t3 --levels mild,severe', 6),
    ('7 表2',                '--suite t2 --lrs 5e-3,1e-3 --bss 32', 6),
    ('8 表13',               '--suite t13 --micro_bs 50', 3),      # バッチ 200 を 50 ずつに分けて勾配を足す（結果は同じ）
]
sh(f'{sys.executable} -m fedtt_repro.count --out "{OUT}/report/param_check.md"')
for name, args, max_par in STEPS:
    if not step(name, args, max_par):
        break
"""


def notebook(cells):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    src, version = sources()
    mrpc = [md(MRPC_INTRO), md("## 設定とセットアップ\n必要なら `USE_DRIVE`・`HOURS`・`RUN_FULL_GRID` だけ変えてください。"),
            code(setup_cell(5, 6, version) + MRPC_SETTINGS_EXTRA)] + code_cells(src) + \
           [md("## 準備"), code(HELPERS), md("## 実行"), code(MRPC_RUN),
            md("## 結果（論文との比較）"), code(MRPC_SUMMARY), code(REPORT), md(DIAG_MD), code(DIAG)]
    full = [md(FULL_INTRO), md("## 設定とセットアップ\n必要なら `USE_DRIVE`・`HOURS` だけ変えてください。"),
            code(setup_cell(6, 23, version))] + code_cells(src) + \
           [md("## 準備"), code(HELPERS), md("## 実行（パラメータ数の照合 → 各ステップ）"), code(FULL_RUN),
            md("## 集計（論文との比較）"), code(REPORT), md(DIAG_MD), code(DIAG)]
    for name, nb in (("FedTT_MRPC_colab.ipynb", mrpc), ("FedTT_repro_colab.ipynb", full)):
        path = os.path.join(ROOT, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook(nb), f, ensure_ascii=False, indent=1)
        print("wrote", path, "| code version", version)


if __name__ == "__main__":
    main()
