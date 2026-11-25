"""Writes the self-contained Colab notebook reproduction/MetaTT_min_colab.ipynb.

The method (../method/metatt), the experiment code (metatt_repro) and the CPU tests are embedded as %%writefile
cells, so nothing has to be uploaded; results go to Google Drive. Re-run this script after changing any of them.
"""
import hashlib
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = os.path.join(os.path.dirname(ROOT), "method")
FILES = [(os.path.join(METHOD, "metatt"), "metatt", f) for f in ("__init__.py", "adapter.py")] + \
        [(os.path.join(ROOT, "metatt_repro"), "metatt_repro", f) for f in
         ("__init__.py", "paper.py", "glue.py", "train.py", "pipeline.py", "report.py", "count.py")] + \
        [(os.path.join(ROOT, "tests"), "tests", "test_core.py")]
CODE = "/content/metatt_code"
NAME = "MetaTT_min_colab.ipynb"


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


INTRO = """
# MetaTT の再現：表 2 の最小実験（RoBERTa-base × MetaTT-4D r=8 × MRPC・RTE）

対象論文：Lopez-Piqueres et al., "MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning", TMLR 2026（arXiv:2506.09105）。
論文の中でいちばん小さい実験（表 2 の 1 マス）で、MetaTT-4D の正解率が論文どおりになるかを確かめます。比較手法（LoRA など）は回しません。

**使い方**：「ランタイム → ランタイムのタイプを変更」で **A100 GPU・ハイメモリ** を選び、「ランタイム → すべてのセルを実行」を押すだけです。コードはこのノートブックの中にあるので、アップロードは不要です。結果は Google ドライブの `MyDrive/MetaTT_repro/results` に保存され、途中で切れても、もう一度「すべてのセルを実行」を押せば終わったランは飛ばします（途中のランは最初からやり直し）。

**設定（論文の値）**

| 項目 | 値 | 論文の出どころ（v2 / v3） |
|---|---|---|
| モデル | RoBERTa-base（`FacebookAI/roberta-base`） | 表 2 |
| アダプタ | MetaTT-4D、ランク 8、Q と V に適用、学習するパラメータ 13,184（論文 13k） | 表 2、付録 D.2 / G.2 |
| 初期化 | G1 はゼロ、G2・G3・G4 は単位行列 | 付録 D.4 / G.4 |
| 学習 | 分類ヘッドは凍結、20 エポック、系列長 256、weight decay 0 | 3.1 節、付録 D.4 / G.4 |
| シード | 33305628、2025、42 | 付録 D.1 / G.1 |
| MRPC | α = 0.5、学習率 1e-3、バッチ 8 | 表 6 / 表 14 |
| RTE | α = 4、学習率 1e-3、バッチ 16 | 表 6 / 表 14 |
| 目標値 | MRPC 87.6 ± 0.2、RTE 72.9 ± 0.5（検証データの正解率、± は標準誤差） | 表 2 |

**論文に書かれていない・食い違っている設定（こちらで決めたこと）**
- warmup ratio：論文の GLUE の段落は 0.6、ほかの実験はすべて 0.06。**両方**回します（0.06 → 0.6 の順）。
- 表 6 / 表 14 の該当行はランク「4」と印刷されていますが、13k のパラメータ数はランク 8 でしか出ないため、ランク 8 の行として使います。
- 最適化は HF Trainer の既定値（AdamW、線形スケジュール、勾配クリップ 1.0）。報告値は「20 エポック中の最良」と「最終エポック」の両方を出します。
- 数値の精度は fp32（TF32 も使わない）。論文には書かれていません。

**ラン数**：2 タスク × 3 シード × warmup 2 通り = 12 本（1 本は MRPC 9,180 ステップ、RTE 3,120 ステップ）。同時に数本ずつ流します。時間はまだ測っていないので、最初のランが終わったときの「[x.x min]」で確かめてください。
"""

SETUP = """
import os, sys, time, subprocess, torch
from google.colab import drive

USE_DRIVE = True                                   # False: 結果を /content に置く（切断すると消える）
OUT       = '/content/drive/MyDrive/MetaTT_repro/results' if USE_DRIVE else '/content/results'
TASKS     = 'mrpc,rte'                             # 'mrpc' だけ、'rte' だけにもできる
WARMUPS   = '0.06,0.6'                             # '0.06' だけにすると 6 本で済む
PRECISION = 'fp32'                                 # 'tf32' / 'bf16' にすると速いが、論文の条件からは外れる（別フォルダに保存）
HOURS     = 10                                     # この時間を過ぎたら新しいランを始めない
CODE      = '{CODE}'

if USE_DRIVE:
    drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
for d in ('metatt', 'metatt_repro', 'tests'):
    os.makedirs(f'{{CODE}}/{{d}}', exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
assert torch.cuda.is_available(), 'GPU がありません。「ランタイム → ランタイムのタイプを変更」で A100 GPU を選んでください'
gpu = torch.cuda.get_device_name(0)
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
cpus = os.cpu_count() or 2
PARALLEL = max(1, min(int((gpu_gb - 4) // 4), max(1, cpus // 2), 6))   # 1 つの GPU で同時に流すラン数
print(f'GPU: {{gpu}}（{{gpu_gb:.0f}} GB）, CPU {{cpus}} 個 -> 同時実行 {{PARALLEL}} 本 | 精度 {{PRECISION}}')
print('結果の保存先:', OUT, '| コードの版:', '{version}')
subprocess.run('pip -q install "datasets>=2.18"', shell=True, check=False)
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

# 実装のテスト（CPU、1 分ほど）：パラメータ数、初期状態で ΔW = 0、分類ヘッドが凍結、論文の擬似コードと同じ出力、など
r = sh(f'{sys.executable} tests/test_core.py')
assert r == 0, 'テストが通りませんでした（上のエラーを見てください）'
"""

RUN = """
# ステップ 0：表 2 の MetaTT-4D のパラメータ数の照合（学習なし）
sh(f'{sys.executable} -m metatt_repro.count --out "{OUT}/report/param_check.md"')
# ステップ 1：12 本のラン（終わったものは飛ばす）
t0 = time.time()
sh(f'{sys.executable} -m metatt_repro.pipeline --out "{OUT}" --tasks {TASKS} --warmups {WARMUPS} '
   f'--precision {PRECISION} --parallel {PARALLEL} --hours {HOURS}')
print(f'経過 {(time.time() - t0) / 60:.1f} 分')
"""

SUMMARY = """
from IPython.display import display, Markdown, Image
r = sh(f'{sys.executable} -m metatt_repro.report --out "{OUT}" > /dev/null')
rep = f'{OUT}/report/report.md'
if os.path.exists(rep):
    display(Markdown(open(rep, encoding='utf-8').read()))
if os.path.exists(f'{OUT}/report/curves.png'):
    display(Image(f'{OUT}/report/curves.png'))
print('ランごとの値:', f'{OUT}/report/summary.csv', '| パラメータ数の照合:', f'{OUT}/report/param_check.md')
"""

DIAG_MD = """
## うまくいかないとき
ランが失敗する・終わらないときは、このセルを実行して、出てきた表示をそのまま共有してください。
各ランのログの最後と、失敗したランのエラー（`error.txt`）を表示します。
"""

DIAG = """
import glob, os
for f in sorted(glob.glob(f'{OUT}/logs/*.log')):
    print('=====', f)
    print(open(f, encoding='utf-8', errors='replace').read()[-2000:])
errs = glob.glob(f'{OUT}/runs/**/error.txt', recursive=True)
print(f'\\n失敗したランの記録: {len(errs)} 件')
for f in errs[:3]:
    print('===== ERROR FILE', f)
    print(open(f, encoding='utf-8', errors='replace').read()[-2500:])
"""


def code_cells(src):
    cells = [md("## コード（変更不要）\n論文の擬似コード（付録 E / H）と本文の式から書いた実装です。リポジトリ "
                "paper-reproductions の `papers/2026-tmlr-metatt/` にあるものと同じで、`metatt/` が論文の手法"
                "（MetaTT-4D）、`metatt_repro/` が再現実験、`tests/` が CPU のテストです。下のセルが "
                f"`{CODE}/` に書き出します。")]
    for _, pkg, f in FILES:            # keep the files byte-identical
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": (f"%%writefile {CODE}/{pkg}/{f}\n" + src[f"{pkg}/{f}"]).splitlines(True)})
    return cells


def notebook(cells):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    src, version = sources()
    cells = [md(INTRO), md("## 設定とセットアップ\n必要なら `USE_DRIVE`・`TASKS`・`WARMUPS`・`HOURS` だけ変えてください。"),
             code(SETUP.format(CODE=CODE, version=version))] + code_cells(src) + \
            [md("## 準備（実装のテスト）"), code(HELPERS), md("## 実行"), code(RUN),
             md("## 結果（論文の表 2 との比較）"), code(SUMMARY), md(DIAG_MD), code(DIAG)]
    path = os.path.join(ROOT, NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, ensure_ascii=False, indent=1)
    print("wrote", path, "| code version", version)


if __name__ == "__main__":
    main()
