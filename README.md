# Synthetic Data Local Generator (sdlg)

ローカル/Colabで動く **合成テキスト生成 → 品質チェック → 保存** のCLIツール（MVP）。

## インストール（開発用）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
PyTorch は環境に合わせて入れてください。CPUの簡易テストなら：

pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

使い方（CLI）

1行=1プロンプトのテキストファイルを用意します：

顧客からの問い合わせ: 返品は可能ですか？ 店舗とオンラインの違いも教えてください。
顧客からの問い合わせ: 配送はどれくらいかかりますか？ 地域別の目安も含めてください。


生成→品質レポ→保存を1コマンドで：

sdlg run --prompts .\prompts.txt -n 1 --model distilgpt2 --lang ja --max-new-tokens 60 --out outputs\synthetic.csv --report outputs\report.json

指示追従モデル（Colab/GPU推奨）
sdlg run --prompts prompts.txt -n 2 --model "Qwen/Qwen2.5-1.5B-Instruct" --chat --lang ja --max-new-tokens 120

機能（現状）

生成バックエンド：Transformers（完全ローカル推論、API不要）

品質レポ v1：言語一致、長さ統計、5-gram重複率、簡易毒性、簡易PII検出

出力：CSV（データセット）、JSON（品質レポ）

開発メモ

パッケージ構成（srcレイアウト）

src/sdlg/
  __init__.py
  cli.py        # sdlg コマンド
  generator.py  # 生成（free-form / chat）
  quality.py    # 品質レポ v1


今後：レシピ保存、並列実行、SFT/RAGフォーマット出力、しきい値のCLI化、Streamlit UI など
