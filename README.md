# Synthetic Data Local Generator (sdlg)

ローカル/Colabで動く **合成テキスト生成 → 品質チェック → 保存** の CLI ツール（MVP）。

## 要件
- Python 3.10+
- PyTorch（環境に応じてインストール）

## インストール（開発用）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
PyTorch の導入例（CPUテスト）

powershell
コードをコピーする
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
使い方（CLI）
1行=1プロンプトのテキストファイルを用意します：

text
コードをコピーする
顧客からの問い合わせ: 返品は可能ですか？ 店舗とオンラインの違いも教えてください。
顧客からの問い合わせ: 配送はどれくらいかかりますか？ 地域別の目安も含めてください。
生成→品質レポ→保存を1コマンドで実行：

powershell
コードをコピーする
sdlg run --prompts .\prompts.txt -n 1 --model distilgpt2 --lang ja --max-new-tokens 60 --out outputs\synthetic.csv --report outputs\report.json
指示追従モデル（Colab/GPU推奨）
bash
コードをコピーする
sdlg run --prompts prompts.txt -n 2 --model "Qwen/Qwen2.5-1.5B-Instruct" --chat --lang ja --max-new-tokens 120
出力
CSV: outputs/synthetic.csv（prompt, response 形式）

JSON: outputs/report.json（品質レポ v1 の集計）

機能（現状）
生成：Transformers によるローカル推論（API不要）／free-form と chat モード

品質レポ v1：言語一致、長さ統計、5-gram重複率、簡易毒性、簡易PII検出

プロジェクト構成
bash
コードをコピーする
src/sdlg/
  __init__.py
  cli.py        # sdlg コマンド
  generator.py  # 生成（free-form / chat）
  quality.py    # 品質レポ v1
今後の予定
レシピ保存・再現（seed / params / model）

並列実行・スケジューラ

SFT/RAGフォーマット出力、しきい値のCLI化

Streamlit UI
