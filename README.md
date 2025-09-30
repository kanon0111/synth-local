# Synthetic Data Local Generator (sdlg)

ローカル / Colab で動く **合成テキスト生成 → 品質チェック → 保存** の CLI ツール（MVP）。

---

## 主な機能
- 生成バックエンド：Hugging Face Transformers（完全ローカル推論 / 外部API不要）
- モード：**free-form**（続き書き） / **chat**（指示追従モデル向け）
- 品質レポ v1：**言語一致**・**長さ統計**・**5-gram重複率**・**簡易毒性**・**簡易PII検出**
- 出力：**CSV**（`prompt, response`）＋ **JSON**（品質集計）

---

## 必要要件
- Python **3.10+**
- PyTorch（CPU でも可。GPU を使う環境では CUDA 版を各自の環境に合わせて導入）

---

## インストール（開発用）
```powershell
# 仮想環境の作成と有効化（Windows PowerShell の例）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# sdlg を開発インストール
pip install --upgrade pip
pip install -e .
```

### PyTorch の導入例
- **CPU テスト用（簡易）**
```powershell
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```
- **GPU（CUDA）**  
  お使いの CUDA / OS / Python に応じたコマンドでインストールしてください（PyTorch 公式の “Get Started” で発行されるコマンド推奨）。

---

## 使い方（CLI）

### 1) プロンプトファイルを用意（1行 = 1プロンプト）
```text
顧客からの問い合わせ: 返品は可能ですか？ 店舗とオンラインの違いも教えてください。
顧客からの問い合わせ: 配送はどれくらいかかりますか？ 地域別の目安も含めてください。
```

### 2) 生成 → 品質レポ → 保存を 1 コマンド
```powershell
sdlg run --prompts .\prompts.txt -n 1 ^
  --model distilgpt2 --lang ja --max-new-tokens 60 ^
  --out outputs\synthetic.csv --report outputs\r
eport.json
```

### 3) 指示追従モデル（Colab/GPU 推奨）
```powershell
sdlg run --prompts .\prompts.txt -n 2 ^
  --model "Qwen/Qwen2.5-1.5B-Instruct" --chat --lang ja --max-new-tokens 120
```

---

## 出力物
- **`outputs/synthetic.csv`** … 生成データセット（`prompt, response`）
- **`outputs/report.json`** … 品質レポ v1（集計値 + 合否）

---

## プロジェクト構成
```text
src/sdlg/
  __init__.py
  cli.py        # sdlg コマンド（生成→品質→保存）
  generator.py  # 生成ロジック（free-form / chat）
  quality.py    # 品質レポ v1（言語/重複/毒性/PII など）
```

---

## よくある質問
- **出力が“質問の繰り返し”になる**  
  → `distilgpt2` など通常の GPT2 系は続き書き用です。Q&A には `--chat` と Instruct 系モデル（例：`Qwen2.5-1.5B-Instruct`）を使用してください。  
- **遅い / 量を捌けない**  
  → GPU 環境（Colab など）で実行し、`--max-new-tokens` を必要十分に抑えると改善します。

---

## 今後の予定
- レシピ保存・再現（model / params / seed）
- 並列実行・優先度付き再生成
- SFT / RAG 用フォーマット出力
- しきい値の CLI オプション化
- 簡易 UI（Streamlit）

---

## ライセンス / 注意
- 本ツールは生成データの**利用結果の責任を利用者が負う**前提です。個人情報・名指しなどの混入にはご注意ください。
- 使用する各モデル・データセットの**ライセンス/利用規約**を必ず確認のうえご利用ください。

## Run with a recipe

```bash
sdlg run --recipe recipes/qwen_ja_det.json --prompts examples/prompts_ja.txt
```

- CLI フラグも併用できるが、**レシピの値が優先して上書き**される  
- レシピを使うと「同じ条件での再現実行」が簡単になる  
- 最小レシピ例:

```json
{
  "model": "Qwen2.5-1.5B-Instruct",
  "chat": true,
  "deterministic": true
}
```
## Run on Google Colab (GPU/T4 recommended)

```bash
# 1) Install (upload the wheel to Colab Files pane or use a URL)
pip install "https://github.com/<owner>/<repo>/releases/download/v0.1.1/sdlg-0.1.1-py3-none-any.whl"

# 2) Prepare recipe & prompts
mkdir -p /content/recipes /content/examples /content/outputs

cat << 'JSON' > /content/recipes/qwen_ja_det.json
{
  "schema_version": "1",
  "model": "Qwen2.5-1.5B-Instruct",
  "chat": true,
  "max_new_tokens": 512,
  "temperature": 0.0,
  "top_p": 1.0,
  "deterministic": true,
  "seed": 42,
  "system": "You are a helpful Japanese assistant.",
  "n_per_prompt": 8,
  "lang": "ja"
}
JSON

cat << 'TXT' > /content/examples/prompts_ja.txt
テスト用の短い質問です。1文で答えてください。
TXT

# 3) Run
sdlg run \
  --recipe /content/recipes/qwen_ja_det.json \
  --prompts /content/examples/prompts_ja.txt \
  --out /content/outputs/synthetic.csv \
  --report /content/outputs/report.json

# 4) Preview outputs
head -n 5 /content/outputs/synthetic.csv
```

- 重いモデルは Colab（GPU）で、ローカルは `recipes/mini_local.json` でスモークテストするのが推奨です。
