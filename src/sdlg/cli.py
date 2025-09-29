import argparse, os, json, pandas as pd
from sdlg.generator import TextGenerator
from sdlg.quality import summarize_quality, pass_fail

def _read_prompts(path: str):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def cmd_run(args):
    prompts = _read_prompts(args.prompts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    gen = TextGenerator(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chat=args.chat,
        system_text=args.system,
        seed=args.seed,
    )
    rows = gen.generate(prompts, n_per_prompt=args.n_per_prompt)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")

    responses = [r["response"] for r in rows]
    rep = summarize_quality(responses, target_lang=args.lang)
    gates = pass_fail(rep)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"report": rep, "pass_fail": gates}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved dataset : {args.out}")
    print(f"Saved report  : {args.report}")
    print("\nQuality report:")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("Pass/Fail:", gates)

def main():
    p = argparse.ArgumentParser(prog="sdlg", description="Synthetic Data Local Generator CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="Generate -> Quality -> Save")
    r.add_argument("--prompts", required=True, help="1行=1プロンプトのUTF-8テキスト")
    r.add_argument("-n", "--n-per-prompt", type=int, default=1)
    r.add_argument("--model", default="distilgpt2", help="HF model id")
    r.add_argument("--chat", action="store_true", help="Instruct/Chatモデルを使用")
    r.add_argument("--system", default="あなたは丁寧な日本語で答えるカスタマーサポート担当者です。")
    r.add_argument("--lang", default="ja", help="品質レポの言語判定ターゲット（ja/enなど）")
    r.add_argument("--max-new-tokens", type=int, default=120)
    r.add_argument("--temperature", type=float, default=0.8)
    r.add_argument("--top_p", type=float, default=0.9)
    r.add_argument("--seed", type=int, default=None, help="乱数シード（再現性用）")
    r.add_argument("--out", default="outputs/synthetic.csv")
    r.add_argument("--report", default="outputs/report.json")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)
