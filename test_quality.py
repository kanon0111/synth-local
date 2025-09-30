from sdlg.generator import TextGenerator
from sdlg.quality import summarize_quality, pass_fail

print("Import OK ✅")
gen = TextGenerator("distilgpt2", max_new_tokens=60, chat=False)
rows = gen.generate(["顧客からの問い合わせ: 返品は可能ですか？"], n_per_prompt=3)
responses = [r["response"] for r in rows]

rep = summarize_quality(responses, target_lang="ja")
print("Sample response:", rows[0]["response"][:200])
print("Report:", rep)
print("Pass/Fail:", pass_fail(rep))
