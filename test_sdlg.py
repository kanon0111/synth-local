from sdlg.generator import TextGenerator
print("Import OK ✅")
gen = TextGenerator(model_id="distilgpt2", max_new_tokens=60, chat=False)  # 小型/続き書き
rows = gen.generate(
    ["顧客からの問い合わせ: 返品は可能ですか？ 店舗とオンラインの違いも教えてください。"],
    n_per_prompt=1
)
print("Sample response:\n", rows[0]["response"][:300])
