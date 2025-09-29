from sdlg.generator import TextGenerator
prompts = ["顧客からの問い合わせ: 返品は可能ですか？ 店舗とオンラインの違いも教えてください。"]
g1 = TextGenerator("distilgpt2", max_new_tokens=40, deterministic=True)
g2 = TextGenerator("distilgpt2", max_new_tokens=40, deterministic=True)
print("run1:", g1.generate(prompts, n_per_prompt=1)[0]["response"][:120])
print("run2:", g2.generate(prompts, n_per_prompt=1)[0]["response"][:120])
