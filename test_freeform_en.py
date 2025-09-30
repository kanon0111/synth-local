from sdlg.generator import TextGenerator
prompts = [
    "Write a short FAQ answer about return policy. Mention store vs online differences.",
    "What is the shipping time by region? Provide rough estimates."
]
gen = TextGenerator("distilgpt2", max_new_tokens=60, deterministic=True)
rows = gen.generate(prompts, n_per_prompt=1)
print("Resp1:", rows[0]["response"][:200])
print("Resp2:", rows[1]["response"][:200])
