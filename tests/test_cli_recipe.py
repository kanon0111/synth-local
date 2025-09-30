import json, sys, subprocess
from pathlib import Path

def _run(args):
    return subprocess.run([sys.executable, "-m", "sdlg.cli"] + args,
                          capture_output=True, text=True)

def test_recipe_overrides_and_deterministic(tmp_path: Path):
    # prompts
    p = tmp_path / "p.txt"
    p.write_text("テスト用の短い質問です。1文で答えてください。\n", encoding="utf-8")

    # recipe (temperature=0.0, deterministic=true)
    r = tmp_path / "r.json"
    r.write_text(json.dumps({
        "model": "distilgpt2",
        "chat": False,
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "deterministic": True
    }, ensure_ascii=False), encoding="utf-8")

    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"

    res1 = _run(["run","--prompts", str(p),"--recipe", str(r),
                 "--out", str(a_csv), "--report", str(tmp_path/"a.json")])
    assert res1.returncode == 0, res1.stderr

    res2 = _run(["run","--prompts", str(p),"--temperature","0.7","--recipe", str(r),
                 "--out", str(b_csv), "--report", str(tmp_path/"b.json")])
    assert res2.returncode == 0, res2.stderr

    assert a_csv.read_bytes() == b_csv.read_bytes(), "CSV should be identical (recipe override + deterministic)"

def test_bad_recipe_fails(tmp_path: Path):
    p = tmp_path / "p.txt"
    p.write_text("hi\n", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{broken json", encoding="utf-8")

    res = _run(["run","--prompts", str(p),"--recipe", str(bad),
                "--out", str(tmp_path/"o.csv"), "--report", str(tmp_path/"r.json")])
    assert res.returncode != 0
