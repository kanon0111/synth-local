import argparse, sys, json, os, csv

def _read_prompts(path: str):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def _load_recipe(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdlg", description="Synthetic Data Local Generator")
    sub = parser.add_subparsers(dest="cmd")

    # ping
    ping = sub.add_parser("ping", help="quick responsiveness check")
    ping.set_defaults(func=lambda a: print("pong"))

    # run
    run = sub.add_parser("run", help="Generate -> Quality -> Save")
    run.add_argument("--prompts", required=True, help="UTF-8 text file, one prompt per line")
    run.add_argument("--recipe", help="JSON recipe path (overrides CLI flags)")
    run.add_argument("--out", default="outputs/synthetic.csv")
    run.add_argument("--report", default="outputs/report.json")

    # generation flags
    run.add_argument("--model", help="HF model id (e.g., distilgpt2)")
    run.add_argument("--chat", action="store_true", help="Use chat/instruct mode")
    run.add_argument("--system", help="System prompt (chat mode only)")
    run.add_argument("--max-new-tokens", type=int, help="Max new tokens")
    run.add_argument("--temperature", type=float)
    run.add_argument("--top_p", type=float)
    run.add_argument("--seed", type=int)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("-n", "--n-per-prompt", type=int, default=1)
    run.add_argument("--lang", default="ja", help="Target language for quality report (ja/en, etc.)")
    run.add_argument("--print-config", action="store_true", help="print effective config and exit")

    def _cmd_run(args):
        # lazy imports so that `--help` is instant
        from sdlg.generator import TextGenerator
        from sdlg.quality import summarize_quality, pass_fail

        # load recipe if any (recipe overrides CLI flags)
        cfg = {}
        if args.recipe:
            cfg = _load_recipe(args.recipe)

        prompts = _read_prompts(args.prompts)

        # ensure dirs
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

        # effective config (recipe has priority over CLI)
        eff = {
            "model_id":       cfg.get("model") or args.model or "distilgpt2",
            "chat":           (cfg["chat"] if "chat" in cfg else args.chat),
            "system_text":    (cfg["system"] if "system" in cfg else args.system),
            "max_new_tokens": (cfg["max_new_tokens"] if "max_new_tokens" in cfg else (args.max_new_tokens if args.max_new_tokens is not None else 64)),
            "temperature":    (cfg["temperature"] if "temperature" in cfg else (args.temperature if args.temperature is not None else 0.8)),
            "top_p":          (cfg["top_p"] if "top_p" in cfg else (args.top_p if args.top_p is not None else 0.9)),
            "seed":           (cfg["seed"] if "seed" in cfg else args.seed),
            "deterministic":  (cfg["deterministic"] if "deterministic" in cfg else args.deterministic),
            "n_per_prompt":   cfg.get("n_per_prompt", args.n_per_prompt),
            "lang":           cfg.get("lang", args.lang),
        }

        # --print-config: show and exit
        if args.print_config:
            print(json.dumps(eff, ensure_ascii=False, indent=2))
            return 0

        # build generator
        gen = TextGenerator(
            model_id=eff["model_id"],
            max_new_tokens=eff["max_new_tokens"],
            temperature=eff["temperature"],
            top_p=eff["top_p"],
            chat=eff["chat"],
            system_text=eff["system_text"],
            seed=eff["seed"],
            deterministic=eff["deterministic"],
        )

        # generate
        rows = gen.generate(prompts, n_per_prompt=eff["n_per_prompt"])

        # write CSV (UTF-8 with BOM)
        fieldnames = ["prompt", "response"]
        if rows and isinstance(rows[0], dict):
            fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # quality
        responses = [r.get("response", "") for r in rows]
        report = summarize_quality(responses, target_lang=eff["lang"])
        gates = pass_fail(report)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"report": report, "pass_fail": gates}, f, ensure_ascii=False, indent=2)

        print("Saved dataset :", args.out)
        print("Saved report  :", args.report)
        print("Quality report:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Pass/Fail:", gates)
        return 0

    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    return args.func(args) or 0

if __name__ == "__main__":
    raise SystemExit(main())
