"""Greedy exactness dump for q27 + DFlash2, in process, production env (run through rocm/serve/run1.sh).
Prompts: code, prose, json, long (~30k tokens of code + a question). Batch 1 each, then one batch-2 pair.
Writes {name: token ids} to --dump; compare two dumps with rocm/serve/cmp_exact.py.
usage: python exact.py -m MODEL -dm DRAFT -cs 40960 -cq 8 [--tokens 512] --dump out.json"""
import sys, os, json, time, argparse, torch
sys.path.insert(0, os.path.join(os.environ["EXL3_TREE"], "examples"))

PROMPTS = {
    "code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings and a pytest test suite.",
    "prose": "Write a 700-word essay on why small households should keep paper records alongside digital ones. Use varied sentence structure.",
    "json": "Extract every person, organisation and date from this text and return a JSON array of {type, value, context}: " + (
        "On 3 March 2024 Dr Helen Park of Monash University met Rajesh Iyer from the CSIRO in Canberra. "
        "They agreed a study with the Bureau of Meteorology starting 1 July 2025, reviewed by Prof. Liam Chen of ANU on 14 August 2025. ") * 4,
    "long": open(os.path.expanduser("~/models/dflash/longctx.txt")).read()
            + "\n\nPick the longest function in the code above. Rewrite it with full type hints and a docstring, then explain each change.",
}

def main():
    import exllamav3
    from exllamav3 import model_init, Generator, Job, GreedySampler
    from common import format_prompt
    print(f"loaded exllamav3 from {os.path.dirname(exllamav3.__file__)}", flush=True)
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--tokens", type=int, default=512); p.add_argument("--dump", required=True)
    p.add_argument("--only", default=None); p.add_argument("--window", type=int, default=None)
    args = p.parse_args()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        gen = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, max_batch_size=2,
                        num_draft_tokens=args.window)
        print(f"window {gen.num_draft_tokens} adaptive {gen.adaptive_window is not None}", flush=True)
        enc = lambda q: tok.encode(format_prompt("chatml", "You are a helpful assistant.", q, no_think=True) + "\n\n",
                                   encode_special_tokens=True)
        def run(names):
            toks = {n: [] for n in names}; res = {}
            for n in names:
                gen.enqueue(Job(input_ids=enc(PROMPTS[n]), max_new_tokens=args.tokens, stop_conditions=[],
                                sampler=GreedySampler(), identifier=n))
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "error": raise r["error"]
                    if r.get("token_ids") is not None: toks[r["identifier"]] += r["token_ids"].view(-1).tolist()
                    if r.get("eos"): res[r["identifier"]] = r
            return toks, res
        out = {}
        names = [n for n in PROMPTS if not args.only or n in args.only.split(",")]
        for n in names:
            t0 = time.time(); toks, res = run([n]); r = res[n]
            out[n] = toks[n]
            print(f"{n:6s} prompt {r.get('prompt_tokens')} new {r['new_tokens']} accepted {r.get('accepted_draft_tokens')} "
                  f"{r['new_tokens']/r['time_generate']:6.1f} tok/s wall {time.time()-t0:5.1f}s", flush=True)
        if not args.only:
            toks, res = run(["code", "json"])
            out["b2_code"], out["b2_json"] = toks["code"], toks["json"]
            print("batch-2 pair done", flush=True)
        from exllamav3.generator import generator as gm
        print("switches: early_accept", getattr(gm, "_early_accept_enable", None), "predraft", getattr(gm, "_predraft_enable", None),
              "early_accept_rounds", getattr(gen, "early_accept_rounds", None), "predraft_discarded", getattr(gen, "predraft_discarded", None), flush=True)
        json.dump(out, open(args.dump, "w"))
        print("dumped", args.dump, flush=True)

if __name__ == "__main__":
    main()
