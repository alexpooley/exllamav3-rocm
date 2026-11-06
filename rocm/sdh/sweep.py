"""Sparse draft head K sweep in one process (q27_bench prompts, greedy, fixed window).
For each K in --ks (0 = full head): one warm pass + --reps counted passes over the 3 prompts.
Prints per prompt tok/s, tokens/round, ms/round, the share of generated tokens outside the subset,
and whether the token ids match K=0 exactly. --dump writes {label: {prompt: ids}}.
usage: python sdh_sweep.py -m MODEL -dm DRAFT -cs 16384 --ks 0,4096,8192,0 [--window 7] [--tokens 512]"""
import sys, os, json, argparse, collections, torch
sys.path.insert(0, os.path.join(os.environ["EXL3_TREE"], "examples"))

def main():
    from exllamav3 import model_init, Generator, Job, GreedySampler
    from common import format_prompt
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--ks", default="0,4096,8192,16384,32768,0"); p.add_argument("--window", type=int, default=7)
    p.add_argument("--tokens", type=int, default=512); p.add_argument("--reps", type=int, default=2)
    p.add_argument("--dump", default=None)
    args = p.parse_args()
    prompts = {
        "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings, then unit tests.",
        "prose": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
        "reason": "A train leaves at 9:40 going 84 km/h; another leaves the same station at 10:15 going 112 km/h on the same track. When and where does the second catch the first? Show your work step by step.",
    }
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        gen = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, num_draft_tokens=args.window)
        gen.adaptive_window = None
        ref = {}; dump = {}; summary = []
        for ki, K in enumerate(int(x) for x in args.ks.split(",")):
            label = f"K{K}#{ki}"
            dmodel.sdh_capacity = K; dmodel.sdh_weight = None; dmodel.sdh_cols = None
            agg = collections.defaultdict(lambda: [0.0, 0, 0])
            ids_out = {}
            for rep in range(args.reps + 1):
                for name, q in prompts.items():
                    ids = tok.encode(format_prompt("chatml", "You are a helpful assistant.", q, no_think=True) + "\n\n", encode_special_tokens=True)
                    job = Job(input_ids=ids, max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler())
                    gen.enqueue(job); r = None; toks = []
                    while gen.num_remaining_jobs():
                        for res in gen.iterate():
                            if res.get("stage") == "error": raise res["error"]
                            if res.get("token_ids") is not None: toks += res["token_ids"].view(-1).tolist()
                            if res.get("eos"): r = res
                    n = r["new_tokens"]; t = r["time_generate"]; rounds = n - r["accepted_draft_tokens"]
                    if rep == 0: continue
                    a = agg[name]; a[0] += t; a[1] += n; a[2] += rounds
                    if name in ids_out and ids_out[name] != toks: print(f"  !! {label} {name}: ids differ between reps")
                    ids_out[name] = toks
            if dmodel.sdh_cols is not None:
                member = set(dmodel.sdh_cols.tolist())
            else:
                member = None
            tt = [0.0, 0, 0]
            for name in prompts:
                t, n, rounds = agg[name]
                tt[0] += t; tt[1] += n; tt[2] += rounds
                toks = ids_out[name]
                miss = sum(1 for x in toks if x not in member) / len(toks) if member is not None else 0.0
                if K == 0 and name not in ref: ref[name] = toks
                exact = toks == ref.get(name)
                print(f"{label:10s} {name:6s} {n/t:6.1f} tok/s  tokens/round {n/rounds:4.2f}  ms/round {t/rounds*1e3:6.2f}  outside {miss*100:5.2f}%  exact={exact}", flush=True)
            print(f"{label:10s} ALL    {tt[1]/tt[0]:6.1f} tok/s  tokens/round {tt[1]/tt[2]:4.2f}  ms/round {tt[0]/tt[2]*1e3:6.2f}", flush=True)
            dump[label] = ids_out
        if args.dump: json.dump(dump, open(args.dump, "w"))

if __name__ == "__main__":
    main()
