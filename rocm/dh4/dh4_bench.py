"""Same-process A/B of the DFlash2 drafter head: base (target's own head) vs EXL3_DRAFT_HEAD file.
Greedy, prompts code / prose / json / reason x --tokens, fixed window 7 and adaptive, --reps interleaved
repeats (one warm pass first). Prints per-prompt tok/s, tokens/round (= 1 + mean accepted drafts) and
ms/round; then checks the output ids are token-identical between heads (bsz 1, plus one bsz-2 pair).
usage: EXL3_DRAFT_HEAD=FILE python dh4_bench.py -m MODEL -dm DRAFT -cs 16384 [--tokens 512 --reps 2]"""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.join(os.environ.get("EXL3_TREE", os.path.expanduser("~/code/exl3/phoenixhaxor_exllamav3-rocm")), "examples"))

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings, then unit tests.",
    "prose": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
    "json": "Output only JSON, no prose: an array of 12 products, each with id, name, category, price, in_stock (bool), tags (array of strings) and a dimensions object (w, h, d in cm).",
    "reason": "A train leaves at 9:40 going 84 km/h; another leaves the same station at 10:15 going 112 km/h on the same track. When and where does the second catch the first? Show your work step by step.",
}

def main():
    from exllamav3 import model_init, Generator, Job, GreedySampler
    from common import format_prompt
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--tokens", type=int, default=512); p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out", default="dh4_bench.json")
    args = p.parse_args()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        lm = model.modules[model.logit_layer_idx]
        dh = dmodel._load_draft_head(lm)            # resident in both arms: same VRAM state
        dmodel.draft_head_path = None
        print(f"target head K={lm.inner.K} {lm.inner.trellis.numel()*2/1e6:.1f} MB; draft head K={dh.K} {dh.trellis.numel()*2/1e6:.1f} MB", flush=True)
        enc = lambda q: tok.encode(format_prompt("chatml", "You are a helpful assistant.", q, no_think=True) + "\n\n", encode_special_tokens=True)
        ids = {k: enc(q) for k, q in PROMPTS.items()}
        # One Generator may own the cache at a time: build a fresh one per arm
        def fixed7():
            g = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, num_draft_tokens=7)
            g.adaptive_window = None
            return g
        gens = {"fixed7": fixed7,
                "adapt": lambda: Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok)}

        def run1(gen, inp):
            job = Job(input_ids=inp, max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler())
            gen.enqueue(job); r = None; out = []
            while gen.num_remaining_jobs():
                for res in gen.iterate():
                    if res.get("stage") == "error": raise res["error"]
                    if "token_ids" in res: out += res["token_ids"].flatten().tolist()
                    if res.get("eos"): r = res
            n = r["new_tokens"]; acc = r.get("accepted_draft_tokens") or 0
            return dict(n=n, t=r["time_generate"], rounds=n - acc, ids=out)

        res = []; outs = {}
        def arm(head, mode, rep, record=True):
            dmodel.draft_head = dh if head == "dh4" else None
            gen = gens[mode]()
            for name in PROMPTS:
                x = run1(gen, ids[name])
                outs.setdefault((head, mode, name), []).append(x.pop("ids"))
                if record:
                    x.update(head=head, mode=mode, name=name, rep=rep); res.append(x)
                    print(f"rep{rep} {head:4s} {mode:6s} {name:6s} {x['n']/x['t']:6.1f} tok/s  t/r {x['n']/x['rounds']:4.2f}  ms/round {x['t']/x['rounds']*1e3:6.2f}", flush=True)
        for head in ("base", "dh4"):
            for mode in gens: arm(head, mode, 0, record=False)   # warm
        for rep in range(1, args.reps + 1):
            order = ("base", "dh4") if rep % 2 else ("dh4", "base")
            for head in order:
                for mode in gens: arm(head, mode, rep)

        print("\n== summary (all prompts pooled; per rep)")
        for mode in gens:
            for head in ("base", "dh4"):
                for rep in range(1, args.reps + 1):
                    rr = [x for x in res if x["mode"] == mode and x["head"] == head and x["rep"] == rep]
                    n = sum(x["n"] for x in rr); t = sum(x["t"] for x in rr); ro = sum(x["rounds"] for x in rr)
                    print(f"{mode:6s} {head:4s} rep{rep}  {n/t:6.1f} tok/s  t/r {n/ro:4.3f}  ms/round {t/ro*1e3:6.3f}")
        print("\n== acceptance by prompt (tokens/round, mean over reps and modes)")
        for name in PROMPTS:
            line = f"{name:6s}"
            for head in ("base", "dh4"):
                for mode in gens:
                    rr = [x for x in res if x["mode"] == mode and x["head"] == head and x["name"] == name]
                    line += f"  {head}/{mode} {sum(x['n'] for x in rr)/sum(x['rounds'] for x in rr):4.3f}"
            print(line)

        print("\n== exactness (bsz 1)")
        ok = True; ref = None
        for name in PROMPTS:
            allv = [tuple(v) for k, vs in outs.items() if k[2] == name for v in vs]
            same = len(set(allv)) == 1
            ok &= same
            print(f"{name:6s} {len(allv)} runs, {'identical' if same else 'DIFFER'} ({len(allv[0])} tokens)")
        # bsz 2: two prompts in flight together, per head
        pair = {}
        for head in ("base", "dh4"):
            dmodel.draft_head = dh if head == "dh4" else None
            gen = gens["adapt"](); o = {}
            jobs = []
            for name in ("code", "json"):
                j = Job(input_ids=ids[name], max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler(), identifier=name)
                gen.enqueue(j); o[name] = []
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "error": raise r["error"]
                    if "token_ids" in r: o[r["identifier"]] += r["token_ids"].flatten().tolist()
            pair[head] = o
        for name in ("code", "json"):
            same = pair["base"][name] == pair["dh4"][name]
            ok &= same
            print(f"bsz2 {name:6s} base vs dh4 {'identical' if same else 'DIFFER'}; bsz2 vs bsz1 {'identical' if pair['base'][name] == outs[('base','adapt',name)][0] else 'differ'}")
        print("EXACT" if ok else "NOT EXACT")
        json.dump(res, open(args.out, "w"))

if __name__ == "__main__":
    main()
