"""q27 in-process round bench on the port: no draft, DFlash2 at fixed windows, adaptive window.
Greedy, 3 prompts x --tokens. Prints tok/s, tokens/round, ms/round.
usage: python q27_bench.py -m MODEL -dm DRAFT -cs 16384 --windows 3,5,7 [--tokens 512]"""
import sys, os, time, argparse, torch
sys.path.insert(0, os.path.join(os.environ.get("EXL3_TREE", os.path.expanduser("~/code/exl3/phoenixhaxor_exllamav3-rocm")), "examples"))

def main():
    import exllamav3
    from exllamav3 import model_init, Generator, Job, GreedySampler
    print(f"loaded exllamav3 from {os.path.dirname(exllamav3.__file__)}", flush=True)
    from common import format_prompt
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--windows", default="3,5,7"); p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--skip_nodraft", action="store_true")
    args = p.parse_args()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        prompts = {
            "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings, then unit tests.",
            "prose": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
            "reason": "A train leaves at 9:40 going 84 km/h; another leaves the same station at 10:15 going 112 km/h on the same track. When and where does the second catch the first? Show your work step by step.",
        }
        def run(gen, label):
            tot_t = tot_n = tot_r = 0.0
            for name, q in list(prompts.items()) * 2:   # first pass warms, second counts
                ids = tok.encode(format_prompt("chatml", "You are a helpful assistant.", q, no_think=True) + "\n\n", encode_special_tokens=True)
                job = Job(input_ids=ids, max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler())
                gen.enqueue(job); r = None
                while gen.num_remaining_jobs():
                    for res in gen.iterate():
                        if res.get("stage") == "error": raise res["error"]
                        if res.get("eos"): r = res
                n = r["new_tokens"]; t = r["time_generate"]; acc = r.get("accepted_draft_tokens")
                rounds = n - acc if acc is not None else n
                if name in prompts and tot_n == 0 and not getattr(run, "warm_" + label, False):
                    pass
                tot_t += t; tot_n += n; tot_r += rounds
                print(f"{label:10s} {name:6s} {n/t:6.1f} tok/s  tokens/round {n/rounds:4.2f}  ms/round {t/rounds*1e3:6.2f}", flush=True)
            print(f"{label:10s} ALL    {tot_n/tot_t:6.1f} tok/s  tokens/round {tot_n/tot_r:4.2f}  ms/round {tot_t/tot_r*1e3:6.2f}", flush=True)
        if not args.skip_nodraft:
            run(Generator(model=model, cache=cache, tokenizer=tok), "no-draft")
        for w in [int(x) for x in args.windows.split(",")]:
            g = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, num_draft_tokens=w)
            g.adaptive_window = None
            run(g, f"fixed-{w}")
        g = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok)
        run(g, f"adapt-{g.num_draft_tokens}")

if __name__ == "__main__":
    main()
