"""Host timeline of one q27 DFlash2 round, untraced. Patches timers around generator phases and every
host sync, runs the q27_bench prompts at a fixed window, prints per-phase median times relative to
iterate() start and the time spent blocked in each sync.
usage: python hostprof.py -m MODEL -dm DRAFT -cs 16384 [--window 7] [--tokens 512] [--dump tokens.json]"""
import sys, os, time, argparse, json, collections, statistics, functools, torch
sys.path.insert(0, os.path.join(os.environ.get("EXL3_TREE", os.path.expanduser("~/code/exl3/phoenixhaxor_exllamav3-rocm")), "examples"))

EV = []
ON = [False]
GEV = []          # (label, cuda event) recorded at selected launch points (--events)
GLAB = set()
ns = time.perf_counter_ns

def wrap(obj, name, label):
    f = getattr(obj, name)
    @functools.wraps(f)
    def g(*a, **k):
        if not ON[0]: return f(*a, **k)
        EV.append((label + "<", ns()))
        if label + "<" in GLAB:
            e = torch.cuda.Event(enable_timing=True); e.record(); GEV.append((label + "<", e))
        r = f(*a, **k)
        if label + ">" in GLAB:
            e = torch.cuda.Event(enable_timing=True); e.record(); GEV.append((label + ">", e))
        EV.append((label + ">", ns()))
        return r
    setattr(obj, name, g)

def main():
    import exllamav3
    from exllamav3 import model_init, Generator, Job, GreedySampler
    from exllamav3.generator import job as jobmod
    from exllamav3.modules import gated_delta_net as gdn
    from common import format_prompt
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--window", type=int, default=7); p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--adapt", action="store_true"); p.add_argument("--dump", default=None)
    p.add_argument("--events", action="store_true")
    p.add_argument("--bsz", type=int, default=1); p.add_argument("--noprof", action="store_true")
    args = p.parse_args()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        if not args.noprof:
            wrap(Generator, "iterate", "iter")
            wrap(Generator, "iterate_draftmodel_dflash_gen", "dgen")
            wrap(Generator, "iterate_gen", "tgen")
            wrap(model, "forward", "tfwd")
            wrap(dmodel, "forward", "dfwd")
            wrap(dmodel, "sample_from_state", "dsamp")
            wrap(dmodel, "update_kv_from_target", "kvupd")
            wrap(jobmod.Job, "receive_sample", "rsamp")
            wrap(jobmod.Job, "presample_window", "presamp")
            wrap(gdn.GDNState, "rewind", "rewind")
            from exllamav3.modules.embedding import Embedding
            wrap(Embedding, "forward", "emb")
            if args.events: GLAB.update({"SSYNC<", "tfwd<", "emb>", "SYNC<", "rewind<", "kvupd<"})
            wrap(torch.cuda, "synchronize", "SYNC")
            wrap(torch.cuda.Stream, "synchronize", "SSYNC")
            wrap(torch.Tensor, "item", "ITEM")
            wrap(torch.Tensor, "cpu", "CPU")
        prompts = {
            "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings, then unit tests.",
            "prose": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
            "reason": "A train leaves at 9:40 going 84 km/h; another leaves the same station at 10:15 going 112 km/h on the same track. When and where does the second catch the first? Show your work step by step.",
        }
        if args.adapt:
            gen = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, max_batch_size=args.bsz)
        else:
            gen = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok, num_draft_tokens=args.window, max_batch_size=args.bsz)
            gen.adaptive_window = None
        dump = {}
        rounds_all = []
        global GROUNDS; GROUNDS = []
        tot = [0, 0, 0]
        for pas in range(2):
            plist = list(prompts.values())
            for pi, (name, q) in enumerate(prompts.items()):
                jobs = []
                for b in range(args.bsz):
                    ids = tok.encode(format_prompt("chatml", "You are a helpful assistant.", plist[(pi + b) % 3], no_think=True) + "\n\n", encode_special_tokens=True)
                    j = Job(input_ids=ids, max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler(), identifier=b)
                    gen.enqueue(j); jobs.append(j)
                res = {}; toks = collections.defaultdict(list)
                ON[0] = pas == 1 and not args.noprof
                while gen.num_remaining_jobs():
                    EV.clear(); GEV.clear()
                    for r in gen.iterate():
                        if r.get("stage") == "error": raise r["error"]
                        if r.get("token_ids") is not None: toks[r["identifier"]] += r["token_ids"].view(-1).tolist()
                        if r.get("eos"): res[r["identifier"]] = r
                    if ON[0] and EV: rounds_all.append(list(EV))
                    if ON[0] and GEV:
                        GEV[-1][1].synchronize()
                        g = {}; seen_t = False
                        for l, e in GEV:
                            if l == "tfwd<": seen_t = True
                            if l == "emb>" and not seen_t: continue
                            g.setdefault(l, e)
                        GROUNDS.append(g)
                ON[0] = False
                r = res[0]
                n = r["new_tokens"]; t = r["time_generate"]; acc = r.get("accepted_draft_tokens")
                rnd = n - acc
                print(f"pass{pas} {name:6s} {n/t:6.1f} tok/s  tokens/round {n/rnd:4.2f}  ms/round {t/rnd*1e3:6.2f}", flush=True)
                if pas == 1:
                    tot[0] += n; tot[1] += t; tot[2] += rnd
                    dump[name] = [res[b]["full_completion"] for b in sorted(res)]
                    dump[name + "_ids"] = [toks[b] for b in sorted(res)]
                    dump[name + "_rounds"] = rnd
        print("early_accept_rounds", getattr(gen, "early_accept_rounds", None), "predraft_discarded", getattr(gen, "predraft_discarded", None))
        print(f"ALL {tot[0]/tot[1]:6.1f} tok/s  tokens/round {tot[0]/tot[2]:4.2f}  ms/round {tot[1]/tot[2]*1e3:6.2f}", flush=True)
        if args.dump:
            json.dump(dump, open(args.dump, "w"))
        if args.noprof: return
        # Per-round timeline: keep rounds that ran both drafter and target (steady state)
        rel = collections.defaultdict(list); blocked = collections.defaultdict(list); cnt = collections.Counter()
        dur = collections.defaultdict(list)
        good = 0
        for evs in rounds_all:
            labels = [l for l, _ in evs]
            if "tfwd<" not in labels or "dfwd<" not in labels: continue
            good += 1
            t0 = evs[0][1]
            seen = collections.Counter()
            stack = {}
            for l, t in evs:
                base = l[:-1]
                seen[l] += 1
                key = f"{l}#{seen[l]}" if base in ("SYNC", "SSYNC", "ITEM", "CPU", "rsamp", "rewind", "presamp") else l
                rel[key].append((t - t0) / 1e3)
                if l.endswith("<"): stack[base] = t
                elif base in stack:
                    d = (t - stack.pop(base)) / 1e3
                    dur[base].append(d)
                    if base in ("SYNC", "SSYNC", "ITEM", "CPU"): blocked[base].append(d)
            for b in ("SYNC", "SSYNC", "ITEM", "CPU", "rsamp", "rewind"):
                cnt[b] += seen[b + "<"]
        def gap(a, b):
            v = [g[a].elapsed_time(g[b]) * 1e3 for g in GROUNDS if a in g and b in g]
            v = [x for x in v if x >= 0]
            return f"med {statistics.median(v):8.1f} us mean {statistics.mean(v):8.1f} n={len(v)}" if v else "n/a"
        if GROUNDS:
            print("GPU event gaps (GPU time between the two points; idle when the GPU had drained):")
            for a, b in (("SSYNC<", "tfwd<"), ("SSYNC<", "emb>"), ("SYNC<", "rewind<"), ("SYNC<", "kvupd<")):
                print(f"  {a:7s} -> {b:8s} {gap(a, b)}")
        print(f"rounds profiled {good}")
        print("mean count per round:", {k: round(v / good, 2) for k, v in cnt.items()})
        print("per-call duration us (median, mean, count/round):")
        for k, v in sorted(dur.items()):
            print(f"  {k:8s} med {statistics.median(v):8.1f}  mean {statistics.mean(v):8.1f}  sum/round {sum(v)/good:8.1f}")
        print("timeline (us from iterate start, median over rounds where present; n):")
        for k, v in sorted(rel.items(), key=lambda kv: statistics.median(kv[1])):
            if len(v) < good * 0.5: continue
            print(f"  {statistics.median(v):9.1f}  {k:14s} n={len(v)}")

if __name__ == "__main__":
    main()
