"""Offline subset coverage of the bench outputs. usage: python sdh_cov.py MODEL_DIR base.json profile.pt..."""
import sys, json, torch
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
d = json.load(open(sys.argv[2])); outs = next(iter(d.values()))
prompts = {
    "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings, then unit tests.",
    "prose": "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
    "reason": "A train leaves at 9:40 going 84 km/h; another leaves the same station at 10:15 going 112 km/h on the same track. When and where does the second catch the first? Show your work step by step.",
}
profs = {"idorder": torch.arange(248320)}
for fn in sys.argv[3:]:
    profs[fn.split("/")[-1]] = torch.load(fn)["ids"]
for pn, ids in profs.items():
    for K in (4096, 8192, 16384, 32768, 65536):
        top = set(ids[:K].tolist())
        row = []
        for name, toks in outs.items():
            pr = set(tok.encode(prompts[name]).ids)
            static = sum(t not in top for t in toks) / len(toks)
            withp = sum(t not in top and t not in pr for t in toks) / len(toks)
            seen = set(pr); dyn = 0
            for t in toks:
                if t not in top and t not in seen: dyn += 1
                seen.add(t)
            row.append(f"{name} {static*100:5.1f}/{withp*100:5.1f}/{dyn/len(toks)*100:5.1f}")
        print(f"{pn:22s} K={K:6d}  " + "  ".join(row))
print("columns: outside% static / +prompt tokens / +prompt and all earlier output tokens (online)")
