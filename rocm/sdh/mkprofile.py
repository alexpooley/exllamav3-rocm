"""Token frequency profile for the sparse draft head.
usage: python mkprofile.py MODEL_DIR OUT.pt GROUP_WEIGHT:file[,file...] ...
Each group's counts are normalised to sum 1 and scaled by its weight; .jsonl files contribute their "content"
and "reasoning" fields (model outputs), other files their whole text. Ids never seen are appended ASCII-first, each
part in token-id order (BPE merge order, roughly frequency), so any capacity can be filled.
Writes {"ids": LongTensor of all vocab ids, best first, "score": matching scores, "seen": number seen}."""
import sys, json, torch
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
cfg = json.load(open(sys.argv[1] + "/config.json"))
vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
score = torch.zeros(vocab, dtype=torch.float64)
def texts(fn):
    if fn.endswith(".jsonl"):
        for line in open(fn):
            d = json.loads(line)
            for f in ("content", "reasoning"):
                if d.get(f): yield d[f]
    else:
        yield open(fn, errors="replace").read()
for arg in sys.argv[3:]:
    w, files = arg.split(":", 1)
    c = torch.zeros(vocab, dtype=torch.float64); n = 0
    for fn in files.split(","):
        for t in texts(fn):
            for i in range(0, len(t), 200000):    # tokenizers is slow on multi-MB strings
                ids = torch.tensor(tok.encode(t[i:i + 200000], add_special_tokens=False).ids, dtype=torch.long)
                ids = ids[ids < vocab]
                c.index_add_(0, ids, torch.ones(len(ids), dtype=torch.float64)); n += len(ids)
    print(f"group w={w}: {n} tokens, {int((c > 0).sum())} distinct  ({files[:80]})")
    score += float(w) * c / c.sum()
seen = int((score > 0).sum())
# Unseen ids: plain-ASCII tokens (English words, code) first, in token-id order, then the rest
ascii_ = torch.tensor([all(32 <= ord(ch) < 127 or ch in "\n\t" for ch in tok.decode([i])) and len(tok.decode([i])) > 0
                       for i in range(vocab)])
prior = torch.where(ascii_, 2.0, 1.0) - torch.arange(vocab, dtype=torch.float64) / vocab
key = torch.where(score > 0, score + 10.0, prior)    # seen ids always rank above unseen ones
order = torch.argsort(key, descending=True, stable=True)
# seen ids by score (argsort is stable over equal zero scores)
cum = score[order].cumsum(0) / score.sum()
print(f"vocab {vocab}, seen {seen}, ascii {int(ascii_.sum())}")
for k in (1024, 2048, 4096, 8192, 16384, 32768):
    print(f"top {k}: weighted coverage {cum[k-1]:.4f}")
torch.save({"ids": order, "score": score[order].float(), "ascii": int(ascii_.sum()), "seen": seen, "sources": sys.argv[3:]}, sys.argv[2])
