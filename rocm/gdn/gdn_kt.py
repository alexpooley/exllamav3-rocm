"""Per-config kernel ledger from a gdnbench.py --out trace. usage: python gdn_kt.py OUT [filter-substring]"""
import csv, json, sys, glob, re, collections
out = sys.argv[1]; flt = sys.argv[2] if len(sys.argv) > 2 else None
win = json.load(open(f"{out}/window.json"))
path = glob.glob(f"{out}/rp/**/*kernel_trace.csv", recursive=True)[0]
rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(path)))
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"<.*>", "<>", n); return n.split("::")[-1][:60]
for label, w in win.items():
    best = max(w["window"].items(), key=lambda kv: sum(1 for s, e, _ in rows if s >= kv[1][0] and e <= kv[1][1]))
    a, b = best[1]; sel = [r for r in rows if r[0] >= a and r[1] <= b]; N = w["iters"]
    busy = sum(e - s for s, e, _ in sel) / 1e6
    print(f"\n== {label}: {len(sel)/N:.0f} kernels/iter, GPU busy {busy/N:.3f} ms/iter")
    agg = collections.defaultdict(lambda: [0, 0])
    for s, e, n in sel: agg[short(n)][0] += e - s; agg[short(n)][1] += 1
    for n, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:14]:
        if flt and flt not in n: continue
        print(f"  {t/1e6/N:7.3f} ms/iter  {c/N:6.1f} calls/iter  {t/c/1e3:8.1f} us/call  {n}")
