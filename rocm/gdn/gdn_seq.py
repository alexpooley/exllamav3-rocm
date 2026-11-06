"""Kernel sequence of one iteration of a gdnbench config. usage: python gdn_seq.py OUT CONFIG [iter] [first_n]"""
import csv, json, sys, glob, re
out, cfg = sys.argv[1], sys.argv[2]; it = int(sys.argv[3]) if len(sys.argv) > 3 else 3
first = int(sys.argv[4]) if len(sys.argv) > 4 else 80
win = json.load(open(f"{out}/window.json"))
path = glob.glob(f"{out}/rp/**/*kernel_trace.csv", recursive=True)[0]
rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(path)))
w = win[cfg]
best = max(w["window"].items(), key=lambda kv: sum(1 for s, e, _ in rows if s >= kv[1][0] and e <= kv[1][1]))
a, b = best[1]; sel = [r for r in rows if r[0] >= a and r[1] <= b]
per = len(sel) // w["iters"]
seq = sel[it * per:(it + 1) * per]
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"<.*>", "<>", n); return n.split("::")[-1][:50]
prev = None
for s, e, n in seq[:first]:
    print(f"{(s - prev)/1e3 if prev else 0:6.1f} gap  {(e - s)/1e3:7.1f} us  {short(n)}"); prev = e
