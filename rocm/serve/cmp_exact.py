"""Compare two exact.py dumps. usage: python cmp_exact.py A.json B.json"""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); ok = True
for k in a:
    if k not in b: print(f"{k}: missing in B"); ok = False; continue
    x, y = a[k], b[k]
    d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    same = x == y
    print(f"{k:8s} len {len(x)}/{len(y)} identical={'yes' if same else 'no'} first_diff={d}")
    ok &= same
print("EXACT" if ok else "DIFF")
