import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); ok = True
for k in a:
    if k.endswith("_ids"):
        for r, (x, y) in enumerate(zip(a[k], b[k])):
            same = x == y
            n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
            print(f"{k} row{r}: len {len(x)}/{len(y)} identical={same} first_diff={n}")
            ok &= same
    elif k.endswith("_rounds"):
        print(f"{k}: {a[k]} / {b[k]}"); ok &= a[k] == b[k]
print("EXACT" if ok else "DIFF")
