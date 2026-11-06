#!/bin/bash
# LDS bank conflicts / active cycles per kbench variant (gate shape, m=8 and m=1), card 1
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
export HIP_VISIBLE_DEVICES=1 KB_IT=20
D=/tmp/q27gemv-kb; O=/tmp/cb-pmc; rm -rf $O; mkdir -p $O
for t in cb0m8 cb2m8 cb6m8 cb1m8 cb7m8 cb8m8 cb3m8 cb0m1 cb6m1 cb7m1; do
  M=${t#*m}
  timeout -s KILL 120 /opt/rocm/bin/rocprofv3 --pmc SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_WAIT_INST_LDS --output-format csv -d $O/$t -o p -- $D/kb_$t 5120 17408 $M 288 > /dev/null 2>&1
  f=$(find $O/$t -name "*counter_collection.csv" | head -1)
  python3 - "$f" "$t" <<'PY'
import csv,sys,collections
s=collections.defaultdict(float); n=collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    if "exl3_rdna3_kernel" not in r["Kernel_Name"]: continue
    s[r["Counter_Name"]]+=float(r["Counter_Value"]); n[r["Counter_Name"]]+=1
k=max(n.values()) if n else 1
print(sys.argv[2], " ".join(f"{c}={v/ (n[c] or 1):.3g}" for c,v in sorted(s.items())), "conflict/active=%.2f" % (s["SQC_LDS_BANK_CONFLICT"]/max(1,s["SQC_LDS_IDX_ACTIVE"])))
PY
done
