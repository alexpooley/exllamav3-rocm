#!/bin/bash
# Follow-up to queue.sh: 4-bit drafter head (q27-dh4 09340f3, opt-in EXL3_DRAFT_HEAD) merged into q27-serve.
# Waits for queue.sh, merges (Python only, no rebuild), then exactness, in-process speed, full-context needle +
# VRAM peak, served speed. Card 1 only.
P=/home/alex/code/exl3/phoenixhaxor_exllamav3-rocm; S=/home/alex/code/exl3/wt-q27serve
R=$S/rocm/serve; O=/home/alex/code/exl3/serve-ab; H=/home/alex/code/exl3/models/work-dh4/lm_head-4bpw.safetensors
until grep -q "QUEUE DONE" $O/queue.log 2>/dev/null; do sleep 30; done
cd $S
if ! git merge-base --is-ancestor 09340f3 HEAD; then
  [ -z "$(git diff --name-only HEAD 09340f3 -- '*.cu' '*.cuh' '*.cpp' '*.h' '*.hip')" ] || { echo "dh4 touches C++: rebuild first"; exit 1; }
  git -c user.name=alex -c user.email=alex@brownbeagle.com.au merge --no-ff --no-edit 09340f3 || exit 1
fi
$R/run1.sh $S $O/exact-servem.log 1800 $R/exact.py -cs 40960 -cq 8 --dump $O/exact-servem.json
EXL3_DRAFT_HEAD=$H $R/run1.sh $S $O/exact-dh4.log 1800 $R/exact.py -cs 40960 -cq 8 --dump $O/exact-dh4.json
python3 $R/cmp_exact.py $O/exact-prod.json $O/exact-servem.json > $O/exact-cmp-servem.txt
python3 $R/cmp_exact.py $O/exact-prod.json $O/exact-dh4.json > $O/exact-cmp-dh4.txt
for r in 1 2; do
  $R/run1.sh $S $O/bench-servem-r$r.log 1200 $R/q27_bench.py -cs 16384 --windows 7 --skip_nodraft
  EXL3_DRAFT_HEAD=$H $R/run1.sh $S $O/bench-dh4-r$r.log 1200 $R/q27_bench.py -cs 16384 --windows 7 --skip_nodraft
done
NEEDLE=1 NOSPEED=1 NOEVAL=1 $R/served.sh prod-needle $P
EXL3_DRAFT_HEAD=$H NEEDLE=1 NOEVAL=1 $R/served.sh dh4 $S
echo "$(date -u +%T) DH4 QUEUE DONE" >> $O/queue.log
