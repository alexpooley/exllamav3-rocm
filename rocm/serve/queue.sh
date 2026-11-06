#!/bin/bash
# The q27-serve A/B, card 1 only: exactness, in-process speed, then served speed + quality. Each step takes the
# card-1 and build locks itself (run1.sh / served.sh). usage: queue.sh [steps]  (default: exact bench served)
P=/home/alex/code/exl3/phoenixhaxor_exllamav3-rocm; S=/home/alex/code/exl3/wt-q27serve
R=$S/rocm/serve; O=/home/alex/code/exl3/serve-ab; mkdir -p $O
for step in ${@:-exact bench served}; do case $step in
exact)
  $R/run1.sh $P $O/exact-prod.log 1800 $R/exact.py -cs 40960 -cq 8 --dump $O/exact-prod.json
  $R/run1.sh $S $O/exact-serve.log 1800 $R/exact.py -cs 40960 -cq 8 --dump $O/exact-serve.json
  python3 $R/cmp_exact.py $O/exact-prod.json $O/exact-serve.json > $O/exact-cmp.txt ;;
bench)
  for r in 1 2; do for b in prod serve; do T=$P; [ $b = serve ] && T=$S
    $R/run1.sh $T $O/bench-$b-r$r.log 1200 $R/q27_bench.py -cs 16384 --windows 7 --skip_nodraft
    EXL3_ADAPT_WINDOW=1 $R/run1.sh $T $O/bench-$b-adaptenv-r$r.log 1200 $R/q27_bench.py -cs 16384 --windows 7 --skip_nodraft
  done; done ;;
served)
  $R/served.sh prod $P
  $R/served.sh serve $S ;;
esac; done
echo "$(date -u +%T) QUEUE DONE $*" >> $O/queue.log
