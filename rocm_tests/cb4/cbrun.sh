#!/bin/bash
# per variant: install so.cbV, then gemv_exact timing + fwd_bench + q27_bench fixed-7 (+ no-draft), card 1.
# usage: cbrun.sh "V1 V2 ..." TAG [what=gfq]
VS=$1; TAG=$2; WHAT=${3:-gfq}
T=/home/alex/code/exl3/wt-q27cb; O=$T/res; mkdir -p $O
systemctl --user is-active -q llama-swap 2>/dev/null && { echo "llama-swap running"; exit 1; }
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
export HIP_VISIBLE_DEVICES=1 PATH=/home/alex/code/exl3/.venv/bin:$PATH PYTHONPATH=$T EXL3_TREE=$T
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn} EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
M=/home/alex/code/exl3/models
for V in $VS; do
  cp $T/so.cb$V $T/exllamav3_ext.cpython-312-x86_64-linux-gnu.so || exit 1
  cd /home/alex/code/exl3/prof
  [[ $WHAT == *g* ]] && { timeout -s KILL 900 python gemv_exact.py /tmp/cb-gx-$V.pt --time --ms 1,5,8 --tms 1,5,8 > $O/gx-$TAG-cb$V.log 2>&1; echo "exit $?" >> $O/gx-$TAG-cb$V.log; }
  [[ $WHAT == *f* ]] && { timeout -s KILL 900 python $T/rocm_tests/cb4/fwd_bench.py -m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -cs 16384 --rows 1,8 > $O/fwd-$TAG-cb$V.log 2>&1; echo "exit $?" >> $O/fwd-$TAG-cb$V.log; }
  [[ $WHAT == *q* ]] && { timeout -s KILL 1500 python q27_bench.py -m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 --windows 7 --tokens 256 > $O/qb-$TAG-cb$V.log 2>&1; echo "exit $?" >> $O/qb-$TAG-cb$V.log; }
  echo "done cb$V"
done
