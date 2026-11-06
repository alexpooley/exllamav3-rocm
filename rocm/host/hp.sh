#!/bin/bash
# card-1 run with the production env. usage: hp.sh TREE LOG script.py [args]
T=$1; L=$2; S=$3; shift 3
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
export HIP_VISIBLE_DEVICES=1 PATH=/home/alex/code/exl3/.venv/bin:$PATH PYTHONPATH=$T EXL3_TREE=$T
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn} EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
M=/home/alex/code/exl3/models
cd /home/alex/code/exl3/prof && timeout -s KILL 1200 python $S -m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 "$@" > $L 2>&1
echo "exit $?" >> $L
