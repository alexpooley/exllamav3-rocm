#!/bin/bash
# Card-0 GPU run with the production env, under the shared GPU + build locks, hard kill.
# usage: grun.sh TIMEOUT_S LOG cmd...   (runs in rocm/dh4's prof dir: ~/code/exl3/prof)
T=$1; L=$2; shift 2
export HIP_VISIBLE_DEVICES=0 PATH=/home/alex/code/exl3/.venv/bin:$PATH
export EXL3_TREE=${EXL3_TREE:-/home/alex/code/exl3/wt-q27dh4}
export PYTHONPATH=$EXL3_TREE
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn} EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
[ -e /tmp/xtx-card0-test ] || { echo "card 0 not in test mode"; exit 1; }
cd /home/alex/code/exl3/prof
exec flock /tmp/xtx-gpu.lock flock /tmp/xtx-build.lock timeout -s KILL $T "$@" > $L 2>&1
