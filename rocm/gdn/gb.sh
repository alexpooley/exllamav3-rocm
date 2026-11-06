#!/bin/bash
# gdnbench on card 1, prod env. usage: gb.sh TREE LOG [--trace OUT] [args]
T=$1; L=$2; shift 2
TR=""; if [ "$1" = "--trace" ]; then TR=$2; shift 2; fi
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
export HIP_VISIBLE_DEVICES=1 PATH=/home/alex/code/exl3/.venv/bin:$PATH PYTHONPATH=$T EXL3_TREE=$T
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn} EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
M=/home/alex/code/exl3/models
R=/home/alex/code/exl3/.venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3
cd /home/alex/code/exl3/prof/gdn
if [ -n "$TR" ]; then
  rm -rf $TR; export ROCPROFILER_REGISTER_ENABLED=1
  timeout -s KILL 600 $R --kernel-trace --output-format csv -d $TR/rp -o kt -- python ${GBS:-gdnbench.py} --out $TR -m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 "$@" > $L 2>&1
else
  timeout -s KILL 600 python ${GBS:-gdnbench.py} -m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 "$@" > $L 2>&1
fi
echo "exit $?" >> $L
