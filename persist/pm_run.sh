#!/bin/bash
# card-0 GPU run of the prototype: GPU lock + build lock, hard timeout. usage: pm_run.sh TIMEOUT LOG args...
T=$1; L=$2; shift 2
[ -e /tmp/xtx-card0-test ] || { echo "card 0 not in test mode"; exit 1; }
export HIP_VISIBLE_DEVICES=0 PATH=/home/alex/code/exl3/.venv/bin:$PATH ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn} EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
cd /home/alex/code/exl3/wt-q27persist/persist
(
  exec 8>/tmp/xtx-gpu.lock; flock 8
  exec 9>/tmp/xtx-build.lock; flock 9
  echo "start $(date -u +%T)"
  timeout -s KILL $T ${RUNNER:-python} "$@"
  echo "exit $? $(date -u +%T)"
) > $L 2>&1
