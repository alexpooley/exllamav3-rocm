#!/bin/bash
# build the persistent MLP prototype (JIT extension) under the shared build lock
export PATH=/home/alex/code/exl3/.venv/bin:$PATH ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=4
exec 8>/tmp/xtx-build.lock; flock 8
cd /home/alex/code/exl3/wt-q27persist/persist
timeout -s KILL 900 python pm_bench.py --build-only 2>&1 | grep -v "^$" | tail -40
