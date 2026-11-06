#!/bin/bash
# build the q27-gdn worktree extension in place. usage: gbuild.sh LOG
T=${2:-/home/alex/code/exl3/wt-q27serve}
L=${1:-$T/build.log}
export PATH=/home/alex/code/exl3/.venv/bin:$PATH
exec 8>/tmp/xtx-build.lock; flock 8
cd $T
ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=10 CCACHE_DIR=/home/alex/.cache/ccache-exl3 CCACHE_BASEDIR=/home/alex/code/exl3 CCACHE_NOHASHDIR=1 CCACHE_NODIRECT=1 CCACHE_MAXSIZE=30G timeout -s KILL 2400 python /home/alex/code/exl3/prof/build-ext-q27.py $T > $L 2>&1
echo "exit $?" >> $L
