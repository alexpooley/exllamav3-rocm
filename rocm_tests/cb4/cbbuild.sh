#!/bin/bash
# build wt-q27cb extension with 4-bit codebook variant V, keep a copy as so.cbV. usage: cbbuild.sh V
V=$1; T=/home/alex/code/exl3/wt-q27cb; L=$T/build-cb$V.log
Q=$T/exllamav3/exllamav3_ext/quant
printf '#ifndef EXL3_CB4\n#define EXL3_CB4 %s\n#endif\n' $V > $Q/exl3_cb4_local.h.new
cmp -s $Q/exl3_cb4_local.h.new $Q/exl3_cb4_local.h && rm $Q/exl3_cb4_local.h.new || mv $Q/exl3_cb4_local.h.new $Q/exl3_cb4_local.h
export PATH=/home/alex/code/exl3/.venv/bin:$PATH
exec 8>/tmp/xtx-build.lock; flock 8
cd $T
ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=10 CCACHE_DIR=/home/alex/.cache/ccache-exl3 CCACHE_BASEDIR=/home/alex/code/exl3 CCACHE_NOHASHDIR=1 CCACHE_NODIRECT=1 CCACHE_MAXSIZE=30G timeout -s KILL 2400 python /home/alex/code/exl3/prof/build-ext-q27.py $T > $L 2>&1
rc=$?; echo "exit $rc" >> $L
[ $rc = 0 ] && cp $T/exllamav3_ext.cpython-312-x86_64-linux-gnu.so $T/so.cb$V
echo "build cb$V exit $rc"
