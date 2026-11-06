#!/bin/bash
# denormal test + kbench sweep on card 1 (under gpu1 + build locks)
systemctl --user is-active -q llama-swap 2>/dev/null && { echo "llama-swap running"; exit 1; }
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
export HIP_VISIBLE_DEVICES=1
D=/tmp/q27gemv-kb; T=/home/alex/code/exl3/wt-q27cb
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx1100 $T/rocm_tests/cb4/denorm.hip -o /tmp/cb-denorm && timeout -s KILL 30 /tmp/cb-denorm
for rep in 1 2; do
for shp in gate:5120:17408 down:17408:5120 gdn_qkv:5120:10240 gdn_out:6144:5120 attn_q:5120:12288 attn_o:6144:5120; do
  IFS=: read NM K N <<< "$shp"
  for M in 1 5 8; do for V in ${VS:-0 9 1 2 4 5 3}; do
    r=$(KB_IT=${KB_IT:-500} timeout -s KILL 60 $D/kb_cb${V}m$M $K $N $M 288 | head -1 | sed 's/.*: //')
    echo "R$rep $NM m=$M cb$V $r"
  done; done
done; done
