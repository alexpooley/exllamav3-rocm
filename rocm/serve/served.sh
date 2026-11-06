#!/bin/bash
# Served A/B on card 1: TabbyAPI with the production config (tabbyAPI/config.xtx.yml) and the serve.sh env, on a
# test port, with exllamav3 from TREE. Then oaibench (avg_short, long32k), batchbench (2 and 4 requests) and,
# unless NOEVAL=1, the evalq battery at the server's sampling preset (production: qwen38). NEEDLE=1 first sends
# a 113,000-token needle prompt (full 114,688 context incl. 1,024 output) and records peak card-1 VRAM. NOSPEED=1 skips the speed benches.
# usage: served.sh LABEL TREE [PORT]   results -> ~/code/exl3/serve-ab/LABEL.*
set -u
LBL=$1; T=$2; PORT=${3:-5821}
D=/home/alex/code/exl3; O=$D/serve-ab; HERE=$(cd $(dirname $0) && pwd); mkdir -p $O $D/batch
exec 9>/tmp/xtx-gpu1.lock; flock 9
exec 8>/tmp/xtx-build.lock; flock 8
held=$(env -u HIP_VISIBLE_DEVICES amd-smi process -g 1 2>/dev/null | awk '/PID:/ {print $2}')
[ -n "$held" ] && { echo "card 1 busy (pids $held)"; exit 75; }
export HIP_VISIBLE_DEVICES=1 PATH=$D/.venv/bin:$PATH PYTHONPATH=$T
export EXL3_NOGRAPH=mlp,gdn EXL3_EXPANDABLE_SEGMENTS=0 OMP_NUM_THREADS=6
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
# Same pool sizes as serve.sh, but a test path so production's pool is never touched.
export EXL3_KVPOOL=/dev/shm/xtx-kv-test EXL3_KVPOOL_GB=8 EXL3_RCPOOL_GB=4
echo "$(date -u +%T) $LBL tree $T $(git -C $T rev-parse --short HEAD) env: $(env | grep ^EXL3_ | sort | tr '\n' ' ')" | tee $O/$LBL.meta
cd $D/tabbyAPI
sed "s/^  port: .*/  port: $PORT/" config.xtx.yml > run/config-$PORT.yml
timeout -s KILL 10800 python main.py --config run/config-$PORT.yml > $O/$LBL.tabby.log 2>&1 &
PID=$!
for i in $(seq 300); do curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 $O/$LBL.tabby.log; exit 1; }; sleep 2; done
echo "$(date -u +%T) loaded" >> $O/$LBL.meta
B=http://127.0.0.1:$PORT; M=turboderp-Qwen3.8-27B-exl3-4.00bpw
echo "vram_after_load_mb $(env -u HIP_VISIBLE_DEVICES amd-smi metric -g 1 -m --csv | tail -1 | cut -d, -f3)" >> $O/$LBL.meta
if [ "${NEEDLE:-0}" = 1 ]; then
  ( while :; do env -u HIP_VISIBLE_DEVICES amd-smi metric -g 1 -m --csv 2>/dev/null | tail -1; sleep 1; done ) > $O/$LBL.vram.csv & VS=$!
  python3 $D/needle2.py $LBL-113k $B $M 113000 $D/models/$M/tokenizer.json >> $O/$LBL.needle.jsonl 2>>$O/$LBL.err
  kill $VS; echo "vram_peak_mb $(cut -d, -f3 $O/$LBL.vram.csv | sort -n | tail -1) of $(tail -1 $O/$LBL.vram.csv | cut -d, -f2)" >> $O/$LBL.meta
  kill -0 $PID 2>/dev/null || { echo "server died during needle" >> $O/$LBL.meta; exit 1; }
fi
# oaibench sends temperature 1.0 etc. explicitly (its historical setting); OAI_EXTRA adds the preset's presence_penalty
[ "${NOSPEED:-0}" = 1 ] || for r in 1 2; do OAI_EXTRA='{"temperature":0.7,"presence_penalty":1.0}' python3 $D/oaibench.py $LBL-r$r $B $M >> $O/$LBL.speed.jsonl 2>>$O/$LBL.err; done
[ "${NOSPEED:-0}" = 1 ] || python3 $D/batchbench.py $LBL $PORT 1 2 4 >> $O/$LBL.conc.jsonl 2>>$O/$LBL.err
echo "$(date -u +%T) speed done" >> $O/$LBL.meta
if [ "${NOEVAL:-0}" != 1 ]; then
  EVAL_SAMPLING=server python3 $HERE/evalq.py $LBL $B $M >> $O/$LBL.quality.jsonl 2>>$O/$LBL.err
  echo "$(date -u +%T) eval done" >> $O/$LBL.meta
fi
echo "rss_gb $(ps -o rss= -p $(pgrep -P $PID | head -1) | awk '{printf "%.1f", $1/1048576}')" >> $O/$LBL.meta
kill $PID; wait $PID 2>/dev/null
rm -rf /dev/shm/xtx-kv-test*
echo "$(date -u +%T) DONE" >> $O/$LBL.meta
