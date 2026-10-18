#!/bin/bash
# Start one TabbyAPI per GPU, all sharing one KV pool in /dev/shm (generator/kvpool.py).
# usage: rocm/scripts/run_two_cards.sh [config.yml] [tabby_dir] [first_port]
# Card N listens on first_port + N. Put a router (for example llama-swap) in front to spread requests.
# The environment from setup_env.sh must be active.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${1:-config.two-card-kvpool.yml}
TABBY=$(cd "${2:-$REPO/../tabbyAPI}" && pwd)
PORT0=${3:-5800}
CARDS=${CARDS:-0 1}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export EXL3_NOGRAPH=${EXL3_NOGRAPH-mlp,gdn}
export EXL3_EXPANDABLE_SEGMENTS=0          # expandable segments fail with hsa_amd_vmem_address_reserve on some ROCm builds
# glibc: return freed 3 MiB recurrent checkpoints to the OS. Without this the heap fragments and each
# instance grows to 15+ GB RSS, which starves RAM and makes prefill several times slower.
export MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=67108864
# The shared pool: 8 GB of KV pages + 4 GB of recurrent checkpoints in tmpfs. Unset EXL3_KVPOOL to turn off.
export EXL3_KVPOOL=${EXL3_KVPOOL-/dev/shm/exl3-kv} EXL3_KVPOOL_GB=${EXL3_KVPOOL_GB:-8} EXL3_RCPOOL_GB=${EXL3_RCPOOL_GB:-4}
cd "$TABBY"
mkdir -p run
pids=()
for c in $CARDS; do
  p=$((PORT0 + c))
  sed "s/^  port: .*/  port: $p/" "$CONFIG" > run/config-$p.yml   # TabbyAPI ignores --port when the config has one
  HIP_VISIBLE_DEVICES=$c python main.py --config run/config-$p.yml > run/tabby-$p.log 2>&1 &
  pids+=($!)
  echo "card $c -> port $p (log $TABBY/run/tabby-$p.log)"
done
trap "kill ${pids[*]} 2>/dev/null" INT TERM
wait
