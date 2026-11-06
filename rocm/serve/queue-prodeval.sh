#!/bin/bash
# Re-run of the production-build battery only (queue.sh first pass called evalq from the wrong tree). Runs after queue-dh4.sh.
O=/home/alex/code/exl3/serve-ab; until grep -q "DH4 QUEUE DONE" $O/queue.log 2>/dev/null; do sleep 30; done
NOSPEED=1 /home/alex/code/exl3/wt-q27serve/rocm/serve/served.sh prod-eval /home/alex/code/exl3/phoenixhaxor_exllamav3-rocm
echo "$(date -u +%T) PRODEVAL DONE" >> $O/queue.log
