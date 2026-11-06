#!/bin/bash
# Exactness: hostprof dumps (bsz 1 and 2), base vs sparse head K=32768, then cmp.py
M=/home/alex/code/exl3/models; HP=/home/alex/code/exl3/wt-q27host/rocm/host
A="-m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 --noprof --window 7"
export EXL3_SDH_PROFILE=/home/alex/code/exl3/prof/sdh_profile2.pt
for b in 1 2; do
  EXL3_SPARSE_DRAFT_HEAD=0 python $HP/hostprof.py $A --bsz $b --dump sdh_ex_base_b$b.json
  EXL3_SPARSE_DRAFT_HEAD=32768 python $HP/hostprof.py $A --bsz $b --dump sdh_ex_k32k_b$b.json
  python $HP/cmp.py sdh_ex_base_b$b.json sdh_ex_k32k_b$b.json
done
