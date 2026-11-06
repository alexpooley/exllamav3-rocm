#!/bin/bash
# Acceptance cost of a low-bit drafter head: full-vocab subset (K = vocab), fp16 vs qdq 4 / 3 bits
M=/home/alex/code/exl3/models
A="-m $M/turboderp-Qwen3.8-27B-exl3-4.00bpw -dm $M/own-DFlash2-EXL3-5.0bpw -cs 16384 --reps 1 --tokens 512"
for q in 4 3; do
  echo "== EXL3_SDH_QDQ=$q"
  EXL3_SDH_QDQ=$q python sdh_sweep.py $A --ks 248320
done
