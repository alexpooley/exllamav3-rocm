#!/bin/bash
# Is greedy output on the production build window-dependent? (explains the dh4 prose diff if yes)
R=/home/alex/code/exl3/wt-q27serve/rocm/serve; P=/home/alex/code/exl3/phoenixhaxor_exllamav3-rocm; O=/home/alex/code/exl3/serve-ab
for w in 5 6; do $R/run1.sh $P $O/exact-prod-w$w.log 900 $R/exact.py -cs 40960 -cq 8 --only prose,code --window $w --dump $O/exact-prod-w$w.json; done
