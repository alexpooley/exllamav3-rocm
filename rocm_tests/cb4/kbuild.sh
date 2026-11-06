#!/bin/bash
T=/home/alex/code/exl3/wt-q27cb
for V in 0 1 2 3 4 5 9; do for MR in 1 5 8; do
  /home/alex/code/exl3/prof/kbc.sh $T cb${V}m$MR 4 $MR -DEXL3_CB4=$V 2>&1 | tr '\n' ' '; echo
done; done
