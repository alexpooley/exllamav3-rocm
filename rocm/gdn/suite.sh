#!/bin/bash
# usage: suite.sh TAG [TREE]  -> prof/gdn/TAG-{np.json,np.log,b2.json,b2.log,qb.log}
G=/home/alex/code/exl3/prof/gdn; T=${2:-/home/alex/code/exl3/wt-q27gdn}; TAG=$1
H=/home/alex/code/exl3/prof/hp.sh
$H $T $G/$TAG-np.log $G/hostprof.py --window 7 --noprof --dump $G/$TAG-np.json
$H $T $G/$TAG-b2.log $G/hostprof.py --window 7 --noprof --bsz 2 --tokens 256 --dump $G/$TAG-b2.json
[ -n "$NOQB" ] || /home/alex/code/exl3/prof/qb.sh $T $G/$TAG-qb.log
echo done > $G/$TAG.done
