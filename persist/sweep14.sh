export PM_VARIANT=_p1 PM_FLAGS=-DPM_PRIO=1
for o in 0 1; do for g in 192 288; do echo "== prio1 order $o grid $g"; timeout -s KILL 120 python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid $g --order $o --layers 3,7,11,15 2>&1 | grep "per MLP"; done; done
timeout -s KILL 120 python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid 288 --order 1 --layers 3 --trace 2>&1 | grep "  L0"
