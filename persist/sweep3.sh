for g in 192 288; do for pf in 0 1; do echo "== grid $g pf $pf"; python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid $g --prefetch $pf 2>&1 | grep "per MLP\|bit-exact False"; done; done
python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid 288 --trace --layers 3 2>&1 | grep "  L0\|units per"
