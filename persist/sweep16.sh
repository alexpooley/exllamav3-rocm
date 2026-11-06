echo "== chain 2"; timeout -s KILL 150 python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid 288 --chain 2 --layers 3,7,11,15 --trace 2>&1 | grep "per MLP\|bit-exact\|  L"
echo "== stress"; timeout -s KILL 400 python pm_bench.py -m ~/code/exl3/models/turboderp-Qwen3.8-27B-exl3-4.00bpw --grid 288 --layers 3 --stress 30000 2>&1 | grep "stress\|bit-exact"
