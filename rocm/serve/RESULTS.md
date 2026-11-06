# q27-serve: A/B against production (xtx-kvpool b001056), card 1, 2026-11-06

All runs on xtx card 1 (HIP_VISIBLE_DEVICES=1), production env (serve.sh: EXL3_NOGRAPH=mlp,gdn,
EXL3_EXPANDABLE_SEGMENTS=0, OMP_NUM_THREADS=6, MALLOC_* thresholds). Raw logs and dumps in `results/`.

Builds:
- **prod**: `~/code/exl3/phoenixhaxor_exllamav3-rocm`, branch xtx-kvpool b001056 (production ext).
- **serve**: this branch. q27-gdn 4630ba6 (GEMV c5072bf + early accept/pre-draft be210a5 + GDN log replay a297ad9),
  plus 88c379a (hostprof.py only; cherry-picked as c410eeb). Ext built in place with `build.sh`.
- **serve+dh4**: serve plus merge of q27-dh4 09340f3 (Python only, no rebuild), with
  `EXL3_DRAFT_HEAD=~/code/exl3/models/work-dh4/lm_head-4bpw.safetensors`.

Defaults in serve: EXL3_EARLY_ACCEPT=1, EXL3_PREDRAFT=1, EXL3_GDN_REPLAY=1 (all "on unless 0"); the GEMV change has
no switch. exact.py prints `early_accept True predraft True early_accept_rounds 672` for serve.

## Exactness (greedy, DFlash2 window 7, Q8 cache, 512 tokens; `exact.py`, `cmp_exact.py`)

| prompt | serve vs prod | serve after dh4 merge, head off | serve+dh4 |
|---|---|---|---|
| code | identical | identical | identical |
| prose | identical | identical | **differs at token 308** |
| json | identical | identical | identical |
| long (30,300-token prompt) | identical | identical | identical |
| batch 2: code + json | identical | identical | identical |

Served at temperature 0 (batchbench, 1/2/4 requests): serve output hashes = prod for every request.
serve+dh4: the python prompt differs at every concurrency (hash 0c8488c4 vs 4be50336 / c7e41d8b).

dh4 prose divergence: prod "Digital formats are subject to obsolescence", dh4 "Digital formats change with alarming
frequency" (a near-tie). Control (`winprobe.sh`): prod with window 5 and 6 gives the same tokens as window 7 on prose
and code, so the round shape alone does not cause it. Cause not found.

## Speed, in process (`q27_bench.py`, greedy, 3 prompts x 512, 2 repeats, same session)

| build | fixed-7 tok/s | ms/round | default ("adapt-7") | EXL3_ADAPT_WINDOW=1 |
|---|---|---|---|---|
| prod | 122.9 / 122.4 | 33.14 / 33.29 | 123.1 / 122.8 | 119.3 / 122.6 |
| serve | 129.7 / 129.8 | 31.41 / 31.40 | 130.1 / 130.2 | 130.8 / 130.8 |
| serve (dh4 merged, off) | 129.6 / 129.7 | 31.45 / 31.41 | 130.1 / 130.2 | - |
| serve+dh4 | 129.5 / 129.7 | 31.29 / 31.25 | 131.6 / 131.6 | - |

serve: +5.6% tok/s, -1.8 ms/round. The default generator does not adapt the window (EXL3_ADAPT_WINDOW unset = off).

## Served (TabbyAPI f07131c, `tabbyAPI/config.xtx.yml` unchanged, port 5821, `served.sh`)

Sampling at the production preset (qwen38: temp 0.7, presence 1.0, top_k 20, top_p 0.95, min_p 0.05; reasoning low).
Shared KV pool on, at a test path (`/dev/shm/xtx-kv-test`).

| | prod | serve | serve+dh4 |
|---|---|---|---|
| oaibench avg_short, run 1 / 2 | 130.2 / 134.6 | 139.4 / 134.6 | 136.1 / 144.9 |
| oaibench long32k tok/s, run 1 / 2 | 114.1 / 121.2 | 134.5 / 140.1 | 124.5 / 127.6 |
| batchbench temp 0, 1 request (4 prompts) | 111.7 | 117.9 | 119.2 |
| 2 at once, total | 137.8 | 139.4 | 138.5 |
| 4 at once, total (first token max) | 149.2 (17.7 s) | 148.2 (18.0 s) | 147.4 (17.9 s) |
| needle 113,076 tokens (full 114,688 ctx) | 3/3, 127.1 s | - | 3/3, 127.4 s |
| card-1 VRAM peak at that prompt (sysfs) | 25,285 / 25,753 MB (98.2%) | - | **25,743 / 25,753 MB (99.96%)** |
| evalq battery: GSM8K / HumanEval / MMLU-Pro / tools | 95.3 / 99.4 / 78.0 / 100 | 97.3 / 97.6 / 80.0 / 100 | not run |
| battery output tokens, wall time | 327k, 40.2 min | 320k, 36.9 min | |

- Full battery (150 GSM8K, 164 HumanEval, 150 MMLU-Pro, 20 tools), not a subset. Differences are 3 items or fewer per
  task, inside sampling noise for one run each.
- At 2 and 4 requests serve gives no gain. Not checked: why. Lead: at batch 2 the verify forward has up to 16 rows,
  so the RDNA3 matmul selects its MR 12/16 kernels (exl3_rdna3.cu select_kernel); c5072bf was measured at 1-8 rows.
- avg_short is sampled, so run-to-run spread is +-4 tok/s; batchbench at temperature 0 is the cleaner single-stream number.

## Observations

- prod RSS after its battery: 17.5 GB (the vault calls > 9 GB a leak). serve's RSS was not recorded (script bug, fixed).
- amd-smi prints nothing when HIP_VISIBLE_DEVICES is set and gave empty output inside served.sh; VRAM was read from
  `/sys/class/drm/card2/device/mem_info_vram_used` (card2 = PCI 0b:00.0 = GPU 1).
