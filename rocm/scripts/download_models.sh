#!/bin/bash
# Download the main model (text + MTP head + vision) and the DFlash2 draft model.
# usage: rocm/scripts/download_models.sh [models_dir]   (default: ./models)
set -euo pipefail
DIR=${1:-models}
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw --local-dir "$DIR/Qwen3.8-27B-EXL3-3.5bpw"
hf download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --local-dir "$DIR/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
# Two-card profile (config.two-card-kvpool.yml): 4.00bpw main model
hf download turboderp/Qwen3.8-27B-exl3 --revision 4.00bpw --local-dir "$DIR/Qwen3.8-27B-exl3-4.00bpw"
