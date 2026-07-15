#!/bin/bash
# Faithful A1 gate-2: a pretrained decoder-only GENERATIVE LLM (SmolLM2-360M) as the
# predictor over frozen ModernBERT features + random-init attribution control.
# batch 8 (reviewer-safe on 64GB MPS), 3k steps, seed 0. Copy floor reused from
# round-4 (predictor-independent). edit-weight 0 (ew10 only if it hazes, decided after).
set -e
cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa
export PYTHONHASHSEED=0
PY=.venv/bin/python
T="$PY -m models.gate2 --mode train --cache runs/gate2 --trunk genllm --edit-weight 0 --batch 8 --steps 3000"
B="$PY -m evals.dynamics --data data/v1 --adapter gate2 --input-regime clean --seeds 0,1,2,3,4"

echo "===== [1/4] TRAIN genllm pretrained ====="
$T --trunk-init pretrained --out runs/gate2/round5-genllm-pretrained
echo "===== [2/4] BATTERY genllm pretrained ====="
$B --ckpt runs/gate2/round5-genllm-pretrained/ckpt.pt \
   --out runs/gate2/round5-genllm-pretrained/battery-clean-hardened.json

echo "===== [3/4] TRAIN genllm random-init ====="
$T --trunk-init random --out runs/gate2/round5-genllm-randinit
echo "===== [4/4] BATTERY genllm random-init ====="
$B --ckpt runs/gate2/round5-genllm-randinit/ckpt.pt \
   --out runs/gate2/round5-genllm-randinit/battery-clean-hardened.json

echo "===== ROUND-5 GENLLM COMPLETE ====="
