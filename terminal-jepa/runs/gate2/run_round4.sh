#!/bin/bash
# Step 4b + contingency: complete the round-4 A1 gate-2 2x2 (init x loss-weight).
# pretrained-ew0 already done (hazed). This adds: feature-space copy floor,
# randinit-ew0, and ew10 variants of both arms (handoff's documented middle,
# since ew0 hazes). All batteries: --adapter gate2, feature-space L2, clean input,
# 5 seeds, hardened referee.
set -e
cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa
export PYTHONHASHSEED=0
PY=.venv/bin/python
G="$PY -m models.gate2 --mode train --cache runs/gate2 --trunk pretrained"
B="$PY -m evals.dynamics --data data/v1 --adapter gate2 --input-regime clean --seeds 0,1,2,3,4"

echo "===== [1/7] feature-space COPY FLOOR (5 seeds) ====="
$PY -m evals.dynamics --data data/v1 --adapter gate2-copy \
    --ckpt runs/gate2/round4-pretrained/ckpt.pt --input-regime clean --seeds 0,1,2,3,4 \
    --out runs/gate2/round4-copyfloor-hardened.json

echo "===== [2/7] TRAIN randinit ew0 ====="
$G --trunk-init random --edit-weight 0 --out runs/gate2/round4-randinit
echo "===== [3/7] BATTERY randinit ew0 ====="
$B --ckpt runs/gate2/round4-randinit/ckpt.pt --out runs/gate2/round4-randinit/battery-clean-hardened.json

echo "===== [4/7] TRAIN pretrained ew10 ====="
$G --trunk-init pretrained --edit-weight 10 --out runs/gate2/round4-pretrained-ew10
echo "===== [5/7] BATTERY pretrained ew10 ====="
$B --ckpt runs/gate2/round4-pretrained-ew10/ckpt.pt --out runs/gate2/round4-pretrained-ew10/battery-clean-hardened.json

echo "===== [6/7] TRAIN randinit ew10 ====="
$G --trunk-init random --edit-weight 10 --out runs/gate2/round4-randinit-ew10
echo "===== [7/7] BATTERY randinit ew10 ====="
$B --ckpt runs/gate2/round4-randinit-ew10/ckpt.pt --out runs/gate2/round4-randinit-ew10/battery-clean-hardened.json

echo "===== ROUND-4 2x2 COMPLETE ====="
