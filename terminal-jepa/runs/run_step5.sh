#!/bin/bash
# Step 5a: gate-1 architecture-matched random-init ModernBERT floor.
# Step 5b: reconstruction twin (4k, matching sigreg arms) + protocol-v2 probe.
set -e
cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa
export PYTHONHASHSEED=0
PY=.venv/bin/python

echo "===== [5a] frozen_probe --random-init (architecture-matched floor) ====="
$PY -m probes.frozen_probe --data data/v1 --random-init --seed 0 \
    --out runs/frozen-modernbert-v1/probe-randominit.json

echo "===== [5b-train] recon twin, 4k steps ====="
$PY -m train.train --data data/v1 --arm recon --steps 4000 --batch 32 --seed 0 \
    --out runs/recon-v1-s0-4k

echo "===== [5b-probe] protocol-v2 probe of the recon encoder (held-out) ====="
$PY -m probes.probe --data data/v1 --ckpt runs/recon-v1-s0-4k/ckpt.pt \
    --seed 0 --out runs/recon-v1-s0-4k/probe-v2.json

echo "===== STEP 5 COMPLETE ====="
