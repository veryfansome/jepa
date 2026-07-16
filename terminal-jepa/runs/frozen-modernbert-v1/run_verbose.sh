#!/bin/bash
set -e
cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa
export PYTHONHASHSEED=0
PY=.venv/bin/python
echo "===== verbose PRETRAINED ====="
$PY -m probes.frozen_probe --data data/v1 --verbose --seed 0 --out runs/frozen-modernbert-v1/probe-verbose.json
echo "===== verbose RANDOM-INIT (matched floor) ====="
$PY -m probes.frozen_probe --data data/v1 --verbose --random-init --seed 0 --out runs/frozen-modernbert-v1/probe-verbose-randominit.json
echo "===== VERBOSE PROBE COMPLETE ====="
