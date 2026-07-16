#!/bin/bash
set -e
cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa
export PYTHONHASHSEED=0
PY=.venv/bin/python
echo "===== WM (5 seeds, + predict-mean baseline) ====="
$PY -m realenv.worldmodel --data data/real --seeds 0,1,2,3,4 --steps 3000 --out runs/real/r3-conclusive.json
echo "===== JEPA-vs-gen (10 seeds) ====="
$PY -m realenv.jepa_vs_gen --data data/real --seeds 0,1,2,3,4,5,6,7,8,9 --steps 3000 --out runs/real/r3-vs-gen-10seed.json
echo "===== CONCLUSIVE DONE ====="
