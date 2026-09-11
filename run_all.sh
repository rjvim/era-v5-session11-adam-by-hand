#!/usr/bin/env bash
# Reproduce every artifact and figure, then verify the README against them.
set -euo pipefail
cd "$(dirname "$0")/src"
python3 task1_adam_by_hand.py
python3 task2_bias_correction.py
python3 task3_update_weight_ratio.py
python3 task4_cosine_vs_wsd.py
python3 task5_lr_width_sweep.py
cd ..
python3 tests/test_invariants.py
