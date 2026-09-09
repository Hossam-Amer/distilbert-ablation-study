#!/usr/bin/env bash
set -euo pipefail

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No NVIDIA GPU was found."
  exit 1
fi

for experiment in E1 E2 E3 E4; do
  for seed in 42 2026; do
    echo "Running ${experiment}, seed ${seed}"
    python train.py \
      --experiment "$experiment" --seed "$seed"
  done
done

python analyze.py
