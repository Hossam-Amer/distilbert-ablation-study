#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results

nohup bash -lc '. .venv/bin/activate && exec bash run.sh' \
  > results/suite.log 2>&1 < /dev/null &

echo "$!" > results/suite.pid
echo "Started training with PID $!"
