#!/usr/bin/env bash
# Reproduce the paper's tables from a results tree: mean +- sd over seeds, and
# the transferred arm against each baseline, paired across seeds and
# Holm-corrected over the four datasets.
#
#   scripts/evaluate.sh                      # all four architectures
#   scripts/evaluate.sh vit gelu             # one arch, one baseline
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS="${1:-vit swin linvit mlpmixer}"
BASELINE="${2:-gelu}"
RESULTS="${RESULTS:-results/logs}"

for arch in $ARCHS; do
  echo "================ $arch"
  python -m left.evaluation.evaluate \
      --aggregate "$RESULTS" --arch "$arch" \
      --arm fact_fixed --baseline "$BASELINE"
done
