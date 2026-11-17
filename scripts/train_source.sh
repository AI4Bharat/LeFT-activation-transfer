#!/usr/bin/env bash
# Step 1 -- LEARN. Train the source network with one learnable, globally shared
# FAct curve on ImageNet-1K, then extract the curve it converged to.
#
#   scripts/train_source.sh [seed]            # default seed 1, the paper's
#
# Cost: ~100 epochs of ImageNet-1K. Most readers should skip this entirely --
# its only output that the rest of the pipeline consumes is the five-number
# artifact, which ships with the repository. Re-run it to test whether the same
# curve is learned again (the published seeds 1-5 agree to ~3 decimals).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1}"
OUT="${OUT:-results/logs/source}"
: "${LEFT_DATA_ROOT:=$HOME/datasets}"
export LEFT_DATA_ROOT

mkdir -p "$OUT"
python -m left.training.train_source \
    --dataset imagenet1k \
    --variants fact_k2_global \
    --seeds "$SEED" \
    --epochs 100 \
    --data-root "$LEFT_DATA_ROOT" \
    --out-dir "$OUT"

# Step 2 -- FREEZE: pull the learned coefficients out as an artifact. The
# published artifact was extracted from the best-validation checkpoint, so this
# prefers that file and falls back to the last epoch.
CKPT="$OUT/checkpoints/cv_imagenet1k_fact_k2_global_seed${SEED}_best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/checkpoints/cv_imagenet1k_fact_k2_global_seed${SEED}_final.pt"
if [[ -f "$CKPT" ]]; then
    python -m left.transfer.extract_cli \
        --checkpoint "$CKPT" \
        --out "artifacts/fact/imagenet1k_vitd6_seed${SEED}_reproduced.json" \
        --note "reproduced by scripts/train_source.sh"
else
    echo "no checkpoint found under $OUT/checkpoints; extract manually with" \
         "python -m left.transfer.extract_cli --checkpoint <ckpt> --out <json>"
fi
