#!/usr/bin/env bash
# Step 3 -- TRANSFER, the full experiment grid: 4 architectures x 4 datasets x
# N seeds x the activation menu, under one common training configuration.
#
#   scripts/train_targets.sh                                  # everything
#   ARCHS=vit DATASETS=cifar10 SEEDS="0 1 2" scripts/train_targets.sh
#   ACTS="fact_fixed gelu" scripts/train_targets.sh            # the headline pair
#
# No source checkpoint is needed: the transferred curve comes from
# artifacts/fact/imagenet1k_vitd6_seed1.json. Every arm gets identical
# hyperparameters, so a ranking cannot be an artefact of tuning effort.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHS="${ARCHS:-vit swin linvit mlpmixer}"
DATASETS="${DATASETS:-fmnist cifar10 cifar100 food101}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ACTS="${ACTS:-fact_fixed gelu relu leaky_relu silu elu tanh mish softplus hardswish selu}"
ARTIFACT="${ARTIFACT:-artifacts/fact/imagenet1k_vitd6_seed1.json}"
OUT_ROOT="${OUT_ROOT:-results/logs}"
: "${LEFT_DATA_ROOT:=$HOME/datasets}"
export LEFT_DATA_ROOT

declare -A CONFIG=( [vit]=vit_d2 [swin]=swin_d2 [linvit]=linvit_d2 [mlpmixer]=mlp_mixer )
declare -A DSCONFIG=( [fmnist]=fashion_mnist [cifar10]=cifar10 [cifar100]=cifar100 [food101]=food101 )

for arch in $ARCHS; do
  for ds in $DATASETS; do
    for act in $ACTS; do
      for seed in $SEEDS; do
        out="$OUT_ROOT/$arch/$ds"
        mkdir -p "$out"
        echo "=== $arch / $ds / $act / seed $seed"
        python -m left.training.train_target \
            --config "configs/targets/${CONFIG[$arch]}.yaml" \
            --config "configs/datasets/${DSCONFIG[$ds]}.yaml" \
            --act "$act" --seed "$seed" \
            --artifact "$ARTIFACT" \
            --data-root "$LEFT_DATA_ROOT" \
            --out-dir "$out"
      done
    done
  done
done
