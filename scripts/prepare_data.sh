#!/usr/bin/env bash
# Datasets. Nothing here is committed to the repository -- this script fetches
# and caches what the training code reads.
#
#   LEFT_DATA_ROOT=/path/to/datasets scripts/prepare_data.sh
#
# * CIFAR-10/100 are decoded once into <root>/hf/c{10,100}_{train,test}_decoded.pt
#   from the HuggingFace parquet release, in stored row order, and checked
#   against the published runs' SHA-256 digests. Row order matters: it fixes the
#   validation carve-out and every augmentation draw, so a cache built from
#   torchvision's differently-ordered CIFAR would silently change every number.
# * Fashion-MNIST and Food-101 are pulled by torchvision on first use.
# * ImageNet-1K (source step only, ~150GB) has its own downloader, below.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${LEFT_DATA_ROOT:=$HOME/datasets}"
export LEFT_DATA_ROOT
echo "data root: $LEFT_DATA_ROOT"

python -m left.utils.build_dataset_cache --data-root "$LEFT_DATA_ROOT" "$@"
python -m left.utils.build_dataset_cache --data-root "$LEFT_DATA_ROOT" --verify

cat <<'MSG'

Fashion-MNIST and Food-101 download themselves on the first run that needs them.

ImageNet-1K is needed only to re-run the source network (step 1). The learned
curve is already in artifacts/fact/imagenet1k_vitd6_seed1.json, so skip this
unless you specifically want to re-learn it:

    python -m left.training.source.download_imagenet1k --data-root "$LEFT_DATA_ROOT"
MSG
