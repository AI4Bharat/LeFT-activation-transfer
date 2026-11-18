# Reproduction

Three paths are provided, in increasing order of cost. The first is sufficient
to verify the protocol.

---

## Path A — verification of the protocol (minutes, CPU, no datasets)

```bash
pip install -e ".[dev]"
pytest -q
```

The suite comprises 58 tests, which verify the claims rather than merely the
execution of the code: that the GELU initialisation matches the reported fit,
that K is configurable, that each of the four architectures holds exactly one
shared activation instance, that a frozen curve receives no update under an
AdamW step with weight decay, that an unfrozen curve does receive one, that
only coefficients are transferred from source to target, and that a
deliberately introduced source weight is detected. Five of the 58 compare the
fused CUDA kernel against the reference path and skip without a GPU, and four
recompute the figures quoted in README.md from `results/tables/`.

The transfer step may also be exercised in isolation, using the distributed
artifact and requiring no checkpoint:

```python
from left.activations import FAct
from left.models.mlp_mixer import build_mlpmixer
from left.transfer import assert_frozen

act = FAct.from_artifact("artifacts/fact/imagenet1k_vitd6_seed1.json")
model = build_mlpmixer("cifar10", act, embed_dim=76)
assert_frozen(model)          # no trainable activation parameters
print(act)
```

---

## Path B — reproduction of the target-network results (GPU-hours, no ImageNet)

This path reproduces the main table of the paper: 4 architectures × 4 datasets
× 11 activations × 5 seeds. **The source network is not required**, as the
transferred curve is distributed in
`artifacts/fact/imagenet1k_vitd6_seed1.json`.

```bash
pip install -e ".[train,dev]"
export LEFT_DATA_ROOT=/path/to/datasets
scripts/prepare_data.sh                    # CIFAR caches; FMNIST and Food-101 download on first use

# a single cell, to confirm the setup (minutes on a modern GPU)
ARCHS=vit DATASETS=cifar10 SEEDS=0 ACTS="fact_fixed gelu" scripts/train_targets.sh

# the full grid
scripts/train_targets.sh

# the tables
scripts/evaluate.sh
```

Each run writes `<run>_result.json`, containing the per-epoch training and
validation loss and accuracy, the learning-rate trace, the elapsed time and the
final test top-1 accuracy, together with two checkpoints: the best-validation
checkpoint and the final one. `scripts/evaluate.sh` traverses that tree and
reports the mean and sample standard deviation per activation, followed by the
paired, Holm-corrected comparison.

The following conditions are required in order to match the published
numbers.

* **fp32 throughout**, with no automatic mixed precision. A periodic
  activation evaluated in bf16 is not the same function as one evaluated in
  fp16, and the runs were executed on hosts with different fast formats; at
  approximately 100K parameters the cost of fp32 is negligible. Enabling AMP
  invalidates a comparison against the published numbers.
* **100 epochs on every dataset.** All four datasets and all four target
  architectures use the same budget, set in `configs/datasets/`. A shorter
  budget can invert a ranking, so it should not be altered when comparing
  against a published number.
* **The validation split is taken from the training set with a fixed seed
  (42)**, and checkpoint selection uses the best validation accuracy. The test
  set is evaluated once.
* **Fused kernel and reference path.** `fact_fixed` uses a fused CUDA kernel by
  default; `--no-cuda-kernel` selects the pure-PyTorch path. The forward
  numerics agree to fp32 rounding, as `tests/test_kernel.py` establishes on a
  GPU, but the execution times do not, so the two paths should not be combined
  within a single timing comparison.
* **Common and activation-specific configurations.** The main table uses the
  common configuration, in which every activation receives identical
  hyperparameters. The appendix additionally tunes the learning rate and weight
  decay per activation; tuning alters the margins and, in one combination, their
  sign, so the configuration should be stated for any reported number.

---

## Path C — re-learning of the curve (ImageNet-1K, several days)

This path is required only in order to determine whether the same curve is
learned again.

```bash
pip install -e ".[train,imagenet,dev]"
python -m left.training.source.download_imagenet1k --data-root "$LEFT_DATA_ROOT"   # ~150GB
scripts/train_source.sh 1        # seed 1, 100 epochs; then extracts the artifact
```

`scripts/train_source.sh` writes
`artifacts/fact/imagenet1k_vitd6_seed1_reproduced.json`, which may be compared
against the distributed artifact; the published seeds 1 to 5 agree to
approximately three decimal places on all five coefficients. Target runs are
then directed at the reproduced artifact:

```bash
ARTIFACT=artifacts/fact/imagenet1k_vitd6_seed1_reproduced.json scripts/train_targets.sh
```

`src/training/source/` contains the published source-training code, reduced to
the arms this paper reports (`standard`, `fact_k2_global`, `pau_global`) and
otherwise unmodified, including its flat imports, so that this path re-runs the
original implementation rather than a refactored version of it.

---

## Datasets

| Dataset       | Source                     | Size   | Prepared by                   |
| ------------- | -------------------------- | ------ | ----------------------------- |
| Fashion-MNIST | torchvision                | 30 MB  | automatic on first use        |
| CIFAR-10/100  | HuggingFace parquet        | 350 MB | `scripts/prepare_data.sh`     |
| Food-101      | torchvision (ETH Zurich)   | 5 GB   | automatic on first use        |
| ImageNet-1K   | HuggingFace (gated)        | 150 GB | `download_imagenet1k` (Path C)|

None of these datasets is redistributed in this repository. CIFAR is decoded
from the parquet release in stored row order, which differs from the order used
by torchvision; the latter would alter the validation split and every
augmentation draw without any visible indication. The cache builder verifies
SHA-256 against the tensors of the published runs.

## Hardware used

The target-scale runs (approximately 100–130K parameters) are accommodated by
any modern GPU and by considerably smaller devices. The ImageNet-1K source run
of Path C is the only one requiring substantial hardware.

## If a number does not reproduce

The following should be checked, in order: the epoch budget; whether automatic
mixed precision was enabled; whether the result corresponds to the common or the
activation-specific configuration; the dataset row order
(`python -m left.utils.build_dataset_cache --verify`); and the set of seeds.
These five account for substantially all observed discrepancies.
