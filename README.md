# LeFT — Learn, Freeze, Transfer

**LeFT learns a globally shared activation function inside a source network,
freezes it, and transfers it to independently initialized target networks
without transferring any source-network weights.**

FAct, a five-parameter truncated Fourier series, is the first activation
implemented in the framework. A single curve is learned jointly with a depth-6,
approximately 3M-parameter ViT on ImageNet-1K, frozen, and installed in four
independently initialized target architectures as a parameter-free activation.
Under a common training configuration it attains the highest mean test accuracy
among eleven activation choices in **15 of 16** architecture × dataset
combinations, by up to **1.71** percentage points.

Paper: *Learn, Freeze, and Transfer an Activation Function Across
Architectures*.

## Methodology

| Step         | Description                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Learn**    | A single activation module is installed at every activation site of the source network, so that one curve is shared across neurons and across depth, and is optimized jointly with the weights. |
| **Freeze**   | The learned coefficients are held fixed: they receive no gradient, are excluded from the optimizer, and are exempt from weight decay. The activation therefore contributes **no trainable parameters**. |
| **Transfer** | The frozen curve is installed in an independently initialized target network as a fixed activation. No other quantity is shared with the source network. |

## FAct

```
φ(t) = a₀ + Σ_{k=1..K} [ a_k cos(k ω t) + b_k sin(k ω t) ]
```

At K = 2 the activation consists of five scalars for an entire network. The
coefficients are initialized at the truncated Fourier fit of GELU, so that the
network initially computes a close approximation of the GELU baseline and any
subsequent difference is attributable to the learning of the curve. The curve
transferred in the paper is

```
θ* = (a₀, a₁, a₂, b₁, b₂) = (0.2182, 0.1193, −0.4237, 0.8315, −0.5918)
```

extracted from the seed-1 ImageNet-1K source checkpoint and distributed as
[`artifacts/fact/imagenet1k_vitd6_seed1.json`](artifacts/fact/imagenet1k_vitd6_seed1.json).

## Architecture support

| Target        | Module                  | Attention       | Parameters        | Config                           |
| ------------- | ----------------------- | --------------- | ----------------- | -------------------------------- |
| **ViT**       | `left.models.vit`       | softmax, global | 105,098 – 123,237 | `configs/targets/vit_d2.yaml`     |
| **Swin**      | `left.models.swin`      | shifted-window  | 102,354 – 110,317 | `configs/targets/swin_d2.yaml`    |
| **LinViT**    | `left.models.linvit`    | elu+1 linear    | 105,098 – 123,237 | `configs/targets/linvit_d2.yaml`  |
| **MLP-Mixer** | `left.models.mlp_mixer` | none            | 103,620 – 126,313 | `configs/targets/mlp_mixer.yaml`  |

The range is over the four datasets, which differ in input channels, resolution
and class count; every activation is compared at the same parameter count on a
given architecture and dataset, since the frozen curve adds none.

Datasets: Fashion-MNIST, CIFAR-10, CIFAR-100 and Food-101. Baselines: ReLU,
LeakyReLU, GELU, SiLU, ELU, Tanh, Mish, Softplus, HardSwish and SELU, each at
its library default.

## Installation

```bash
pip install -e ".[dev]"                  # activation, transfer and tests (CPU is sufficient)
pip install -e ".[train,dev]"            # and target-network training
pip install -e ".[train,imagenet,dev]"   # and re-training of the ImageNet-1K source
# alternatively: conda env create -f environment.yml && conda activate left

pytest -q                                # 58 tests; no datasets required (5 need a GPU and skip without one)
```

`ninja` is required only for the fused CUDA kernels; the pure-PyTorch path
(`--no-cuda-kernel`) is numerically identical.

## Dataset preparation

```bash
export LEFT_DATA_ROOT=/path/to/datasets
scripts/prepare_data.sh
```

No dataset is redistributed in this repository.

* **Fashion-MNIST** and **Food-101** are downloaded on first use.
* **CIFAR-10** and **CIFAR-100** are decoded once from the HuggingFace parquet
  release in stored row order and verified against the SHA-256 digests of the
  published runs.
* **ImageNet-1K** is required only in order to re-learn the curve.

## Training

Step 1, **Learn** (ImageNet-1K; several days). This step may be omitted, as the
only output consumed by the remainder of the pipeline is distributed in
`artifacts/`.

```bash
scripts/train_source.sh 1          # seed 1, 100 epochs, followed by extraction of the curve
```

Step 2, **Freeze and transfer** (no source checkpoint required):

```bash
ARCHS=vit DATASETS=cifar10 SEEDS=0 ACTS="fact_fixed gelu" scripts/train_targets.sh
scripts/train_targets.sh           # the complete grid
```

All arms use identical hyperparameters. The recipes in `configs/` are consumed
directly: `--config` merges a YAML file into the defaults of the script,
explicit flags take precedence over it, and an unrecognized key raises an error.

## Transfer

```bash
python -m left.transfer.extract_cli --checkpoint <source-ckpt> --out <artifact.json>
scripts/transfer.sh <source-ckpt>    # extraction, installation and verification
```

```python
from left.activations import FAct
from left.models.mlp_mixer import build_mlpmixer
from left.transfer import assert_frozen, verify_weight_independence

act = FAct.from_artifact("artifacts/fact/imagenet1k_vitd6_seed1.json")  # frozen
model = build_mlpmixer("cifar10", act, embed_dim=76)   # one instance at every site

assert_frozen(model)                                   # no trainable activation parameters
verify_weight_independence(model, "…source_final.pt")  # no weight was transferred
```

## Evaluation

```bash
scripts/evaluate.sh
```

Results are reported as means over seeds with sample standard deviations, and
comparisons between two activations as seed-matched mean differences.

## Pretrained artifacts

| Artifact                                 | Location                                              |
| ---------------------------------------- | ----------------------------------------------------- |
| Learned FAct coefficients (5 scalars)    | this repository — `artifacts/fact/*.json`             |
| Checkpoints behind every reported number | Hugging Face `arun-AiBharat/LeFT-activation-transfer` |
| Aggregated per-seed results              | this repository — `results/tables/*.json`             |
| Raw datasets                             | official sources (`scripts/prepare_data.sh`)          |

The coefficient artifact is the only model artifact retained in version control:
five scalars, and the sole input required to reproduce every transfer experiment
without the source network.

## Layout

```
src/activations/   FAct (configurable K), PAU, the GELU-fit quadratures and the
                   eleven activations under comparison, together with the
                   published implementations verbatim, including the fused CUDA
                   kernel
src/models/        the four target architectures
src/training/      train_source (step 1), train_target (step 2) and run_zoo;
                   training/source/ is the published source-scale code, vendored
                   unchanged so that a source reproduction re-runs the original
src/transfer/      extraction, installation, freezing and verification
src/evaluation/    test-set evaluation and the seed statistics
src/utils/         seeding, checkpoints, the dataset pipeline, the config loader
configs/           source, target and dataset recipes
artifacts/fact/    the learned coefficients
tests/             the protocol: sharing, freezing and weight independence
results/tables/    the per-seed numbers reported in the manuscript
notebooks/         the learned curve and the results tables
docs/              methodology.md, reproduction.md
```

## License

MIT — see [`LICENSE`](LICENSE).
