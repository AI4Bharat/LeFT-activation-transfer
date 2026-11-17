# Artifacts

An artifact is the unit of transfer. Each file holds the coefficients of a
learned activation together with the provenance required to establish their
origin.

```
artifacts/fact/imagenet1k_vitd6_seed1.json
```

is the curve used by every target-network result in the paper: five scalars
learned inside a depth-6, approximately 3M-parameter ViT over 100 epochs of
ImageNet-1K (seed 1, source top-1 65.12%).

```
theta* = (a0, a1, a2, b1, b2) = (0.2182, 0.1193, -0.4237, 0.8315, -0.5918)
```

The file additionally records the harmonic count, the base frequency, the GELU
fit at which the run was initialised, the name of the source checkpoint, and the
coefficients to which all five published seeds converged; these agree to
approximately three decimal places, so the learned curve is reproducible rather
than specific to a single seed.

The artifact is tracked in version control deliberately: `.gitignore` excludes
checkpoints but admits `artifacts/**/*.json`. It is the sole input required to
reproduce every transfer experiment without obtaining or training the source
network.

## Use

```python
from left.activations import FAct
act = FAct.from_artifact("artifacts/fact/imagenet1k_vitd6_seed1.json")  # frozen
```

or, within a training run:

```bash
python -m left.training.train_target --act fact_fixed \
    --artifact artifacts/fact/imagenet1k_vitd6_seed1.json ...
```

## Creation

```bash
python -m left.transfer.extract_cli --checkpoint <source-ckpt> --out <artifact.json>
```

The extractor reads every activation site in the checkpoint and writes an
artifact only if all sites agree; an artifact therefore also constitutes
evidence that the source curve was globally shared.

## Model weights

Checkpoints are not stored in this repository. The checkpoints behind every
reported number, including the source networks from which these coefficients
were extracted, are available on Hugging Face
(`arun-AiBharat/LeFT-activation-transfer`); no component of this repository
requires them.
