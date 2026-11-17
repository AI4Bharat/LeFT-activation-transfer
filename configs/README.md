# Configs

Each file specifies one configuration.
`python -m left.training.train_target --config <file> ...` merges it into the
defaults of the script, and explicit command-line flags take precedence. Keys
are validated against `left.utils.config.KEY_TO_DEST`, so that a misspelled
hyperparameter raises an error rather than being ignored.

```
configs/
  source/    step 1 (Learn): the ImageNet-1K run that learns the curve
  targets/   step 2 (Freeze and transfer): the four target architectures
  datasets/  per-dataset loader settings and epoch budget, included by the above
```

Two configurations appear in the paper, and both are reproducible from these
files.

* **Common configuration** (the `*.yaml` files as distributed): a single set of
  hyperparameters for every activation. This is the principal comparison; no
  activation is tuned individually, so a ranking cannot be attributed to
  differing tuning effort.
* **Activation-specific configuration**: `lr` and `wd` are searched per
  activation over lr ∈ {3e-4, 5e-4, 1e-3, 2e-3, 3e-3} × wd ∈ {0, 0.05, 0.2}.
  The values may be overridden on the command line, or a target configuration
  may be copied and its `optim` section edited. Both configurations are reported
  in the appendix.
