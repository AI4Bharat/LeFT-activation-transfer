"""Training entry points.

* `train_source` -- step 1 (Learn): the source network with a learnable shared
  activation. Needs ImageNet-1K; delegates to the vendored published launcher.
* `train_target` -- steps 2-3 applied (Freeze, Transfer): an independently
  initialised target network with the activation installed as a fixed
  function. One run = one (architecture, dataset, activation, seed).
* `run_zoo` -- sweeps `train_target` over the activation menu and the seeds.
"""
