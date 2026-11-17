"""The source-scale training code, as published.

These modules -- `cv_run`, `cv_train`, `cv_vit`, `cv_data`, `fourier_ffn`,
`fourier_layers` -- are the code that trained the paper's ImageNet-1K source
network (the depth-6, ~3M-parameter ViT whose learned curve every target run
transfers). They are reduced to the three FFN variants this paper reports --
`standard`, `fact_k2_global` and `pau_global` -- and otherwise unchanged,
including their flat intra-package imports, so that a reproduction of the
source run re-runs the published implementation rather than a refactor of it.
New work should prefer `left.activations.FAct` and
`left.training.train_target`; this subtree exists to be *re-run*, not extended.

The line below puts this directory on `sys.path` so the flat imports resolve
when the modules are reached as `left.training.source.*`.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
