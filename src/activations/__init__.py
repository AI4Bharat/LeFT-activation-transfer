"""Activation implementations that satisfy the LeFT transfer contract.

* `base.SharedActivation` -- the contract itself.
* `fact.FAct`             -- truncated Fourier series, configurable K.
* `pade.PAU`              -- Pade rational function, the non-Fourier control.
* `zoo`                   -- the parameter-free comparison menu (the
  transferred curve plus ten conventional activations), used by the trainers.

The modules `fact_fixed`, `fact_k2_global_trainable` and
`fact_k2_global_frozen` are the paper's published runs verbatim (including the
fused CUDA kernel); see `fact.py`'s docstring.
"""
from left.activations.base import SharedActivation  # noqa: F401
from left.activations.fact import FAct, gelu_init_coefficients  # noqa: F401
from left.activations.pade import PAU  # noqa: F401

__all__ = ["SharedActivation", "FAct", "PAU", "gelu_init_coefficients"]
