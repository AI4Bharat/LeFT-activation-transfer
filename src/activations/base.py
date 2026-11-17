"""The contract a LeFT-transferable activation must satisfy.

This is the seam that keeps the framework independent of any one activation
family. `left.transfer` only ever calls the five methods below, so a new
globally shared nonlinearity (spline, rational, polynomial, ...) joins LeFT by
subclassing `SharedActivation` -- no change to the transfer protocol.

Two invariants every implementation must honour:

1. **Global sharing.** One module instance is installed at every activation
   site in the network, so the parameters are shared *by reference*. The model
   builders in `left.models` take an activation *instance* (not a factory) for
   exactly this reason; `tests/test_sharing.py` checks it.
2. **A frozen activation has zero trainable parameters.** `freeze()` must make
   the coefficients invisible to the optimizer, so a frozen arm is capacity-
   identical to a conventional fixed activation and weight decay cannot drift
   the curve. `tests/test_transfer.py` checks it.
"""
from abc import ABC, abstractmethod

import torch.nn as nn


class SharedActivation(nn.Module, ABC):
    """A parameterised nonlinearity whose parameters can be extracted from one
    network and installed in another."""

    #: Short name used in coefficient artifacts and run names.
    family = "abstract"

    @abstractmethod
    def coefficients(self):
        """Return a JSON-serialisable dict fully describing the curve.

        Must round-trip through `load_coefficients`. Should contain whatever a
        reader needs to re-create the function exactly -- for FAct: K, w, a0,
        a, b.
        """

    @abstractmethod
    def load_coefficients(self, state, strict=True):
        """Install the curve described by `state` (as returned by
        `coefficients`). With `strict`, raise on any structural mismatch
        (different K, different family) rather than silently reinterpreting."""

    @abstractmethod
    def freeze(self):
        """Pin the coefficients: no gradients, no optimizer state, no weight
        decay. Returns self."""

    @abstractmethod
    def unfreeze(self):
        """Make the coefficients learnable again. Returns self."""

    @property
    def frozen(self):
        """True if no coefficient currently requires grad."""
        return not any(p.requires_grad for p in self.parameters(recurse=False))

    def num_activation_parameters(self, trainable_only=True):
        """How many scalars this activation contributes to the model. A frozen
        LeFT activation returns 0 with `trainable_only=True` -- that is the
        equal-capacity guarantee the comparisons rest on."""
        ps = self.parameters(recurse=False)
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)
