"""Freeze -- step 2 of Learn, Freeze, Transfer.

Freezing is not a detail of the optimizer; it is what the scientific claim
rests on. A LeFT target run must be able to say: the activation installed here
is a *fixed* function, identical to the one the source network converged to,
and the target network enjoyed no extra capacity for having it. These helpers
enforce and verify exactly that, for any `SharedActivation`.
"""
import copy

import torch
import torch.nn as nn

from left.activations.base import SharedActivation


def freeze_activation(module):
    """Freeze every `SharedActivation` reachable from `module` (or the
    activation itself). Returns the list of frozen activation instances --
    de-duplicated by identity, so a globally shared curve appears once."""
    acts = find_activations(module)
    for a in acts:
        a.freeze()
    return acts


def find_activations(module):
    """Every distinct `SharedActivation` instance in `module`, de-duplicated by
    object identity. Under correct global sharing this has length 1 however
    deep the network is -- that is what `tests/test_sharing.py` asserts."""
    if isinstance(module, SharedActivation):
        return [module]
    seen, out = set(), []
    for m in module.modules():
        if isinstance(m, SharedActivation) and id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def activation_parameter_count(module, trainable_only=True):
    """Activation scalars contributed to `module`, counting a shared curve
    once. A correctly frozen LeFT arm returns 0."""
    return sum(a.num_activation_parameters(trainable_only=trainable_only)
               for a in find_activations(module))


def assert_frozen(module):
    """Raise unless every activation in `module` is frozen and contributes no
    trainable scalar. Call this once after building a target model."""
    acts = find_activations(module)
    if not acts:
        raise AssertionError("no SharedActivation found; nothing to freeze. "
                             "A conventional activation (GELU, ReLU) is "
                             "already fixed and needs no freezing.")
    for a in acts:
        if not a.frozen:
            raise AssertionError(f"activation {a} is not frozen")
    n = activation_parameter_count(module, trainable_only=True)
    if n != 0:
        raise AssertionError(f"{n} activation parameters are still trainable")
    return True


def optimizer_sees_no_activation(optimizer, module):
    """Raise if any activation tensor of `module` is in `optimizer`'s param
    groups. requires_grad=False is usually enough, but an optimizer built
    from `model.parameters()` *before* freezing would still hold references
    and apply weight decay to them -- a silent way to drift a 'frozen' curve.
    """
    act_ids = {id(p) for a in find_activations(module)
               for p in a.parameters(recurse=False)}
    for g in optimizer.param_groups:
        for p in g["params"]:
            if id(p) in act_ids:
                raise AssertionError(
                    "the optimizer holds an activation parameter; build the "
                    "optimizer from [p for p in model.parameters() if "
                    "p.requires_grad] AFTER freezing the activation")
    return True


class ActivationWatch:
    """Context manager that snapshots the activation coefficients on entry and
    verifies on exit that nothing moved.

    Wrap a training run (or a single step, in a test) in it:

        >>> with ActivationWatch(model):
        ...     train_one_epoch(...)

    Raises on exit if any coefficient changed. This is the check that catches a
    frozen arm which was silently trained -- the failure mode that would
    invalidate a transfer result while still producing plausible numbers.
    """

    def __init__(self, module, atol=0.0):
        self.module = module
        self.atol = atol
        self.acts = find_activations(module)
        self.before = None

    def __enter__(self):
        self.before = [copy.deepcopy(a.coefficients()) for a in self.acts]
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        for a, before in zip(self.acts, self.before):
            after = a.coefficients()
            for key in ("a0", "a", "b", "w"):
                if key not in before:
                    continue
                v0, v1 = before[key], after[key]
                v0 = [v0] if not isinstance(v0, list) else v0
                v1 = [v1] if not isinstance(v1, list) else v1
                for x, y in zip(v0, v1):
                    if abs(x - y) > self.atol:
                        raise AssertionError(
                            f"frozen activation moved: {key} {x!r} -> {y!r}")
        return False
