"""PAU -- a globally shared activation parameterised as a Pade rational
function. The second activation implemented inside the LeFT framework.

    phi(t) = P(t) / Q(t),   P(t) = sum_{j=0..m} a_j t^j,
                            Q(t) = 1 + sum_{k=1..n} |b_k| t^k

The absolute value in Q is PAU's own device for keeping the denominator from
crossing zero (Molina et al., 2020, "Pade Activation Units"). At the paper's
(m, n) = (5, 4) that is ten scalars for the whole network, against FAct's five
at K = 2.

PAU exists in this repository for one reason: it answers "is the transfer
result about Fourier series, or about globally shared learned nonlinearities?"
It is the same learn-freeze-transfer protocol with a different, non-periodic,
*local*-basis parameterisation. In the paper's appendix, a shared PAU learned in
the depth-6 ImageNet-1K source network and frozen into the depth-2 ViT target
beats GELU on all four datasets -- and beats frozen FAct there by 0.15 to 0.87
percentage points. The transfer effect is therefore not specific to the Fourier
representation. That comparison covers one target architecture only.

Implementation note: unlike FAct, a Pade basis is local -- a coefficient's
influence concentrates where the corresponding power dominates -- so a
transferred PAU curve is in principle more exposed to a shift in the target
network's pre-activation distribution. That contrast is the point.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn

from left.activations.base import SharedActivation
from left.activations.fourier_fit import pade_coeffs


class PAU(SharedActivation):
    """Globally shared Pade activation of degree (m, n).

    Args:
        m, n: numerator / denominator degrees. (5, 4) is PAU's own default and
            the paper's setting.
        ref: "gelu" to initialise at the Pade fit of GELU (the paper's
            setting), or None for zeros -- what `from_artifact` uses, since it
            overwrites the coefficients immediately afterwards.
        trainable: source arms use True; transferred target arms use False.
    """

    family = "pau"

    def __init__(self, m=5, n=4, ref="gelu", trainable=True):
        super().__init__()
        if ref not in ("gelu", None):
            raise ValueError(f"ref must be 'gelu' or None, got {ref!r}")
        self.m, self.n, self.ref = int(m), int(n), ref
        if ref is None:
            a = np.zeros(m + 1, np.float32)
            b = np.zeros(n, np.float32)
        else:
            a, b = pade_coeffs(m, n)
        self.a = nn.Parameter(torch.from_numpy(np.asarray(a)).reshape(1, m + 1).clone())
        self.b = nn.Parameter(torch.from_numpy(np.asarray(b)).reshape(1, n).clone())
        self.register_buffer("jvec", torch.arange(0, m + 1, dtype=torch.float32))
        self.register_buffer("kvec", torch.arange(1, n + 1, dtype=torch.float32))
        if not trainable:
            self.freeze()

    def forward(self, t):
        tp = t.unsqueeze(-1) ** self.jvec
        tq = t.unsqueeze(-1) ** self.kvec
        P = (tp * self.a).sum(-1)
        Q = 1.0 + (tq * self.b.abs()).sum(-1)
        return P / Q

    # -- the LeFT transfer interface ---------------------------------------- #
    def coefficients(self):
        return {
            "family": self.family,
            "m": self.m,
            "n": self.n,
            "a": [float(v) for v in self.a.detach().flatten().tolist()],
            "b": [float(v) for v in self.b.detach().flatten().tolist()],
        }

    def load_coefficients(self, state, strict=True):
        if strict:
            fam = state.get("family", "pau")
            if fam != self.family:
                raise ValueError(f"coefficient artifact is for {fam!r}, "
                                 f"this module is {self.family!r}")
            if (int(state["m"]), int(state["n"])) != (self.m, self.n):
                raise ValueError(f"artifact is degree ({state['m']}, {state['n']}), "
                                 f"this module is ({self.m}, {self.n})")
        with torch.no_grad():
            self.a.data = torch.tensor([list(map(float, state["a"]))],
                                       dtype=torch.float32, device=self.a.device)
            self.b.data = torch.tensor([list(map(float, state["b"]))],
                                       dtype=torch.float32, device=self.b.device)
        return self

    def freeze(self):
        for p in (self.a, self.b):
            p.requires_grad_(False)
        return self

    def unfreeze(self):
        for p in (self.a, self.b):
            p.requires_grad_(True)
        return self

    @classmethod
    def from_artifact(cls, path, trainable=False, **kw):
        with open(path) as f:
            state = json.load(f)
        self = cls(m=int(state["m"]), n=int(state["n"]), ref=None,
                   trainable=trainable, **kw)
        self.load_coefficients(state, strict=False)
        if not trainable:
            self.freeze()
        return self

    def extra_repr(self):
        return (f"m={self.m}, n={self.n}, frozen={self.frozen}, ref={self.ref!r}")
