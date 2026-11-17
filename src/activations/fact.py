"""FAct -- a globally shared activation parameterised as a truncated Fourier
series. The first activation implemented inside the LeFT framework.

    phi(t) = a0 + sum_{k=1..K} [ a_k cos(k w t) + b_k sin(k w t) ]

At K = 2 that is five scalars -- a0, a1, a2, b1, b2 -- for an entire network:
one curve shared by every activation site at every depth. The curve is
initialised at the Fourier fit of a conventional activation (GELU by default,
see `left.activations.fourier_fit`), learned jointly with the weights in the
source network, then frozen and installed in independently initialised target
networks.

This module is the *framework-facing* implementation: K is configurable, the
curve can be trainable or frozen, and `coefficients()` / `load_coefficients()`
are the transfer interface. The paper's exact published runs additionally ship
as byte-faithful modules alongside it --
`fact_k2_global_trainable.FactK2GlobalTrainable` (the learnable source arm),
`fact_k2_global_frozen.FactK2GlobalFrozen` (the GELU-init-frozen null control)
and `fact_fixed.FixedFAct` (the transferred curve, with a fused CUDA kernel).
`tests/test_fact.py` asserts this module agrees with them numerically, so new
work can use `FAct` and still sit on the published numbers.

Why a Fourier series at all: it is a *global* basis. Every coefficient affects
the whole real line, so a curve learned in one network cannot encode a
pre-activation distribution local to that network's layers -- which is the
property that makes the learned shape worth transferring. See
docs/methodology.md.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn

from left.activations.base import SharedActivation
from left.activations.fourier_fit import fourier_coeffs


class FAct(SharedActivation):
    """Truncated-Fourier shared activation.

    Args:
        K: number of harmonics. K=2 (five parameters) is the paper's setting;
            other values are supported but were not evaluated in the paper.
        ref: "gelu" to initialise at the GELU fit (the paper's setting), or
            None for zeros -- what `from_artifact` uses, since it overwrites
            the coefficients immediately afterwards.
        w: base frequency. Defaults to the fit's own w (1.0 for L = pi).
        trainable: whether the coefficients start learnable. The source arm
            uses True; a transferred target arm uses False.
        quadrature: which GELU-fit quadrature to initialise from -- see
            `left.activations.fourier_fit`. "trapezoid" is the exact one and
            matches the source-scale runs; "riemann" reproduces the
            target-scale published initialisation.

    One instance, installed everywhere:

        >>> act = FAct(K=2)                      # five parameters, total
        >>> model = build_vit("cifar10", act)
        >>> act.num_activation_parameters()
        5
    """

    family = "fact"

    def __init__(self, K=2, ref="gelu", w=None, trainable=True,
                 quadrature="trapezoid"):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if ref not in ("gelu", None):
            raise ValueError(f"ref must be 'gelu' or None, got {ref!r}")
        self.K = int(K)
        self.ref = ref
        self.quadrature = quadrature
        if ref is None:
            a0, a, b, w_fit = 0.0, np.zeros(K, np.float32), np.zeros(K, np.float32), 1.0
        else:
            a0, a, b, w_fit = fourier_coeffs(self.K, quadrature=quadrature)
        w = float(w_fit if w is None else w)

        self.a0 = nn.Parameter(torch.tensor([a0], dtype=torch.float32))
        self.a = nn.Parameter(torch.from_numpy(np.asarray(a)).reshape(1, self.K).clone())
        self.b = nn.Parameter(torch.from_numpy(np.asarray(b)).reshape(1, self.K).clone())
        # A buffer, never a parameter: the paper fixes w = 1. w is a gauge
        # direction -- weight decay slides it without changing the function
        # class -- so there is no learnable-frequency arm to support here.
        self.register_buffer("omega", torch.tensor([w], dtype=torch.float32))
        self.register_buffer("kvec", torch.arange(1, self.K + 1, dtype=torch.float32))
        if not trainable:
            self.freeze()

    # -- the curve ---------------------------------------------------------- #
    @property
    def w(self):
        return float(self.omega.item())

    def forward(self, t):
        # (1, K) coefficients broadcast against (..., M, K) angles: tied across
        # every feature and every call site.
        ang = t.unsqueeze(-1) * (self.kvec * self.omega)
        return (self.a0
                + (ang.cos() * self.a).sum(-1)
                + (ang.sin() * self.b).sum(-1))

    # -- the LeFT transfer interface ---------------------------------------- #
    def coefficients(self):
        return {
            "family": self.family,
            "K": self.K,
            "w": self.w,
            "a0": float(self.a0.detach().reshape(()).item()),
            "a": [float(v) for v in self.a.detach().flatten().tolist()],
            "b": [float(v) for v in self.b.detach().flatten().tolist()],
        }

    def load_coefficients(self, state, strict=True):
        if strict:
            fam = state.get("family", "fact")
            if fam != self.family:
                raise ValueError(f"coefficient artifact is for {fam!r}, "
                                 f"this module is {self.family!r}")
            if int(state["K"]) != self.K:
                raise ValueError(f"artifact has K={state['K']}, this module has "
                                 f"K={self.K}; build FAct(K={state['K']}) instead")
        K = int(state["K"])
        if K != self.K:  # non-strict: adopt the artifact's K
            self.K = K
            self.register_buffer("kvec", torch.arange(1, K + 1, dtype=torch.float32,
                                                      device=self.kvec.device))
        with torch.no_grad():
            self.a0.copy_(torch.tensor([float(state["a0"])]))
            self.a.data = torch.tensor([list(map(float, state["a"]))],
                                       dtype=torch.float32, device=self.a.device)
            self.b.data = torch.tensor([list(map(float, state["b"]))],
                                       dtype=torch.float32, device=self.b.device)
            self.omega.copy_(torch.tensor([float(state.get("w", self.w))],
                                          device=self.omega.device))
        return self

    def freeze(self):
        for p in (self.a0, self.a, self.b):
            p.requires_grad_(False)
        return self

    def unfreeze(self):
        for p in (self.a0, self.a, self.b):
            p.requires_grad_(True)
        return self

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_artifact(cls, path, trainable=False, **kw):
        """Build the activation described by a coefficient artifact JSON (see
        `artifacts/fact/`). This is the ordinary way to install a transferred
        curve: no source checkpoint and no source weights are needed."""
        with open(path) as f:
            state = json.load(f)
        self = cls(K=int(state["K"]), ref=None, w=float(state.get("w", 1.0)),
                   trainable=trainable, **kw)
        self.load_coefficients(state, strict=False)
        if not trainable:
            self.freeze()
        return self

    def extra_repr(self):
        return (f"K={self.K}, w={self.w:.6f}, frozen={self.frozen}, "
                f"ref={self.ref!r}, a0={self.a0.item():.6f}, "
                f"a={[round(v, 6) for v in self.a.flatten().tolist()]}, "
                f"b={[round(v, 6) for v in self.b.flatten().tolist()]}")


def gelu_init_coefficients(K=2, quadrature="trapezoid"):
    """The GELU-fit initialisation as a coefficient artifact dict -- the
    starting point of every LeFT source run, and the null control a transferred
    curve is compared against."""
    a0, a, b, w = fourier_coeffs(K, quadrature=quadrature)
    return {"family": "fact", "K": K, "w": w, "a0": a0,
            "a": [float(v) for v in a], "b": [float(v) for v in b],
            "note": f"GELU Fourier fit, K={K}, {quadrature} quadrature"}
