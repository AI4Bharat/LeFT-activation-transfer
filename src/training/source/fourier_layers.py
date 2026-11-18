"""The learnable shared activation of the source network.

FourierActivation writes the pointwise nonlinearity as a truncated Fourier
series phi(t) = a0 + sum_k a_k cos(k t) + b_k sin(k t). The coefficients are
the LEARNABLE parameters and are INITIALISED FROM THE TRUE SERIES FIT OF GELU,
not from random values.
"""
import math
import re

import numpy as np
import torch
import torch.nn as nn

# np.trapz was removed in NumPy 2.0 in favor of np.trapezoid (same function,
# renamed); this keeps true_fourier_coeffs working on NumPy versions on
# either side of that split.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------- #
#  "fact_kK_global" variant-name parsing (K = any positive int, not just the
#  2 this study reports) -- shared by cv_vit.py (which builds the ONE shared
#  FourierActivation) and fourier_ffn.build_ffn (which dispatches every
#  layer's FFN to reuse it).
# --------------------------------------------------------------------------- #
GLOBAL_VARIANT_RE = re.compile(r"^fact_k(\d+)_global$")


def parse_global_k(kind):
    """K iff `kind` is a fact_kK_global variant, else None."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
#  True series coefficients of GELU on [-pi, pi]
# --------------------------------------------------------------------------- #
def gelu_np(t):
    # exact (erf) GELU
    from scipy.special import erf
    return 0.5 * t * (1.0 + erf(t / math.sqrt(2.0)))




def true_fourier_coeffs(K, L=math.pi, n_grid=4096):
    """Return (a0, a[K], b[K]): the true Fourier-series coefficients of GELU
    on [-L, L], computed by numerical (trapezoid)
    integration of the Euler formulas.

        a0  = 1/(2L) * int_{-L}^{L} f(t) dt
        a_k = 1/L    * int_{-L}^{L} f(t) cos(k*w*t) dt      (w = pi/L)
        b_k = 1/L    * int_{-L}^{L} f(t) sin(k*w*t) dt

    With L = pi the base frequency w = 1, so phi(t) uses cos(k t), sin(k t).
    """
    f = gelu_np
    t = np.linspace(-L, L, n_grid)
    ft = f(t)
    w = math.pi / L
    a0 = _trapz(ft, t) / (2 * L)
    a = np.zeros(K)
    b = np.zeros(K)
    for k in range(1, K + 1):
        a[k - 1] = _trapz(ft * np.cos(k * w * t), t) / L
        b[k - 1] = _trapz(ft * np.sin(k * w * t), t) / L
    return float(a0), a.astype(np.float32), b.astype(np.float32), float(w)


def fourier_series_eval(t, a0, a, b, w=1.0):
    """Evaluate the truncated series on a numpy array (used for plots/tests)."""
    out = np.full_like(t, a0, dtype=np.float64)
    for k in range(1, len(a) + 1):
        out += a[k - 1] * np.cos(k * w * t) + b[k - 1] * np.sin(k * w * t)
    return out


# --------------------------------------------------------------------------- #
#  Learnable Fourier-series activation (per neuron)
# --------------------------------------------------------------------------- #
class FourierActivation(nn.Module):
    """phi(t) = a0 + sum_{k=1..K} a_k cos(k w t) + b_k sin(k w t), per neuron.

    Parameters (learnable): a0 (M,), a (M,K), b (M,K).  They are initialised
    from the true Fourier coefficients of GELU, broadcast across all M
    neurons, so every neuron starts as a faithful copy of GELU and is then
    free to specialise during training.

    Set shared=True to tie ONE set of coefficients {a0, a, b} across all M
    neurons in the layer (param_dim=1 instead of M) -- every neuron applies
    the exact same learned Fourier-series activation, instead of each neuron
    specialising independently. This is what every "_global" variant uses.
    Relies on ordinary broadcasting in forward(): self.a/self.b of shape
    (1, K) broadcast against the (B, M, K) angle tensor exactly like (M, K)
    would, just tied across the M axis instead of varying over it, so
    forward() needs no shared-specific branch.
    """

    def __init__(self, num_features, K=8, w=1.0, shared=False):
        super().__init__()
        self.M, self.K, self.w, self.shared = num_features, K, w, shared
        param_dim = 1 if shared else num_features
        a0, a, b, w_true = true_fourier_coeffs(K)
        self.w = w_true
        self.a0 = nn.Parameter(torch.full((param_dim,), float(a0)))
        self._a = nn.Parameter(torch.from_numpy(a).repeat(param_dim, 1).clone())
        self._b = nn.Parameter(torch.from_numpy(b).repeat(param_dim, 1).clone())
        self.register_buffer("kvec", torch.arange(1, K + 1).float())
        # Unused, kept so the published checkpoints load into this model.
        self.register_buffer("progress", torch.tensor(0.0))

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def forward(self, t):
        # t: (B, M) ; build (B, M, K) angles = k * w * t
        ang = t.unsqueeze(-1) * (self.kvec * self.w)          # (B, M, K)
        cos, sin = torch.cos(ang), torch.sin(ang)
        # a, b: (1, K) (shared) or (M, K) broadcast over batch
        return (cos * self.a).sum(-1) + (sin * self.b).sum(-1) + self.a0
