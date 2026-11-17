"""The learnable shared activations of the source network.

FourierActivation writes the pointwise nonlinearity as a truncated Fourier
series phi(t) = a0 + sum_k a_k cos(k t) + b_k sin(k t); PAU writes it as a
rational function P(t)/Q(t). The coefficients are the LEARNABLE parameters and
are INITIALISED FROM THE TRUE SERIES FIT OF GELU, not from random values.
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


def parse_pau_global(kind):
    """True iff `kind` is "pau_global" -- the Pade Activation Unit (see PAU
    below) with ONE {a, b} coefficient pair shared across every neuron AND
    every block, mirroring fact_kK_global's sharing convention exactly so the
    two globally-shared activations are directly comparable. Plain equality
    check: PAU has no suffix family to parse."""
    return kind == "pau_global"


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


def true_pade_coeffs(m=5, n=4, L=math.pi, n_grid=4096):
    """Return (a[m+1], b[n]): numerator/denominator coefficients of a degree
    (m, n) Pade rational approximation to GELU on [-L, L], P(t) = sum_{j=0}^m a_j t^j, Q(t) = 1 + sum_{k=1}^n b_k t^k
    (PAU's forward pass takes |b_k| in Q -- see PAU below -- so this fit,
    which allows either sign, is only a starting point, not the guaranteed
    pole-free form).

    Solved as a LINEAR least-squares problem, the standard trick for fitting
    a rational function without iterating: the target relation
    f(t) = P(t)/Q(t) rearranges to P(t) - f(t)*Q(t) = 0, i.e.

        sum_j a_j t^j  -  sum_k b_k (f(t) t^k)  =  f(t)

    which is linear in the unknowns {a_j, b_k} even though the fitted
    function itself is not.
    """
    f = gelu_np
    t = np.linspace(-L, L, n_grid)
    ft = f(t)
    cols = [t ** j for j in range(m + 1)] + [-ft * (t ** k) for k in range(1, n + 1)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, ft, rcond=None)
    a = coef[:m + 1].astype(np.float32)
    b = coef[m + 1:].astype(np.float32)
    return a, b


def fourier_series_eval(t, a0, a, b, w=1.0):
    """Evaluate the truncated series on a numpy array (used for plots/tests)."""
    out = np.full_like(t, a0, dtype=np.float64)
    for k in range(1, len(a) + 1):
        out += a[k - 1] * np.cos(k * w * t) + b[k - 1] * np.sin(k * w * t)
    return out


# --------------------------------------------------------------------------- #
#  Pade Activation Unit baseline (per neuron, or one shared curve)
# --------------------------------------------------------------------------- #
class PAU(nn.Module):
    """Pade Activation Unit (Molina et al., "Pade Activation Units: End-to-end
    Learning of Flexible Activation Functions in Deep Networks", ICLR 2020):

        f(t) = P(t) / Q(t),   P(t) = sum_{j=0}^m a_j t^j,
                               Q(t) = 1 + sum_{k=1}^n |b_k| t^k

    Order (m, n) = (5, 4) is the paper's own default -- a degree-5 numerator
    over a degree-4 denominator, 10 learnable shape parameters per
    shared/per-neuron copy -- the numerator has m+1 = 6 coefficients, not m
    (vs FourierActivation K=2's 5).

    NOT THE "SAFE PAU". The absolute value here is taken per coefficient,
    1 + sum_k |b_k| t^k, not around the whole sum, 1 + |sum_k b_k t^k|. Only
    the latter is >= 1 for every real t and therefore provably pole-free;
    this one is not, because the ODD powers are negative for t < 0:

      * Q dips below 1 for t < 0. On the ImageNet-1K pau_global seed-1 curve
        the minimum is Q = 0.855 at t = -0.427.
      * Q can reach 0. Starting from the GELU fit below and moving b_3 alone
        to 0.1 puts a real root at t = -3.19; at b_3 = 0.2 it is at t = -2.09,
        where |f| exceeds 8e4 and changes sign.

    The runs in this repository stayed clear of a pole -- the fit leaves
    b_1 = b_3 = 0, so Q starts even in t, and the trained curve keeps
    min Q = 0.855 -- but nothing in the parameterisation enforces that, and
    the b_k are ordinary nn.Parameters. Switching to 1 + |sum_k b_k t^k|
    would restore the guarantee at the cost of changing the baseline, so it
    is left alone and documented rather than silently altered; cuda_pau/
    implements the same 1 + sum_k |b_k| t^k, so the two paths agree.

    The coefficients start from the least-squares Pade fit to GELU on
    [-L, L] (see true_pade_coeffs), the same "start as a faithful copy of
    GELU" convention as FourierActivation. Note the fit itself does not
    enforce Q(t) > 0 (it solves for signed b_k, see true_pade_coeffs'
    docstring); forward() takes |b_k| regardless, so the realised initial
    curve is the fit's P(t) over that rectified Q, a close but not exact
    reproduction of the unconstrained fit.

    shared=True ties ONE {a, b} pair across all `width` neurons (param shape
    (1, m+1)/(1, n) instead of (width, m+1)/(width, n)) -- the same
    "_global" sharing convention as fact_kK_global, so the two "one global
    learnable shape" activations are directly comparable.
    """

    def __init__(self, width, m=5, n=4, shared=False, L=math.pi):
        super().__init__()
        self.width, self.m, self.n, self.shared = width, m, n, shared
        param_dim = 1 if shared else width
        a_true, b_true = true_pade_coeffs(m, n, L=L)
        self.a = nn.Parameter(torch.from_numpy(a_true).repeat(param_dim, 1).clone())
        self.b = nn.Parameter(torch.from_numpy(b_true).repeat(param_dim, 1).clone())
        self.register_buffer("jvec", torch.arange(0, m + 1).float())
        self.register_buffer("kvec", torch.arange(1, n + 1).float())

    def forward(self, t):
        # t: (B, width) -> powers (B, width, m+1) / (B, width, n), broadcast
        # against self.a/self.b of shape (1 or width, m+1)/(1 or width, n) --
        # exactly the same broadcasting convention as FourierActivation's
        # (cos, sin) angle tensors against its (1 or M, K) coefficients.
        tp = t.unsqueeze(-1) ** self.jvec
        tq = t.unsqueeze(-1) ** self.kvec
        P = (tp * self.a).sum(-1)
        Q = 1.0 + (tq * self.b.abs()).sum(-1)
        return P / Q

    def extra_repr(self):
        return f"width={self.width}, m={self.m}, n={self.n}, shared={self.shared}"


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
