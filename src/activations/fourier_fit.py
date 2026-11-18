"""The GELU series fits -- how a LeFT activation is initialised.

A LeFT run does not start its shared nonlinearity from noise. It starts it at a
truncated series approximation of GELU, so that at step 0 the network computes very nearly the conventional baseline and
every subsequent difference is attributable to the curve having been *learned*.

Two quadratures are provided, and the distinction matters for exact
reproduction:

``trapezoid`` (default)
    Trapezoid rule on an endpoint-inclusive grid -- the more accurate of the
    two, and what the **source-scale** ImageNet-1K runs used
    (``src/training/source/fourier_layers.py:true_fourier_coeffs``). The
    manuscript's reported GELU-fit initialisation is this one.

``riemann``
    Left Riemann sum on an endpoint-excluded grid -- what the **target-scale**
    (~100K-parameter) published runs used
    (``fact_k2_global_trainable.true_fourier_coeffs``).

At K = 2 the two disagree by about 8e-4 on a1/a2 (riemann gives a2 = 0.0253,
trapezoid 0.0261, the exact value). Neither is wrong; each reproduces the
initialisation of its own scale's published runs, and the coefficients are
trainable from step 1 in either case. Nothing in the *transfer* result depends
on the choice, because the transferred curve is the learned one.
"""
import math

import numpy as np


def gelu_np(t):
    """Exact (erf) GELU -- the only reference activation the paper fits."""
    from scipy.special import erf
    return 0.5 * t * (1.0 + erf(t / math.sqrt(2.0)))


QUADRATURES = ("trapezoid", "riemann")


def _trapz(y, x):
    return float(np.trapz(y, x))


def fourier_coeffs(K, L=math.pi, n_grid=4096, quadrature="trapezoid"):
    """Return ``(a0, a[K], b[K], w)``: the Fourier coefficients of GELU
    truncated to K harmonics on [-L, L], by numerical integration of the Euler
    formulas

        a0  = 1/(2L) int_{-L}^{L} f(t) dt
        a_k = 1/L    int_{-L}^{L} f(t) cos(k w t) dt      (w = pi / L)
        b_k = 1/L    int_{-L}^{L} f(t) sin(k w t) dt

    With L = pi the base frequency is w = 1, so phi uses cos(kt), sin(kt).
    See the module docstring on `quadrature`.
    """
    if quadrature not in QUADRATURES:
        raise ValueError(f"quadrature must be one of {QUADRATURES}, got {quadrature!r}")
    f = gelu_np
    w = math.pi / L
    if quadrature == "trapezoid":
        t = np.linspace(-L, L, n_grid)
        ft = f(t)
        a0 = _trapz(ft, t) / (2 * L)
        a = np.array([_trapz(ft * np.cos(k * w * t), t) / L for k in range(1, K + 1)])
        b = np.array([_trapz(ft * np.sin(k * w * t), t) / L for k in range(1, K + 1)])
    else:  # riemann
        t = np.linspace(-L, L, n_grid, endpoint=False)
        dt = t[1] - t[0]
        ft = f(t)
        a0 = (ft.sum() * dt) / (2.0 * L)
        a = np.array([(ft * np.cos(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
        b = np.array([(ft * np.sin(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
    return float(a0), a.astype(np.float32), b.astype(np.float32), float(w)


def fourier_series_eval(t, a0, a, b, w=1.0):
    """Evaluate the truncated series on a numpy array (plots and tests)."""
    out = np.full_like(np.asarray(t, dtype=np.float64), float(a0))
    for k in range(1, len(a) + 1):
        out += a[k - 1] * np.cos(k * w * t) + b[k - 1] * np.sin(k * w * t)
    return out
