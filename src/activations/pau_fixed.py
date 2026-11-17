"""The learned PAU (Pade Activation Unit) curve, frozen into a FIXED activation.

The PAU counterpart of fact_fixed.FixedFAct, and built to the same contract, so
that "frozen FAct" and "frozen PAU" differ only in which learned curve they
carry:

  * the imagenet1k_100ep `pau_global` run trained ONE rational activation

        f(t) = P(t) / Q(t),   P(t) = sum_{j=0..5} a_j t^j,
                              Q(t) = 1 + sum_{k=1..4} |b_k| t^k

    end-to-end alongside a depth-6 / embed-192 ViT, shared by reference across
    every block and every neuron, starting from the least-squares Pade fit to
    GELU on [-pi, pi]. It converged to a specific curve (ImageNet-1k top-1
    0.6449). THAT CURVE -- not the mechanism that learned it -- is what this
    module provides, as a drop-in nonlinearity with NO learnable parameters;
  * so the coefficients live in `register_buffer`, not `nn.Parameter`: the arm
    contributes exactly zero activation parameters, which is what keeps it
    inside this study's equal-capacity guarantee (train_target.py asserts it),
    keeps weight decay from drifting a frozen curve, and still lets a
    checkpoint describe its own activation.

Coefficients come from code/learned_pau_coeffs.json (see
extract_learned_pau_coeffs.py, which also records the curve diagnostics below).

TWO properties of this particular curve are worth knowing before reading any
result, because neither holds for frozen FAct:

  1. NOT pole-free by construction, only pole-free in fact. PAU is published as
     "safe" on the grounds that Q(t) = 1 + sum |b_k| t^k >= 1, but that is
     false for odd k, where t^k is signed (see
     FAct/Theory/Lipschitz-of-activations). This curve dips to Q = 0.8547 near
     t = -0.427. It never reaches zero, so the function is finite everywhere --
     but the margin is a measured fact about these coefficients, not a
     guarantee, so __init__ re-verifies it on every construction instead of
     trusting the JSON.
  2. It has a NEGATIVE tail slope, a5/|b4| = -0.5225: f(t) ~ -0.52 t as
     |t| -> inf. f(-10) = +5.01 and f(+10) = -5.63. The learned curve is
     eventually decreasing in both directions -- closer to a sign-flip than to
     any of the ten monotone-tailed fixed baselines. Frozen FAct is periodic
     and so has no tail slope to compare against.

Evaluation path: pure PyTorch, deliberately, unlike FixedFAct's forked CUDA
kernel.
  * It is the arithmetic of fourier_layers.PAU.forward term for term, i.e. the
    reference implementation of the function that was actually trained -- and
    with the coefficients in buffers, autograd already computes grad wrt the
    input ONLY, which is the whole benefit FixedFAct needed a kernel fork
    (`backward_input`) to obtain.
  * FAct/cuda_pau exists and is ~21x faster on the ImageNet shape, but it
    JIT-compiles at import time against three different nvcc/torch pairings
    across this study's hosts, and a killed JIT load leaves a lock that hangs
    every later import. At this experiment's shape (d_ff = 256, ~50 tokens,
    batch 256) the (B, N, 6) power tensor is a few tens of MB and the
    activation is not the bottleneck, so the kernel buys nothing worth that
    risk.
  * CONSEQUENCE, stated because it would otherwise be a silent unfairness:
    wall-clock is NOT comparable between this arm and the CUDA-kernel
    fact_fixed arm. Test accuracy at a fixed 100-epoch budget -- the metric
    this experiment reports -- is unaffected.
"""
import json
import os

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS_DIR))
#: The transferred PAU curve, as shipped in this repository. Override with
#: $LEFT_PAU_COEFFS to evaluate a different one.
DEFAULT_COEFF_JSON = os.path.join(_REPO, "artifacts", "pau",
                                  "imagenet1k_vitd6_pau_global.json")


def coeff_json():
    return os.environ.get("LEFT_PAU_COEFFS", DEFAULT_COEFF_JSON)


COEFF_JSON = DEFAULT_COEFF_JSON


def load_coeffs(path=None):
    path = coeff_json() if path is None else path
    with open(path) as f:
        d = json.load(f)
    assert (d["m"], d["n"]) == (5, 4), (
        f"this module is order (5,4) only, got ({d['m']}, {d['n']})")
    assert len(d["a"]) == d["m"] + 1 and len(d["b"]) == d["n"]
    return d


class FixedPAU(nn.Module):
    """f(t) = P(t)/Q(t) with the coefficients frozen at the learned
    ImageNet-1k values. Zero learnable parameters."""

    M, N = 5, 4

    def __init__(self, coeff_path=None, use_cuda_kernel=True):
        super().__init__()
        d = load_coeffs(coeff_path)
        # Accepted and ignored: every activation in this zoo takes the same
        # build_act(kind, use_cuda_kernel) signature so callers never branch on
        # the variant name. There is no kernel path here (see module docstring),
        # and recording the flag keeps `extra_repr` honest about that.
        self.requested_cuda_kernel = bool(use_cuda_kernel)
        self.coeff_source = d.get("source_checkpoint", coeff_path or coeff_json())
        # (1, m+1) / (1, n): the "shared" layout -- one coefficient set
        # broadcast over every feature, matching how pau_global trained it
        # (tied across neurons AND across depth).
        self.register_buffer("a", torch.tensor([d["a"]], dtype=torch.float32))
        self.register_buffer("b", torch.tensor([d["b"]], dtype=torch.float32))
        self.register_buffer("jvec", torch.arange(0, self.M + 1, dtype=torch.float32))
        self.register_buffer("kvec", torch.arange(1, self.N + 1, dtype=torch.float32))
        # A frozen rational function is only an activation if its denominator
        # never vanishes, and a pole would show up as a silent inf/NaN mid-run
        # rather than as an error. Cheap enough to re-derive here on every
        # construction rather than trust what the JSON says.
        self._assert_pole_free()

    def _assert_pole_free(self):
        t = torch.linspace(-50.0, 50.0, 200_001, dtype=torch.float64)
        b = self.b.double().flatten().abs()
        Q = 1.0 + sum(b[k - 1] * t ** k for k in range(1, self.N + 1))
        qmin = float(Q.min())
        assert qmin > 1e-3, (
            f"frozen PAU denominator reaches Q = {qmin:.6g} on [-50, 50]: this "
            "curve has a pole (or comes within 1e-3 of one) and must not be "
            "used as an activation")
        self.min_Q = qmin

    def forward(self, t):
        # fourier_layers.PAU.forward, term for term. Promote half inputs to
        # fp32 first: t**5 in fp16 overflows at |t| ~ 25 (65504 ** (1/5)), and
        # this experiment's trainer runs fp32 anyway -- an AMP caller must not
        # silently get a different function.
        if t.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type=t.device.type, enabled=False):
                return self._eval(t.float()).to(t.dtype)
        return self._eval(t)

    def _eval(self, t):
        tp = t.unsqueeze(-1) ** self.jvec
        tq = t.unsqueeze(-1) ** self.kvec
        P = (tp * self.a).sum(-1)
        Q = 1.0 + (tq * self.b.abs()).sum(-1)
        return P / Q

    def extra_repr(self):
        return (f"m={self.M}, n={self.N}, frozen=True, params=0, "
                f"cuda_kernel=False (requested={self.requested_cuda_kernel}), "
                f"min_Q={self.min_Q:.6f}, "
                f"a={[round(v, 6) for v in self.a.flatten().tolist()]}, "
                f"b={[round(v, 6) for v in self.b.flatten().tolist()]}")
