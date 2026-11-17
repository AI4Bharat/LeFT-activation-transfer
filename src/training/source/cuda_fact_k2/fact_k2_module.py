"""Drop-in CUDA-accelerated replacement for fourier_layers.FourierActivation
at K=2 -- the truncation the paper's source-network runs use, and the one the
transferred coefficients in `fact/coefficients/` were learned at.

phi(t) = a0 + a1*cos(w*t) + b1*sin(w*t) + a2*cos(2*w*t) + b2*sin(2*w*t)

The extension is JIT-compiled on first import via torch.utils.cpp_extension.load
(NOT setup.py's build_ext -- nvcc and the torch build commonly disagree on CUDA
version, and build_ext's _check_cuda_version hard-fails on that mismatch even
when the toolchain itself compiles and runs fine; load() skips that check). Compiled output is cached under torch's extensions dir, so
only the first import per machine pays the compile cost.

Usage (mirrors FourierActivation's constructor):

    from fact_k2_module import FourierActivationK2CUDA
    act = FourierActivationK2CUDA(d_ff, shared=True).cuda()
    y = act(x)   # x: (..., d_ff) CUDA float32/float64 tensor

`shared=True` matches every "_global" variant (one set of 5 coefficients for
the whole network); `shared=False` matches per-neuron coefficients (P=M).
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_layers import true_fourier_coeffs  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
fact_k2_cuda = load(
    name="fact_k2_cuda",
    sources=[os.path.join(_THIS_DIR, "fact_k2_cuda_kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class _FactK2Fn(torch.autograd.Function):
    """Autograd hookup for the K=2 kernel.

    `w` may be a plain Python float (every fixed-w variant) OR a 0-dim
    Parameter (the `_lw` learnable-frequency variants). Only the tensor case
    asks the kernel for a d(phi)/dw reduction, and the kernel templates that
    branch away at compile time, so a fixed-w call costs exactly what it did
    before this was added.

    Nothing but `t` and the five coefficients is stashed for backward: the
    kernel recomputes sin/cos from `t` inside the backward pass rather than
    keeping the (B, M, K) cos/sin intermediates autograd would otherwise save.
    That recompute is the whole point of using the kernel here -- at WMT14
    shapes those intermediates are what makes the pure-PyTorch path cost
    ~5 GiB more than GELU.
    """

    @staticmethod
    def forward(ctx, t, a0, a, b, w):
        t_c = t.contiguous()
        a0_c, a_c, b_c = a0.contiguous(), a.contiguous(), b.contiguous()
        w_val = float(w)
        out = fact_k2_cuda.forward(t_c, a0_c, a_c, b_c, w_val)
        ctx.save_for_backward(t_c, a0_c, a_c, b_c)
        ctx.w = w_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t, a0, a, b = ctx.saved_tensors
        need_grad_w = ctx.needs_input_grad[4]
        grad_input, grad_a0, grad_a, grad_b, grad_w = fact_k2_cuda.backward(
            grad_output.contiguous(), t, a0, a, b, ctx.w, need_grad_w)
        return (grad_input, grad_a0, grad_a, grad_b,
                grad_w if need_grad_w else None)


class FourierActivationK2CUDA(nn.Module):
    """CUDA-kernel FourierActivation, K fixed at 2. See module docstring."""

    K = 2

    def __init__(self, num_features, w=None, shared=False):
        super().__init__()
        self.M = num_features
        self.shared = shared
        param_dim = 1 if shared else num_features
        a0, a, b, w_true = true_fourier_coeffs(self.K)
        self.w = float(w_true if w is None else w)
        self.a0 = nn.Parameter(torch.full((param_dim,), float(a0)))
        self._a = nn.Parameter(torch.from_numpy(a).repeat(param_dim, 1).clone())
        self._b = nn.Parameter(torch.from_numpy(b).repeat(param_dim, 1).clone())

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def forward(self, t):
        assert t.is_cuda, "FourierActivationK2CUDA requires a CUDA input tensor"
        assert t.size(-1) == self.M or self.shared, (
            f"input last dim ({t.size(-1)}) must equal num_features ({self.M})")
        if t.dtype in (torch.float16, torch.bfloat16):
            # The kernel has no half/bfloat16 forward (fact_k2_cuda_kernel.cu
            # only dispatches scalar_t in {half, bfloat16, float, double} for
            # the OUTPUT copy, but accumulates/launches trig math in acc_t --
            # actually it's the CUDA op itself that has no registered half
            # kernel here) and this autograd.Function carries no autocast
            # policy, so a caller training under AMP (e.g. cv_train.py) would
            # otherwise hand it a fp16 `t` and hard-crash. Upcast for the
            # call and cast back, matching what the pure-PyTorch
            # FourierActivation already does under autocast (its `mul`
            # against the fp32 kvec buffer promotes a half input to fp32
            # before any trig runs).
            with torch.autocast(device_type=t.device.type, enabled=False):
                out = _FactK2Fn.apply(t.float(), self.a0, self._a, self._b, self.w)
            return out.to(t.dtype)
        return _FactK2Fn.apply(t, self.a0, self._a, self._b, self.w)

    def extra_repr(self):
        return (f"num_features={self.M}, K={self.K}, w={float(self.w):.6f}, "
                f"shared={self.shared}")
