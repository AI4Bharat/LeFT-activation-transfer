"""GELU-fit-initialised K=2 Fourier activation, coefficients FROZEN.

The paper's "Frozen GELU-init" control (Appendix D, the Swin depth sweep). The
same five-scalar shared Fourier curve as
fact_k2_global_trainable.FactK2GlobalTrainable,

    phi(t) = a0 + a1 cos(w t) + b1 sin(w t) + a2 cos(2w t) + b2 sin(2w t),

initialised at the true K=2 Fourier fit of GELU, but with a0/a/b pinned
(requires_grad=False) so they never leave that GELU-shaped initialisation.

This is deliberately NOT fact_fixed.FixedFAct. FixedFAct freezes the curve that
`fact_k2_global` *learned end-to-end on ImageNet-1K* and transplants it. This
module freezes the *GELU initialisation itself* -- the control that isolates
whether the Fourier reparameterisation starting from GELU, with the curve never
learned, differs from plain GELU. In the paper it separates that question from
the transfer question: across the Swin depth sweep the fixed Fourier
approximation of GELU gives no consistent advantage over GELU, while the
transferred curve beats it in all 24 dataset-depth cells.

Forward numerics are identical to FactK2GlobalTrainable (same eager sin/cos
path, same GELU-fit init), so a frozen result sits on precisely the same
footing as the learnable fact_k2_global arm -- the two differ only in whether
the five scalars receive gradients. The coefficients stay nn.Parameters (not
buffers) with requires_grad=False:
they are excluded from the trainable- and activation-parameter counts (0
activation parameters, equal capacity with the fixed zoo) yet still travel in
the checkpoint's state_dict. Because they carry no gradient, AdamW skips them
entirely, so neither the optimizer nor weight decay can drift them.
"""
from left.activations.fact_k2_global_trainable import FactK2GlobalTrainable


class FactK2GlobalFrozen(FactK2GlobalTrainable):
    K = 2

    def __init__(self, ref="gelu"):
        super().__init__(ref=ref)
        for p in (self.a0, self.a, self.b):
            p.requires_grad_(False)

    def extra_repr(self):
        return super().extra_repr().replace(
            "frozen=False (trainable)", "frozen=True (GELU-init, pinned)")
