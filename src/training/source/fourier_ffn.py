"""The position-wise FFN (the per-token MLP) of a Transformer block, in the
two forms this study compares:

    standard FFN : y = W2 @ GELU(W1 @ x + b1) + b2      (fixed activation)
    FAct FFN     : y = W2 @ phi(W1 @ x + b1) + b2       (phi learned, shared)

Only the activation changes; W1, W2, attention, embeddings, LayerNorm and
residuals are identical in both.
"""
import torch.nn as nn

from fourier_layers import parse_global_k, parse_pau_global


class StandardFFN(nn.Module):
    """The ordinary Transformer FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model, d_ff, dropout=0.1, act="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class FactFFN(nn.Module):
    """Standard FFN (fully learnable W1, W2, d_ff = 4*d_model) with GELU
    replaced by the shared learnable activation `act` -- only the pointwise
    nonlinearity changes, so the comparison isolates the activation alone.

        FAct FFN : y = W2 @ phi( W1 @ x + b1 ) + b2     (W1, W2 learned)

    `act` is the ONE activation module the caller (cv_vit.VisionTransformer)
    built once and reuses for every layer: passing the same nn.Module (same
    parameter tensors) to every FactFFN is what makes `fact_k2_global` /
    `pau_global` one nonlinearity for the whole network rather than one per
    layer.
    """

    def __init__(self, d_model, d_ff, act, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = act
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


# --------------------------------------------------------------------------- #
FFN_VARIANTS = [
    "standard",             # d_ff = 4*d_model, learned FFN, fixed GELU (reference)
    "fact_k2_global",       # d_ff = 4*d_model; ONE FourierActivation K=2 (tied
                            # across neurons too) is instantiated ONCE and
                            # shared BY REFERENCE across every block's FFN --
                            # the whole network applies the exact same
                            # nonlinearity everywhere. W1/W2 stay per-layer.
                            # Any "fact_kK_global" name works the same way.
    "pau_global",           # fact_k2_global's sharing convention with a Pade
                            # Activation Unit (order (5,4)) in place of the
                            # Fourier series.
]


def build_ffn(kind, d_model, d_ff_standard, dropout, act=None):
    if kind == "standard":
        return StandardFFN(d_model, d_ff_standard, dropout)
    if parse_global_k(kind) is not None or parse_pau_global(kind):
        # `act` must be the ONE activation module the caller
        # (cv_vit.VisionTransformer) built once and is reusing across every
        # layer -- these variants have no meaning per-layer in isolation.
        assert act is not None, (
            f"{kind} needs a shared `act` module built once by the caller "
            "(see cv_vit.VisionTransformer.__init__) and passed to every layer")
        return FactFFN(d_model, d_ff_standard, act, dropout=dropout)
    raise ValueError(f"unknown ffn kind: {kind}")
