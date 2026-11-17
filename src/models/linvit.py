"""Linear-attention counterpart of left.models.vit's ~100K ViT -- identical for
every activation, with the attention *operator* as the single injected
difference.

WHY: the manuscript tests whether a nonlinearity learned-then-frozen in a ViT
transfers to an independently initialized target. The ViT arm (left.models.vit) and
the windowed-attention Swin arm (left.models.swin) both mix tokens with *softmax
dot-product* attention; they differ only in attention *locality* (global vs
shifted-window) and in hierarchy. Neither varies the attention operator itself.
This file supplies that missing axis: the same pre-LN ViT, the same FFN
sandwich, the same single shared activation, but with softmax self-attention
replaced by the kernelized *linear* attention of Katharopoulos et al. (2020),
"Transformers are RNNs". It re-measures the frozen-transfer result under a
genuinely different attention mechanism, since the curve was learned under
softmax attention.

WHAT CHANGES, AND ONLY THIS:
  MHSA (left.models.vit) computes softmax(QK^T / sqrt(d)) V -- an explicit N x N
  attention matrix. LinearMHSA below replaces the softmax kernel with a feature
  map phi(x) = elu(x) + 1 (elementwise, non-negative) and the identity

      attn(Q,K,V)_i = phi(Q_i) . (sum_j phi(K_j) (x) V_j) / (phi(Q_i) . sum_j phi(K_j))

  i.e. the keys and values are summed into a (d x d) context once and every
  query reads from that context. No N x N matrix is ever formed. The result is
  a global, content-based token mixer like softmax attention, but with a
  different (non-softmax, non-normalised-exponential) similarity kernel.

WHAT IS HELD IDENTICAL to left.models.vit.ActViT, so a GELU-vs-FAct comparison here is
an activation-only comparison exactly as it is for the ViT and Swin arms:
  * PatchEmbed and MLP are IMPORTED from left.models.vit, not re-implemented -- the
    shared activation sits in exactly the same Linear -> act -> dropout ->
    Linear site.
  * qkv is Linear(dim, 3*dim, bias=True) and proj is Linear(dim, dim), the
    SAME shapes MHSA uses, so the linear-attention block has the identical
    parameter count to the softmax block -- the attention swap costs zero
    parameters, and the equal-capacity assertion in train_target.py still holds.
  * [CLS] token, learnable absolute pos-embed, pre-LN block order, trunc_normal
    init (std=0.02, zero biases, LN at 1/0), depth-2 / embed-64 / heads-4 /
    mlp_ratio-4 budget, dropout 0.1 -- all verbatim from left.models.vit.

HONEST NOTE ON SCALE: at this study's sequence length (49-64 patches + 1 CLS)
linear attention has no speed advantage over softmax -- an N x N matrix at
N~65 is trivial. Speed is not the point; the point is that the *operator* is
different, so a surviving transfer margin cannot be an artifact of the softmax
kernel specifically. Linear attention is also run in fp32 like everything else
(train_target.py), and phi=elu+1 keeps the denominator strictly positive, so no
softmax-style numerical stabilisation is needed; a small eps guards it anyway.

ONE activation module is built per network and handed by reference to every
block, matching how fact_k2_global trained the curve being tested (tied across
neurons and across depth), exactly as in left.models.vit.ActViT.
"""
import torch
import torch.nn as nn

from left.activations.zoo import build_act
from left.models.vit import (DATASET_CFG, EMBED_DIM_DEFAULT, DEPTH_DEFAULT,
                      NUM_HEADS_DEFAULT, MLP_RATIO_DEFAULT, PatchEmbed, MLP)

#: phi(x) = elu(x) + 1 -- the non-negative feature map of Katharopoulos et al.
#: (2020). Non-negativity is what makes the normaliser (a dot product of phi's)
#: strictly positive, the linear-attention analogue of softmax's positivity.
def _phi(x):
    return torch.nn.functional.elu(x) + 1.0


class LinearMHSA(nn.Module):
    """Multi-head *linear* attention. Same projections and head layout as
    left.models.vit.MHSA; only the kernel differs (phi=elu+1 instead of softmax). The
    keys/values are contracted into a per-head (head_dim x head_dim) context
    and every query reads from it, so no N x N attention matrix is formed."""

    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        # Identical shapes to MHSA: the operator changes, the parameters do not.
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))               # 3, B, H, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _phi(q), _phi(k)
        # attn_drop acts on the value stream here (there is no N x N matrix to
        # drop); it is 0.0 in every run of this study, so this is a no-op on the
        # paths that matter and only keeps the knob faithful to MHSA.
        v = self.attn_drop(v)
        # Context: sum_j phi(K_j) (x) V_j  -> (B, H, d, e); and the key sum for
        # the normaliser: sum_j phi(K_j) -> (B, H, d).
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)                           # B, H, d
        num = torch.einsum("bhnd,bhde->bhne", q, kv)   # B, H, N, e
        den = torch.einsum("bhnd,bhd->bhn", q, k_sum)  # B, H, N
        out = num / (den.unsqueeze(-1) + self.eps)
        out = out.transpose(1, 2).reshape(B, N, C)     # B, N, C
        out = self.proj(out)
        return self.proj_drop(out)


class Block(nn.Module):
    """Pre-LN block, byte-identical to left.models.vit.Block except the attention is
    LinearMHSA. The FFN half IS left.models.vit.MLP (imported), so the shared
    activation lives in exactly the same place."""

    def __init__(self, dim, num_heads, d_ff, act, dropout=0.1, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearMHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, d_ff, act, dropout=dropout)

    def forward(self, x, return_acts=False):
        x = x + self.attn(self.norm1(x))
        if return_acts:
            y, h = self.ffn(self.norm2(x), return_acts=True)
            return x + y, h
        return x + self.ffn(self.norm2(x))


class ActLinViT(nn.Module):
    """left.models.vit.ActViT with LinearMHSA blocks. Same patch embed, CLS token,
    pos-embed, init, head and single shared activation."""

    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 act_kind="gelu", embed_dim=EMBED_DIM_DEFAULT,
                 depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
                 mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
                 use_cuda_kernel=True):
        super().__init__()
        self.act_kind = act_kind
        self.embed_dim = embed_dim
        d_ff = int(embed_dim * mlp_ratio)
        self.d_ff = d_ff

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Same name/placement as left.models.vit.ActViT.shared_act so the checkpoint
        # readers and activation-snapshot tooling find it unchanged.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, d_ff, self.shared_act,
                  dropout=dropout, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_acts=False):
        acts = [] if return_acts else None
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            if return_acts:
                x, h = blk(x, return_acts=True)
                acts.append(h)
            else:
                x = blk(x)
        x = self.norm(x)
        out = self.head(x[:, 0])
        return (out, acts) if return_acts else out


def build_linvit(dataset, act_kind, embed_dim=EMBED_DIM_DEFAULT,
                 depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
                 mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
                 use_cuda_kernel=True):
    cfg = DATASET_CFG[dataset]
    return ActLinViT(act_kind=act_kind, embed_dim=embed_dim, depth=depth,
                     num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                     attn_drop=attn_drop, use_cuda_kernel=use_cuda_kernel, **cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act = sum(p.numel() for p in model.shared_act.parameters() if p.requires_grad)
    ffn = 0
    seen = set()
    for blk in model.blocks:
        for p in blk.ffn.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                ffn += p.numel()
    return {"total": total, "trainable": train, "ffn_trainable": ffn,
            "act_trainable": act}
