"""The ~100K-parameter depth-2 Swin transformer this experiment trains -- the
Swin counterpart of left.models.vit's ViT-100K, identical for all 12 activations
with the activation module as the single injected difference.

WHY: the manuscript's Sections 4.2/4.3 (frozen transfer to an independently
initialized ViT-d2, under the common configuration and under per-activation
LR/WD tuning) are measured on a ViT target only. The transferred curve was
*learned* in a ViT, so a ViT-only target leaves open whether transferability
is a property of the curve or of the architecture. This file provides the
target the same two sections can be re-measured on.

Architecture (Liu et al. 2021, "Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows"), at Swin's own default patch size 4:

    patch=4  dim=64  depth=2 (stages=[2])  heads=4  mlp_ratio=4.0 (d_ff=256)
    window=4, block 0 = W-MSA (no shift), block 1 = SW-MSA (shift = window//2)
    relative position bias instead of an absolute pos-embed, no [CLS] token,
    global average pool over tokens before the linear head.

      fmnist    102,354 params   (28x28 -> 7x7 token grid, 1 chan,  10 classes)
      cifar10   104,402 params   (32x32 -> 8x8 token grid,  3 chan,  10 classes)
      cifar100  110,252 params   (32x32 -> 8x8 token grid,  3 chan, 100 classes)
      food101   110,317 params   (64x64 -> 16x16 token grid, 3 chan, 101 classes)

Three things are held byte-identical to left.models.vit so that a Swin-vs-ViT
difference is attributable to the attention mechanism and not to the FFN or
the training surface:

  1. The FFN half IS left.models.vit.MLP, imported, not re-implemented: the shared
     activation sits in exactly the same Linear -> act -> dropout -> Linear
     sandwich, with the same d_ff, at the same number of call sites (depth).
  2. ONE activation module per network, handed by reference to every block
     (`shared_act`), matching how the transferred curve was trained and where
     every other driver in this study looks for it.
  3. Weight init (trunc_normal_ std=0.02, zero biases, LayerNorm 1/0) and
     dropout placement (attn_drop=0, proj_drop=dropout, FFN-internal dropout)
     are the ViT's.

DEPTH 2 = ONE STAGE. `stages=[2]` gives two transformer blocks and therefore
two activation call sites, the same as the depth-2 ViT. No PatchMerging runs
at this depth (it would need >=2 stages), so the token grid is constant and
the hierarchy is a no-op here; PatchMerging is implemented anyway so the same
file serves a deeper sweep without a second code path.

PADDING, the one place this deviates from a naive port: Fashion-MNIST's 28px
input at patch 4 gives a 7x7 token grid, and 7 is not a multiple of the
window size 4. Official Swin pads the feature map up to a multiple of the
window and masks; we do the same, padding 7 -> 8 and isolating the pad tokens
with the attention mask (see SwinBlock._mask), so a real token never attends
to a pad token in either the shifted or the unshifted block, and the padded
positions are cropped off before the residual add. The alternative --
window=7 on a 7x7 grid -- would collapse to a single window and remove
shifted-window attention from that dataset entirely, which is the one thing
the architecture is being tested for.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from left.activations.zoo import build_act
from left.models.vit import MLP  # the ViT's FFN, verbatim -- see docstring point 1

EMBED_DIM_DEFAULT = 64
DEPTH_DEFAULT = 2
NUM_HEADS_DEFAULT = 4
MLP_RATIO_DEFAULT = 4.0
WINDOW_DEFAULT = 4
#: Swin's own default patch size, and the point of this experiment: unlike
#: left.models.vit.DATASET_CFG (which uses patch 8 on the 64px datasets to hold the
#: sequence length at 64), every dataset here is patchified at Swin's native 4.
PATCH_DEFAULT = 4

DATASET_CFG = {
    "fmnist": dict(img_size=28, patch_size=4, in_chans=1, n_classes=10),
    "cifar10": dict(img_size=32, patch_size=4, in_chans=3, n_classes=10),
    "cifar100": dict(img_size=32, patch_size=4, in_chans=3, n_classes=100),
    # 64px at patch 4 -> a 16x16 = 256-token grid, 4x the ViT target's 64
    # tokens. Affordable precisely because windowed attention is linear in
    # the token count: 16 windows of 16 tokens, not one 256-token softmax.
    "food101": dict(img_size=64, patch_size=4, in_chans=3, n_classes=101),
}


class PatchEmbed(nn.Module):
    """Conv patchify + LayerNorm, Swin's own stem (the ViT's has no norm)."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


def window_partition(x, ws):
    """(B,H,W,C) -> (B*nW, ws*ws, C); nW = (H/ws)(W/ws) disjoint windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse(w, ws, H, W, B):
    """(B*nW, ws*ws, C) -> (B,H,W,C)"""
    x = w.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    """W-MSA: multi-head self-attention inside one window, plus Swin's learned
    relative position bias (Swin Eq. 4) -- (2M-1)^2 biases per head indexed by
    the query/key relative offset. This is what replaces the ViT's absolute
    pos-embed, and it is the only parameter group Swin adds.

    Dropout placement follows left.models.vit.MHSA: none on the attention matrix,
    `proj_drop` on the output projection."""

    def __init__(self, dim, ws, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.ws, self.heads = ws, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws),
                                       indexing="ij")).flatten(1)          # 2,N
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (ws - 1)   # N,N,2
        self.register_buffer("rpi", rel[..., 0] * (2 * ws - 1) + rel[..., 1])

    def forward(self, x, mask=None):
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        att = (q * self.scale) @ k.transpose(-2, -1)                  # BnW,h,N,N
        att = att + self.rpb[self.rpi.reshape(-1)].view(N, N, -1).permute(2, 0, 1)
        if mask is not None:
            nW = mask.shape[0]
            att = att.view(BnW // nW, nW, self.heads, N, N) + mask[:, None]
            att = att.view(-1, self.heads, N, N)
        att = self.attn_drop(att.softmax(-1))
        out = (att @ v).transpose(1, 2).reshape(BnW, N, C)
        return self.proj_drop(self.proj(out))


class SwinBlock(nn.Module):
    """One Swin block: (S)W-MSA + FFN, pre-norm, both residual.

    Even blocks use a plain window partition (W-MSA); odd blocks shift it by
    window//2 (SW-MSA) with a cyclic roll plus an attention mask, so tokens
    that wrapped around the image edge cannot attend to each other. The FFN
    half is left.models.vit.MLP with the network's single shared activation."""

    def __init__(self, dim, res, heads, ws, shift, d_ff, act, dropout=0.1,
                 attn_drop=0.0):
        super().__init__()
        self.res = res
        # A grid no larger than the window is a single global window, and a
        # shift of it is the same partition -- so collapse to W-MSA.
        self.ws, self.shift = (res, 0) if res <= ws else (ws, shift)
        self.pad = (self.ws - res % self.ws) % self.ws
        self.resp = res + self.pad
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.ws, heads, attn_drop=attn_drop,
                                    proj_drop=dropout)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, d_ff, act, dropout=dropout)
        self.register_buffer("attn_mask", self._mask(), persistent=False)

    def _mask(self):
        """Attention mask over the padded grid, combining two exclusions in one
        region map: (i) Swin's shift regions -- the 3x3 spans whose tokens were
        brought together by the cyclic roll and must not mix; (ii) the pad
        tokens, given their own region id (-1) so no real token attends to one.
        Returns None when neither applies (no shift and no padding)."""
        if not self.shift and not self.pad:
            return None
        Hp = self.resp
        region = torch.zeros(1, Hp, Hp, 1)
        if self.shift:
            cnt = 0
            spans = (slice(0, -self.ws), slice(-self.ws, -self.shift),
                     slice(-self.shift, None))
            for h in spans:
                for w in spans:
                    region[:, h, w, :] = cnt
                    cnt += 1
        if self.pad:
            valid = torch.zeros(1, Hp, Hp, 1)
            valid[:, :self.res, :self.res, :] = 1.0
            region = region.masked_fill(valid == 0, -1.0)
        if self.shift:
            # roll the region map exactly as the features are rolled, so the
            # mask describes the partition the attention actually sees
            region = torch.roll(region, (-self.shift, -self.shift), dims=(1, 2))
        mw = window_partition(region, self.ws).squeeze(-1)               # nW,N
        m = mw[:, None, :] - mw[:, :, None]
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x, return_acts=False):
        B, L, C = x.shape
        H = W = self.res
        h = self.n1(x).view(B, H, W, C)
        if self.pad:
            h = F.pad(h, (0, 0, 0, self.pad, 0, self.pad))
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), dims=(1, 2))
        h = window_reverse(self.attn(window_partition(h, self.ws), self.attn_mask),
                           self.ws, self.resp, self.resp, B)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), dims=(1, 2))
        if self.pad:
            h = h[:, :H, :W, :]
        x = x + h.reshape(B, L, C)
        if return_acts:
            y, a = self.ffn(self.n2(x), return_acts=True)
            return x + y, a
        return x + self.ffn(self.n2(x))


class PatchMerging(nn.Module):
    """Swin's between-stage downsample: concatenate each 2x2 token group (4C)
    and project to 2C -- resolution halves, channels double. Unused at depth 2
    (one stage), kept so a deeper sweep needs no second code path."""

    def __init__(self, dim, res):
        super().__init__()
        self.res = res
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H = W = self.res
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.reshape(B, -1, 4 * C)))


class ActSwin(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 act_kind="gelu", embed_dim=EMBED_DIM_DEFAULT,
                 depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
                 mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
                 window=WINDOW_DEFAULT, stages=None, use_cuda_kernel=True):
        super().__init__()
        self.act_kind = act_kind
        self.embed_dim = embed_dim
        d_ff = int(embed_dim * mlp_ratio)
        self.d_ff = d_ff
        self.window = window
        #: default = one stage holding every block, i.e. no PatchMerging, so
        #: depth-2 Swin has the same two activation call sites as depth-2 ViT
        self.stages = list(stages) if stages else [depth]
        assert sum(self.stages) == depth, (self.stages, depth)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        # Registered as `shared_act` so every reader in this study (checkpoint
        # tooling, the param assertions, the coefficient snapshots) finds the
        # activation exactly where left.models.vit.ActViT keeps it.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        res, d, h = self.patch_embed.grid, embed_dim, num_heads
        self.blocks, self.layers = nn.ModuleList(), nn.ModuleList()
        for si, nb in enumerate(self.stages):
            for bi in range(nb):
                blk = SwinBlock(d, res, h, window, window // 2 if bi % 2 else 0,
                                int(d * mlp_ratio), self.shared_act,
                                dropout=dropout, attn_drop=attn_drop)
                self.blocks.append(blk)
                self.layers.append(blk)
            if si < len(self.stages) - 1:
                self.layers.append(PatchMerging(d, res))
                res //= 2
                d *= 2
                h *= 2          # heads double with channels, head_dim fixed
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """left.models.vit.ActViT._init_weights, unchanged. Does not touch the
        relative-position-bias Parameter, which WindowAttention already
        trunc_normal_'d at std=0.02 (the same std)."""
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
        """return_acts=True additionally returns the POST-activation tensor of
        every FFN site in block order, the same contract as ActViT.forward."""
        acts = [] if return_acts else None
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            if return_acts and isinstance(layer, SwinBlock):
                x, a = layer(x, return_acts=True)
                acts.append(a)
            else:
                x = layer(x)
        out = self.head(self.norm(x).mean(1))       # global average pool
        return (out, acts) if return_acts else out


def build_swin(dataset, act_kind, embed_dim=EMBED_DIM_DEFAULT,
               depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
               mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
               window=WINDOW_DEFAULT, stages=None, patch_size=None,
               use_cuda_kernel=True):
    cfg = dict(DATASET_CFG[dataset])
    if patch_size is not None:
        cfg["patch_size"] = patch_size
    return ActSwin(act_kind=act_kind, embed_dim=embed_dim, depth=depth,
                   num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                   attn_drop=attn_drop, window=window, stages=stages,
                   use_cuda_kernel=use_cuda_kernel, **cfg)


def count_params(model):
    """left.models.vit.count_params's contract, on this model's block list."""
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
