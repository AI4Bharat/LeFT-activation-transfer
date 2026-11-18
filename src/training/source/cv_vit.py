"""A small, from-scratch ViT (patchify -> [CLS] + learnable pos-embed ->
pre-LN Transformer encoder -> linear head).

The architecture (embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0,
patch/img size per dataset) is deliberately plain and was fixed before any
activation was compared on it, so no choice here was tuned for FAct. The one
non-obvious piece is that the MLP block is built by fourier_ffn.build_ffn
rather than a bespoke MLP class, which is what makes the FFN_VARIANTS menu --
standard and fact_k2_global -- selectable by string on one model.

fact_k2_global: ONE FourierActivation module is built once in
VisionTransformer.__init__ and handed by reference to every one of the
`depth` blocks' FFN. That sharing-by-reference is the mechanism the paper
studies; see fourier_ffn.build_ffn for the per-variant wiring.
"""
import torch
import torch.nn as nn

from fourier_ffn import build_ffn
from fourier_layers import FourierActivation, parse_global_k

DEPTH_DEFAULT = 6
NUM_HEADS_DEFAULT = 6
EMBED_DIM_DEFAULT = 192
MLP_RATIO_DEFAULT = 4.0

DATASET_CFG = {
    "fmnist": dict(img_size=28, patch_size=4, in_chans=1, n_classes=10),
    "cifar10": dict(img_size=32, patch_size=4, in_chans=3, n_classes=10),
    "cifar100": dict(img_size=32, patch_size=4, in_chans=3, n_classes=100),
    # img_size/patch_size match the standard ViT-Ti/16 ImageNet convention
    # (224x224 -> 14x14=196 patches); embed_dim/depth/num_heads/mlp_ratio
    # stay at this study's usual defaults (see build_vit) rather than scaling
    # up for ImageNet -- the point of this comparison is the activation
    # function, not architecture size, so it's held fixed across every
    # dataset.
    "imagenet1k": dict(img_size=224, patch_size=16, in_chans=3, n_classes=1000),
    # Food-101 is cached at 64x64 (cv_data.FOOD101_IMG_SIZE); patch=8 gives
    # an 8x8=64-patch grid, the same convention an earlier (much smaller)
    # Food-101 ViT used.
    "food101": dict(img_size=64, patch_size=8, in_chans=3, n_classes=101),
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class Block(nn.Module):
    """Pre-LN Transformer block: x + Attn(LN(x)); x + FFN(LN(x)).

    `ffn_kind` dispatches to fourier_ffn.build_ffn instead of hardcoding one
    MLP class, which is the only thing that makes this block activation-
    agnostic."""

    def __init__(self, dim, num_heads, ffn_kind, d_ff, dropout=0.1,
                 attn_drop=0.0, shared_act=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = build_ffn(ffn_kind, dim, d_ff, dropout, act=shared_act)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
                 num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                 ffn_kind="standard", dropout=0.1, attn_drop=0.0,
                 use_cuda_act=False):
        super().__init__()
        self.ffn_kind = ffn_kind
        self.embed_dim = embed_dim
        d_ff_standard = int(embed_dim * mlp_ratio)
        self.d_ff_standard = d_ff_standard

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # fact_kK_global (any K): ONE FourierActivation, tied across neurons
        # too, shared BY REFERENCE across every block's FFN. Sharing is by
        # object identity, so constructing one per block here would silently
        # make this a different experiment.
        shared_act = None
        k_global = parse_global_k(ffn_kind)
        if k_global is not None:
            if use_cuda_act:
                # The CUDA-kernel activation is fixed at K=2 (see
                # cuda_fact_k2/fact_k2_module.py) -- only swap it in when the
                # variant's own K matches, so a caller sweeping K doesn't
                # silently get a different K than requested.
                assert k_global == 2, (
                    f"use_cuda_act=True requires K=2, got fact_k{k_global}_global "
                    "(the CUDA kernel only supports K=2)")
                from cuda_fact_k2.fact_k2_module import FourierActivationK2CUDA
                shared_act = FourierActivationK2CUDA(d_ff_standard, shared=True)
            else:
                shared_act = FourierActivation(d_ff_standard, K=k_global, shared=True)
        self.shared_act = shared_act

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, ffn_kind, d_ff_standard, dropout=dropout,
                  attn_drop=attn_drop, shared_act=shared_act)
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

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_vit(dataset, ffn_kind, embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
              num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
              dropout=0.1, attn_drop=0.0, use_cuda_act=False):
    cfg = DATASET_CFG[dataset]
    return VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, ffn_kind=ffn_kind,
                              dropout=dropout, attn_drop=attn_drop,
                              use_cuda_act=use_cuda_act,
                              **cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # id()-based dedup (not name-based): the shared activation is the same
    # module in every block, so a name-substring filter would miscount it.
    seen, ffn_train = set(), 0
    for blk in model.blocks:
        for p in blk.ffn.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                ffn_train += p.numel()
    return {"total": total, "trainable": train, "ffn_trainable": ffn_train}
