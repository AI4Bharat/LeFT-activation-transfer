"""MLP-Mixer counterpart of left.models.vit's ~100K ViT -- identical for every
activation, with the activation module as the single injected difference.

WHY: the manuscript tests whether a nonlinearity learned-then-frozen in a ViT
transfers to an independently initialised target. The three targets measured so
far all mix tokens with *attention* -- global softmax (left.models.vit),
shifted-window softmax (left.models.swin), kernelised linear attention
(left.models.linvit). MLP-Mixer (Tolstikhin et al., 2021, arXiv:2105.01601) removes
attention entirely: tokens are mixed by a *plain MLP applied across the token
axis*, and channels by a second MLP applied across the channel axis. If the
frozen ImageNet-1k curve still helps here, the transfer result cannot be an
artifact of attention as a token mixer at all.

WHAT CHANGES, AND ONLY THIS, relative to the attention arms: the block is the
Mixer block -- no [CLS] token, no positional embedding (token order is carried
implicitly by the token-mixing MLP's weights), no attention. Each block is

    u = x + token_mlp(LN(x))        # mix across the S tokens, per channel
    y = u + channel_mlp(LN(u))      # mix across the C channels, per token

and the head is LayerNorm -> global average pool over tokens -> Linear, the
Mixer paper's classifier head.

WHAT IS HELD IDENTICAL to the other arms, so a GELU-vs-FAct comparison here is
an activation-only comparison exactly as it is for ViT/Swin/linvit:
  * PatchEmbed, the MLP sandwich (Linear -> act -> dropout -> Linear) and
    DATASET_CFG are IMPORTED from left.models.vit -- the shared activation sits in
    exactly the same Linear -> act -> dropout -> Linear site, now used twice per
    block (token mixer + channel mixer).
  * ONE activation module is built per network and handed by reference to every
    mixing MLP in every block, matching how fact_k2_global trained the curve
    being tested (tied across neurons AND across depth).
  * trunc_normal init (std=0.02, zero biases, LN at 1/0), pre-norm residual
    block order, dropout 0.1, depth 2, and the SAME per-dataset patchification
    as the ViT (so the token count S, and hence the attention-free sequence the
    mixer operates on, is identical to the ViT's patch grid).
  * No activation has learnable parameters, so train_target.py's equal-capacity
    assertion (act_trainable == 0) holds verbatim.

SIZE. Attention costs ~4*C^2 per block; the Mixer spends that budget on its two
mixing MLPs instead, so to land in this study's ~100-130K band at depth 2 the
hidden width is embed_dim=76 (vs the ViT's 64) with the paper's mixing ratios
(token hidden = 0.5*C, channel hidden = 4*C). The resulting trainable counts
track the ViT target almost exactly:

      fmnist    ~103,620   (28x28 -> 49 tokens, 1 chan, 10 classes)
      cifar10   ~108,362   (32x32 -> 64 tokens, 3 chan, 10 classes)
      cifar100  ~115,292   (32x32 -> 64 tokens, 3 chan, 100 classes)
      food101   ~126,313   (64x64 -> 64 tokens, patch 8, 3 chan, 101 classes)

(cf. ViT cifar10 108,106 / cifar100 113,956 -- see left.models.vit). The channel
mixer keeps the standard 4.0 ratio, so the activation keeps its usual share of
the network, as in every other arm.
"""
import torch
import torch.nn as nn

from left.activations.zoo import build_act
from left.models.vit import DATASET_CFG, PatchEmbed, MLP

#: Mixer defaults. embed_dim is 76 (not the ViT's 64) so the attention-free
#: budget lands in the ~100-130K band -- see the SIZE note above. depth 2 and
#: the two mixing ratios match the ViT's depth and FFN ratio.
EMBED_DIM_DEFAULT = 76
DEPTH_DEFAULT = 2
TOKEN_MLP_RATIO_DEFAULT = 0.5
CHANNEL_MLP_RATIO_DEFAULT = 4.0


class MixerBlock(nn.Module):
    """One Mixer layer: a token-mixing MLP then a channel-mixing MLP, each
    pre-normed and residual. Both MLPs are left.models.vit.MLP instances fed the SAME
    shared activation module, so the nonlinearity sits in exactly the call-site
    it occupies in the ViT/Swin/linvit FFN."""

    def __init__(self, num_tokens, dim, token_hidden, channel_hidden, act,
                 dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # token mixer: operates over the S-token axis (d_model = num_tokens)
        self.token_mlp = MLP(num_tokens, token_hidden, act, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        # channel mixer: operates over the C-channel axis (d_model = dim)
        self.channel_mlp = MLP(dim, channel_hidden, act, dropout=dropout)

    def forward(self, x, return_acts=False):
        """x: (B, S, C). return_acts=True additionally hands back this block's
        two POST-activation tensors (token mixer, channel mixer), the same hook
        the ViT block exposes -- one implementation, so the diagnostic path can
        never drift from the training path."""
        # token mixing: transpose so the MLP's last axis is the S tokens
        xn = self.norm1(x).transpose(1, 2)              # (B, C, S)
        if return_acts:
            tm, h_tok = self.token_mlp(xn, return_acts=True)
            x = x + tm.transpose(1, 2)
            y, h_ch = self.channel_mlp(self.norm2(x), return_acts=True)
            return x + y, (h_tok, h_ch)
        x = x + self.token_mlp(xn).transpose(1, 2)      # (B, S, C)
        x = x + self.channel_mlp(self.norm2(x))         # (B, S, C)
        return x


class ActMLPMixer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 act_kind="gelu", embed_dim=EMBED_DIM_DEFAULT,
                 depth=DEPTH_DEFAULT, token_mlp_ratio=TOKEN_MLP_RATIO_DEFAULT,
                 channel_mlp_ratio=CHANNEL_MLP_RATIO_DEFAULT, dropout=0.1,
                 use_cuda_kernel=True):
        super().__init__()
        self.act_kind = act_kind
        self.embed_dim = embed_dim
        self.token_hidden = int(round(embed_dim * token_mlp_ratio))
        self.channel_hidden = int(round(embed_dim * channel_mlp_ratio))
        self.d_ff = self.channel_hidden  # the activation's "FFN width", for logs

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_tokens = num_patches
        self.pos_drop = nn.Dropout(dropout)

        # Registered as `shared_act` so downstream tooling finds it in the same
        # place every other arm keeps its activation (see left.models.vit.ActViT).
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, embed_dim, self.token_hidden,
                       self.channel_hidden, self.shared_act, dropout=dropout)
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
        """return_acts=True additionally returns the POST-activation tensors of
        every mixing MLP, in (block, {token, channel}) order -- the hook a
        dead/saturated-unit diagnostic needs. Kept explicit so it costs nothing
        on the normal path."""
        acts = [] if return_acts else None
        x = self.patch_embed(x)          # (B, S, C) -- no CLS, no pos-embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            if return_acts:
                x, hs = blk(x, return_acts=True)
                acts.extend(hs)
            else:
                x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                # global average pool over tokens
        out = self.head(x)
        return (out, acts) if return_acts else out


def build_mlpmixer(dataset, act_kind, embed_dim=EMBED_DIM_DEFAULT,
                   depth=DEPTH_DEFAULT, token_mlp_ratio=TOKEN_MLP_RATIO_DEFAULT,
                   channel_mlp_ratio=CHANNEL_MLP_RATIO_DEFAULT, dropout=0.1,
                   use_cuda_kernel=True):
    cfg = DATASET_CFG[dataset]
    return ActMLPMixer(act_kind=act_kind, embed_dim=embed_dim, depth=depth,
                       token_mlp_ratio=token_mlp_ratio,
                       channel_mlp_ratio=channel_mlp_ratio, dropout=dropout,
                       use_cuda_kernel=use_cuda_kernel, **cfg)


def count_params(model):
    """Same return shape as left.models.vit.count_params. ffn_trainable sums BOTH
    mixing MLPs across all blocks (de-duplicated by parameter id, so a shared
    activation's buffers -- which aren't Parameters anyway -- can't double
    count); it is the Mixer analogue of the ViT's single per-block FFN."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act = sum(p.numel() for p in model.shared_act.parameters() if p.requires_grad)
    ffn = 0
    seen = set()
    for blk in model.blocks:
        for mlp in (blk.token_mlp, blk.channel_mlp):
            for p in mlp.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    ffn += p.numel()
    return {"total": total, "trainable": train, "ffn_trainable": ffn,
            "act_trainable": act}
