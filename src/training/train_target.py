"""Train a ~100K model on FMNIST / CIFAR-10 / CIFAR-100 / Food-101, one
activation per run.

Two architectures share this trainer, selected with `--arch`:

  vit  -- the ~100K ViT of left.models.vit (embed=64, depth=2, heads=4,
           mlp_ratio=4.0) that produced the paper's target-scale tables.
  swin -- the parameter-/token-matched depth-2 Swin of left.models.swin: the same
           FFN sandwich and the same single shared activation, with
           shifted-window attention in place of global attention, at Swin's
           native patch 4. It re-measures the transfer result under a
           different attention mechanism, since the curve was learned in a ViT.

Both record `arch` in every checkpoint's config, and run_name builds the
filename prefix from it, so the two arms share one results tree.

What varies across runs is the activation (--act), the dataset and the seed.
Optimizer, schedule, batch size, augmentation, validation split, checkpoint
selection and the equal-capacity assertion are shared code, so an activation
ranking measured on one dataset is measured the same way on the others.

Training recipe is a faithful port of the main study's cv_train.train_one, so
these numbers sit on the same footing as every other CV run in this repo:

    AdamW(lr=1e-3, weight_decay=0.05), batch 256, label smoothing 0.1,
    5 warmup epochs then a cosine arc down to 1% of base lr,
    pad-and-random-crop + horizontal flip augmentation (cv_data.py),
    5000-image validation split carved out of train with a FIXED seed (42),
    best-val checkpoint selection, top-1 reported on the real test set.
    100 epochs on every dataset -- the budget of all published target runs.

Three deliberate departures from cv_train.train_one, all applied identically to
every variant so they cannot favour one activation:

  1. NO automatic mixed precision -- fp32 everywhere. Devices differ in which
     low-precision format is fast (bf16 on some, fp16 only on others), and a
     periodic activation evaluated in bf16 is not the same function as one
     evaluated in fp16. Running fp32 keeps a number computed on one device
     directly comparable to a number computed on another. At ~100K parameters
     the throughput cost is negligible.
  2. The model is left.models.vit.build_vit (activation injected as a module), not
     cv_vit.build_vit (activation selected by FFN-variant string).
  3. Only two checkpoints per run are written -- best-val and last -- per the
     study instruction. No per-epoch checkpoints.

Everything needed to plot convergence (per-epoch train/val loss and accuracy,
the LR trace, wall-clock) is written to <run>_result.json.
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy("file_system")

from left.utils.config import apply_config
from left.utils.data import get_dataloaders
from left.activations.zoo import ACT_KINDS, act_param_count
from left.models.vit import (DEPTH_DEFAULT, EMBED_DIM_DEFAULT, MLP_RATIO_DEFAULT,  # noqa: E402
                      NUM_HEADS_DEFAULT, build_vit)
from left.models.vit import count_params as vit_count_params
from left.models.swin import (WINDOW_DEFAULT as SWIN_WINDOW_DEFAULT,  # noqa: E402
                       PATCH_DEFAULT as SWIN_PATCH_DEFAULT, build_swin)
from left.models.swin import count_params as swin_count_params
from left.models.linvit import build_linvit
from left.models.linvit import count_params as linvit_count_params
from left.models.mlp_mixer import build_mlpmixer
from left.models.mlp_mixer import count_params as mlpmixer_count_params
from left.models.mlp_mixer import EMBED_DIM_DEFAULT as MIXER_EMBED_DEFAULT

ARCHS = ("vit", "swin", "linvit", "mlpmixer")

#: Epoch budget. Every published target run, on all four datasets and all four
#: architectures, trained for 100 epochs; the per-dataset budgets of
#: cv_train.EPOCHS (20/40/60/60) belong to the source study and were not used
#: here. A shorter budget can invert a ranking, so this is not a free choice
#: when comparing against a published number.
EPOCHS_DEFAULT = 100
EPOCHS_BY_DATASET = {d: EPOCHS_DEFAULT
                     for d in ("fmnist", "cifar10", "cifar100", "food101")}
#: Dropout, the source experiment's 0.1, applied identically to every entry
#: in left.activations.zoo.ACT_KINDS so the equal-capacity comparison is unaffected.
DROPOUT_DEFAULT = 0.1
#: Filename prefix per architecture, so all arms share one results tree.
NAME_PREFIX = {"vit": "vit100k", "swin": "swin100k", "linvit": "linvit100k",
               "mlpmixer": "mlpmixer100k"}
DATASETS = ("fmnist", "cifar10", "cifar100", "food101")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    """Byte-identical to cv_train.cosine_warmup_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def run_name(dataset, act_kind, seed, arch="vit"):
    return f"{NAME_PREFIX[arch]}_{dataset}_{act_kind}_seed{seed}"


@torch.no_grad()
def evaluate(model, loader, device, lossf):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = lossf(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_one(dataset, act_kind, data_root, out_dir, epochs=None, seed=0,
              batch_size=256, lr=1e-3, wd=0.05, warmup_epochs=5,
              label_smoothing=0.1, dropout=None, num_workers=4,
              embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
              num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
              use_cuda_kernel=True, device=None, log=print,
              log_every=50, arch="vit", window=SWIN_WINDOW_DEFAULT,
              patch_size=None, stages=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or EPOCHS_BY_DATASET[dataset]
    dropout = DROPOUT_DEFAULT if dropout is None else dropout
    set_seed(seed)

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset, data_root, batch_size=batch_size, num_workers=num_workers)

    if arch == "vit":
        model = build_vit(dataset, act_kind, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                          use_cuda_kernel=use_cuda_kernel).to(device)
        pc = vit_count_params(model)
        shape_str = (f"embed_dim={embed_dim} depth={depth} heads={num_heads} "
                     f"mlp_ratio={mlp_ratio}")
    elif arch == "linvit":  # see left.models.linvit; same ViT, linear attention
        model = build_linvit(dataset, act_kind, embed_dim=embed_dim, depth=depth,
                             num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                             use_cuda_kernel=use_cuda_kernel).to(device)
        pc = linvit_count_params(model)
        shape_str = (f"embed_dim={embed_dim} depth={depth} heads={num_heads} "
                     f"mlp_ratio={mlp_ratio} attn=linear(elu+1)")
    elif arch == "mlpmixer":  # see left.models.mlp_mixer; no attention, two mixing
        # MLPs per block, the shared activation in both. --mlp-ratio is the
        # channel mixer's ratio (the paper's 4.0); the token mixer's ratio is
        # left.models.mlp_mixer.TOKEN_MLP_RATIO_DEFAULT (0.5).
        model = build_mlpmixer(dataset, act_kind, embed_dim=embed_dim,
                               depth=depth, channel_mlp_ratio=mlp_ratio,
                               dropout=dropout,
                               use_cuda_kernel=use_cuda_kernel).to(device)
        pc = mlpmixer_count_params(model)
        shape_str = (f"embed_dim={embed_dim} depth={depth} "
                     f"token_hidden={model.token_hidden} "
                     f"channel_hidden={model.channel_hidden} "
                     f"patch={model.patch_embed.proj.kernel_size[0]} "
                     f"grid={model.patch_embed.grid} "
                     f"tokens={model.patch_embed.num_patches}")
    else:  # swin -- see left.models.swin; FFN and shared activation held identical
        model = build_swin(dataset, act_kind, embed_dim=embed_dim, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio,
                           dropout=dropout, window=window, stages=stages,
                           patch_size=patch_size,
                           use_cuda_kernel=use_cuda_kernel).to(device)
        pc = swin_count_params(model)
        shape_str = (f"embed_dim={embed_dim} depth={depth} heads={num_heads} "
                     f"mlp_ratio={mlp_ratio} patch={model.patch_embed.proj.kernel_size[0]} "
                     f"grid={model.patch_embed.grid} window={window} "
                     f"stages={model.stages} tokens={model.patch_embed.num_patches}")
    # The equal-capacity guarantee the ACT_KINDS comparison rests on: no
    # activation in ACT_KINDS may contribute a single learnable parameter, so
    # every variant's ViT has the identical parameter count. Asserted per run
    # rather than trusted, because it is the one way an "activation-only"
    # comparison can silently stop being one. "fact_k2_global" (Evolving
    # FAct) is the deliberate, documented exception -- see left.activations.zoo --
    # and is asserted against 5 (a0, a1, b1, a2, b2) instead of 0.
    expect_act_params = 5 if act_kind == "fact_k2_global" else 0
    assert pc["act_trainable"] == expect_act_params, (
        f"activation {act_kind!r} contributed {pc['act_trainable']} learnable "
        f"parameters, expected {expect_act_params}")

    name = run_name(dataset, act_kind, seed, arch)
    log(f"[{name}] arch={arch} device={device} params total={pc['total']:,} "
        f"trainable={pc['trainable']:,} ffn={pc['ffn_trainable']:,} "
        f"act={pc['act_trainable']} "
        f"{shape_str} epochs={epochs} bs={batch_size} lr={lr} wd={wd} "
        f"dropout={dropout} amp=off(fp32) cuda_kernel={use_cuda_kernel} "
        f"n_train={meta['n_train']} n_val={meta['n_val']} n_test={meta['n_test']}")
    log(f"[{name}] activation: {model.shared_act}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "epoch_wall_s": []}
    best_val_acc, best_state, best_epoch = -1.0, None, -1
    global_step, lr_now = 0, lr
    t0 = time.time()

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run_config = dict(arch=arch, dataset=dataset, act_kind=act_kind, seed=seed, epochs=epochs,
                      batch_size=batch_size, lr=lr, weight_decay=wd,
                      warmup_epochs=warmup_epochs, label_smoothing=label_smoothing,
                      dropout=dropout, optimizer="adamw",
                      lr_scheduler="cosine_warmup", amp=False,
                      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, use_cuda_kernel=use_cuda_kernel,
                      window=window if arch == "swin" else None,
                      patch_size=(model.patch_embed.proj.kernel_size[0]
                                  if arch in ("swin", "mlpmixer") else None),
                      stages=(list(model.stages) if arch == "swin" else None),
                      params=pc)

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for step_in_ep, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, lr)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x.size(0)
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_n += x.size(0)
            global_step += 1
            if log_every and (step_in_ep % log_every == 0 or step_in_ep == 1):
                log(f"[{name}] epoch {ep} step {step_in_ep}/{steps_per_epoch} "
                    f"loss={loss.item():.4f} ({time.time() - t0:.1f}s elapsed)")

        train_loss, train_acc = ep_loss / ep_n, ep_correct / ep_n
        val_loss, val_acc = evaluate(model, val_loader, device, lossf)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["epoch_wall_s"].append(time.time() - ep_t0)
        log(f"[{name}] epoch {ep}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} lr={lr_now:.2e} ({time.time() - t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # Flushed to disk the moment it improves, so a crash mid-run still
            # leaves a usable best checkpoint (test_acc is filled in below).
            torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                        "epoch": ep, "config": run_config},
                       os.path.join(ckpt_dir, f"{name}_best.pt"))

    train_time_s = time.time() - t0
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss_best, test_acc_best = evaluate(model, test_loader, device, lossf)
    model.load_state_dict(final_state)
    test_loss_final, test_acc_final = evaluate(model, test_loader, device, lossf)

    torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                "epoch": best_epoch, "test_acc": test_acc_best,
                "config": run_config}, os.path.join(ckpt_dir, f"{name}_best.pt"))
    torch.save({"state_dict": final_state, "epoch": epochs,
                "test_acc": test_acc_final, "config": run_config},
               os.path.join(ckpt_dir, f"{name}_last.pt"))

    result = {
        "name": name, "arch": arch, "dataset": dataset, "act_kind": act_kind, "seed": seed,
        "epochs": epochs, "params": pc, "meta": meta, "config": run_config,
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "test_acc": test_acc_best, "test_loss": test_loss_best,
        "test_acc_last_epoch": test_acc_final, "test_loss_last_epoch": test_loss_final,
        "wall_s": train_time_s, "history": history,
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, f"{name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log(f"[{name}] DONE test_acc(best-val)={test_acc_best:.4f} "
        f"test_acc(last)={test_acc_final:.4f} best_val_acc={best_val_acc:.4f} "
        f"@epoch{best_epoch} time={train_time_s:.1f}s")

    del train_loader, val_loader, test_loader, opt, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--arch", default="vit", choices=ARCHS)
    ap.add_argument("--act", required=True,
                    choices=list(ACT_KINDS) + ["fact_k2_global",
                                               "fact_k2_global_frozen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05,
                    help="AdamW weight decay. The common configuration uses 0.05; the "
                         "activation-specific condition searches it jointly with --lr "
                         "over {0, 0.05, 0.2}, so reproducing those runs needs this.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root",
                    default=os.environ.get("LEFT_DATA_ROOT",
                                           os.environ.get("FACT_DATA_ROOT",
                                                          os.path.expanduser("~/datasets"))),
                    help="dataset root (default: $LEFT_DATA_ROOT, else "
                         "$FACT_DATA_ROOT, else ~/datasets)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact", default=None,
                    help="coefficient artifact JSON to transfer (see "
                         "artifacts/fact/). Only meaningful with "
                         "--act fact_fixed, which otherwise reads the "
                         "repository's default artifact; sets "
                         "$LEFT_FACT_COEFFS for this run.")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="default is arch-specific: 64 for vit/swin/linvit, "
                         f"{MIXER_EMBED_DEFAULT} for mlpmixer (so the "
                         "attention-free Mixer lands in the same ~100-130K "
                         "band, see left.models.mlp_mixer)")
    ap.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    ap.add_argument("--num-heads", type=int, default=NUM_HEADS_DEFAULT)
    ap.add_argument("--mlp-ratio", type=float, default=MLP_RATIO_DEFAULT)
    ap.add_argument("--window", type=int, default=SWIN_WINDOW_DEFAULT,
                    help="swin only: window size M (grids not divisible by M "
                         "are zero-padded and masked, see left.models.swin)")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="swin only: override left.models.swin's DATASET_CFG patch "
                         f"size (default {SWIN_PATCH_DEFAULT}, Swin's own)")
    ap.add_argument("--stages", type=int, nargs="+", default=None,
                    help="swin only: blocks per stage, must sum to --depth; "
                         "default = one stage holding every block (no "
                         "PatchMerging), which keeps the activation call-site "
                         "count equal to the ViT's depth")
    ap.add_argument("--dropout", type=float, default=None,
                    help=f"override the default ({DROPOUT_DEFAULT})")
    ap.add_argument("--no-cuda-kernel", action="store_true",
                    help="evaluate fact_fixed in pure PyTorch instead of the "
                         "custom CUDA kernel (reference path; no effect on the "
                         "ten builtin activations)")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="linear warmup before the cosine arc (the published "
                         "recipe's value; shared by every activation)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50)
    # `--config configs/...yaml` supplies defaults; explicit flags still win.
    args = apply_config(ap)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.artifact:
        # fact_fixed reads its curve from $LEFT_FACT_COEFFS (see
        # left.activations.fact_fixed), so a run can transfer any artifact
        # without editing code.
        os.environ["LEFT_FACT_COEFFS"] = os.path.abspath(args.artifact)
    if args.embed_dim is None:
        args.embed_dim = (MIXER_EMBED_DEFAULT if args.arch == "mlpmixer"
                          else EMBED_DIM_DEFAULT)
    expect = 5 if args.act == "fact_k2_global" else 0
    assert act_param_count(args.act, not args.no_cuda_kernel) == expect
    train_one(args.dataset, args.act, args.data_root, args.out_dir,
              epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
              lr=args.lr, wd=args.wd, warmup_epochs=args.warmup_epochs,
              label_smoothing=args.label_smoothing,
              num_workers=args.num_workers,
              embed_dim=args.embed_dim, depth=args.depth,
              num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
              use_cuda_kernel=not args.no_cuda_kernel,
              log_every=args.log_every, arch=args.arch, dropout=args.dropout,
              window=args.window, patch_size=args.patch_size,
              stages=tuple(args.stages) if args.stages else None)


if __name__ == "__main__":
    main()
