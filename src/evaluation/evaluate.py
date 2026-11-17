"""Evaluate a trained checkpoint on the real test set, or aggregate a results
tree into the paper's tables.

Two entry points:

    # one checkpoint, recomputed from scratch
    python -m left.evaluation.evaluate --checkpoint runs/…_best.pt

    # a whole results tree -> mean+-sd and Holm-corrected paired tests
    python -m left.evaluation.evaluate --aggregate results/logs --arch vit \
        --arm fact_fixed --baseline gelu

The trainers already write the test accuracy into `<run>_result.json`, so the
aggregate path never retrains; the single-checkpoint path exists so a reader
can verify a reported number against the weights rather than the log.
"""
import argparse
import glob
import json
import os

import torch
import torch.nn as nn

from left.evaluation.metrics import compare_arms, mean_sd
from left.utils.data import get_dataloaders


def _build(config):
    arch = config.get("arch", "vit")
    if arch == "vit":
        from left.models.vit import build_vit as build
    elif arch == "swin":
        from left.models.swin import build_swin as build
    elif arch == "linvit":
        from left.models.linvit import build_linvit as build
    elif arch == "mlpmixer":
        from left.models.mlp_mixer import build_mlpmixer as build
    else:
        raise ValueError(f"unknown arch {arch!r}")
    # training writes the activation under "act_kind"; older configs used "act"
    act = config.get("act_kind", config.get("act"))
    if act is None:
        raise ValueError("config names no activation (looked for 'act_kind' and 'act')")
    # Rebuild the exact geometry the checkpoint was trained with -- a run from
    # the depth sweep is not depth 2, and loading it into the default geometry
    # fails with "unexpected key(s) in state_dict". Only keys the builder
    # actually accepts, and only where the config states a value, are passed.
    import inspect
    sig = inspect.signature(build).parameters
    geom = {k: v for k, v in config.items()
            if k in sig and k not in ("dataset", "act_kind", "act") and v is not None}
    if not torch.cuda.is_available():
        geom["use_cuda_kernel"] = False
    return build(config["dataset"], act, **geom)


@torch.no_grad()
def evaluate_loader(model, loader, device="cuda"):
    """Top-1 and mean cross-entropy over `loader`."""
    model.eval().to(device)
    lossf = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += lossf(logits, y).item() * y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total), loss_sum / max(1, total)


def evaluate_checkpoint(path, data_root=None, device="cuda", batch_size=256):
    from left.utils.checkpoint import load_checkpoint
    sd, config, blob = load_checkpoint(path)
    if not config:
        raise ValueError(f"{path} has no config block; cannot rebuild the model")
    model = _build(config)
    model.load_state_dict(sd, strict=True)
    loaders = get_dataloaders(config["dataset"], data_root=data_root,
                              batch_size=batch_size)
    test_loader = loaders[2]   # (train, val, test[, meta])
    acc, loss = evaluate_loader(model, test_loader, device=device)
    return {"checkpoint": os.path.basename(path), "config": config,
            "test_top1": acc, "test_loss": loss,
            "reported_test_acc": blob.get("test_acc")}


def collect(results_dir, arch="vit", pattern="*_result.json"):
    """Walk a results tree and return `{dataset: {act: {seed: test_top1}}}`."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "**", pattern), recursive=True):
        with open(path) as f:
            r = json.load(f)
        cfg = r.get("config", r)
        if cfg.get("arch", "vit") != arch:
            continue
        acc = r.get("test_acc", r.get("test_top1"))
        if acc is None:
            continue
        # train_target records the activation as `act_kind`; `act` is accepted
        # too, so a results tree written by either name aggregates.
        act = cfg.get("act", cfg.get("act_kind"))
        out.setdefault(cfg["dataset"], {}).setdefault(act, {})[int(cfg["seed"])] = acc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--aggregate", help="results directory to walk")
    ap.add_argument("--arch", default="vit",
                    choices=("vit", "swin", "linvit", "mlpmixer"))
    ap.add_argument("--arm", default="fact_fixed")
    ap.add_argument("--baseline", default="gelu")
    ap.add_argument("--data-root")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.checkpoint:
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data_root,
                                             args.device), indent=2))
    if args.aggregate:
        res = collect(args.aggregate, arch=args.arch)
        if not res:
            raise SystemExit(f"no {args.arch} results under {args.aggregate}")
        for ds in sorted(res):
            print(f"\n{ds}")
            for act in sorted(res[ds]):
                m, sd = mean_sd(res[ds][act].values())
                print(f"  {act:<16} {100*m:6.2f} +/- {100*sd:4.2f}  "
                      f"(n={len(res[ds][act])})")
        usable = {ds: v for ds, v in res.items()
                  if args.arm in v and args.baseline in v}
        if usable:
            print(f"\n{args.arm} vs {args.baseline} "
                  f"(paired over seeds, Holm over {len(usable)} datasets)")
            rows = compare_arms(usable, args.arm, args.baseline)
            for ds in sorted(rows):
                r = rows[ds]
                print(f"  {ds:<10} {r['margin_pp']:+6.2f}pp  t={r['t']:+6.2f}  "
                      f"p={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f}  "
                      f"{'significant' if r['significant'] else 'ns'}")


if __name__ == "__main__":
    main()
