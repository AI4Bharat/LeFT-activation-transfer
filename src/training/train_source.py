"""Learn -- step 1 of Learn, Freeze, Transfer: train the source network with a
learnable, globally shared activation.

This is the expensive step and the only one that needs ImageNet-1K. The paper's
source run is a depth-6, ~3M-parameter ViT trained for 100 epochs on
ImageNet-1K with one `FAct` K=2 curve (five scalars) shared across every block,
initialised at the Fourier fit of GELU and learned jointly with the weights:

    python -m left.training.train_source \\
        --dataset imagenet1k --variants fact_k2_global --seeds 1 \\
        --epochs 100 --out-dir runs/source

**Most readers do not need to run this.** The learned coefficients are already
in `artifacts/fact/imagenet1k_vitd6_seed1.json`, which is all the transfer step
consumes -- see `docs/reproduction.md`. Reproduce the source run only if the
question is whether the *same curve* is learned again (seeds 1-5 of the
published run agree to about three decimal places on all five coefficients;
those values are recorded in the artifact under `all_seeds_converged`).

Implementation: every run is executed by `left.training.source.cv_train`'s
`train_one` -- the published trainer, vendored verbatim -- so the recipe cannot
drift from the one the manuscript reports. Only the loop over variants and
seeds lives here; the vendored `cv_run.py` holds the same loop but its dataset
list predates the ImageNet-1K runs, and it writes its checkpoints beside the
installed package rather than under `--out-dir`.

`--variants` selects the activation in the source parameterisation
(`fact_k2_global` is the paper's learnable shared Fourier arm, `gelu` its
baseline, `pau_global` the shared Pade control).

Outputs, all under `--out-dir`:

    checkpoints/cv_<dataset>_<variant>_seed<n>_best.pt    best validation epoch
    checkpoints/cv_<dataset>_<variant>_seed<n>_final.pt   last epoch
    cv_<dataset>_source.json                              one entry per run
    cv_<dataset>_<variant>_seed<n>_coeff_traj.json         per-epoch coefficients

The published artifact was extracted from the **best-validation** checkpoint:

    python -m left.transfer.extract_cli \\
        --checkpoint runs/source/checkpoints/cv_imagenet1k_fact_k2_global_seed1_best.pt \\
        --out artifacts/fact/my_source_run.json
"""
import argparse
import json
import os
import time

from left.training.source import cv_train

#: The datasets the vendored trainer accepts (cv_train.py's own list).
DATASETS = ("fmnist", "cifar10", "cifar100", "food101", "imagenet1k")


def _coeff_hook(traj):
    """Record the shared curve after every epoch, as the published run did."""
    def hook(epoch, model, val_acc, val_loss):
        act = getattr(model, "shared_act", None)
        if act is None or not hasattr(act, "a0"):
            return
        traj.append({"epoch": epoch, "val_acc": val_acc,
                     "a0": act.a0.detach().cpu().tolist(),
                     "a": act.a.detach().cpu().tolist(),
                     "b": act.b.detach().cpu().tolist()})
    return hook


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--variants", nargs="+", default=["fact_k2_global"],
                    help="source-parameterisation activations to train "
                         "(default: the paper's fact_k2_global)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1],
                    help="default: 1, the seed the published artifact came from")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override cv_train's per-dataset default epoch count "
                         "(the published ImageNet-1K run used 100)")
    ap.add_argument("--data-root",
                    default=os.environ.get("LEFT_DATA_ROOT",
                                           os.environ.get("FACT_DATA_ROOT",
                                                          os.path.expanduser("~/datasets"))))
    ap.add_argument("--out-dir", required=True,
                    help="run tree: checkpoints/ and the results JSON land here")
    ap.add_argument("--out", default=None,
                    help="results JSON (default <out-dir>/cv_<dataset>_source.json); "
                         "an existing file is extended, and runs already in it are skipped")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = args.out or os.path.join(args.out_dir, f"cv_{args.dataset}_source.json")
    results = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["variant"], r["seed"]) for r in results}

    for variant in args.variants:
        for seed in args.seeds:
            if (variant, seed) in done:
                print(f"skip {variant} seed{seed} (already in {out})", flush=True)
                continue
            t0 = time.time()
            traj = []
            result, model = cv_train.train_one(
                args.dataset, variant, args.data_root, args.out_dir, seed=seed,
                epochs=args.epochs, num_workers=args.num_workers,
                log=lambda *a: print(*a, flush=True),
                epoch_hook=_coeff_hook(traj), return_model=True)
            results.append(result)
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            if traj:
                name = cv_train.run_name(args.dataset, variant, seed)
                with open(os.path.join(args.out_dir, f"{name}_coeff_traj.json"), "w") as f:
                    json.dump({"source": f"{name}, per-epoch shared coefficients",
                               "rows": "[a0, a_1..a_K, b_1..b_K] after each epoch",
                               "epochs": traj}, f, indent=2)
            print(f"=== saved {variant} seed{seed} in {time.time() - t0:.1f}s "
                  f"(test_acc={result['test_acc']:.4f}) ===\n", flush=True)
            del model

    print(f"results: {out}")
    print(f"checkpoints: {os.path.join(args.out_dir, 'checkpoints')}")


if __name__ == "__main__":
    main()
