#!/usr/bin/env python3
"""Bounded-pool driver for the MLP-Mixer transfer grid under the common
training configuration, across the full conventional-activation zoo.

Runs the common recipe (lr 1e-3, wd 0.05, 100 epochs, seeds 0-4) for one or
more (dataset, activation) cells, with a bounded pool of concurrent
`train_target.py` processes on one GPU. Resumable (skip-if-result-exists) and
safe to re-launch. Results land in the layout the published Mixer runs used, so
`left.evaluation.evaluate --aggregate` and the `_100ep` filenames behind
`results/tables/mixer_5seed_raw.json` both line up:

  <out-root>/<ds>/mlpmixer100k_<ds>_<act>_seed<N>_100ep_result.json

`train_target.py` writes the plain `..._seed<N>_result.json`; this driver
renames it (and the best/last checkpoints) to the `_100ep` form afterwards.

Usage (one dataset per GPU, as in the study's host-controlled design):
  python -m left.training.run_zoo --datasets fmnist \
      --acts silu elu tanh mish softplus hardswish selu \
      --seeds 0 1 2 3 4 --gpu 0 --workers 4 --data-root <dr> [--no-cuda-kernel]

Only `fact_fixed` uses the fused CUDA kernel, so --no-cuda-kernel is a harmless
no-op for the ten builtin activations but is accepted for parity with the other
drivers and for hosts without ninja.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
#: repository root (src/training -> src -> repo), so a run tree never lands
#: inside the installed package.
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(HERE, "train_target.py")
OUT_ROOT_DEFAULT = os.path.join(ROOT, "results", "logs", "mlpmixer_100ep")

NAME_PREFIX = "mlpmixer100k"
LR = 1e-3
WD = 0.05
EPOCHS = 100


def out_dir(out_root, ds):
    return os.path.join(out_root, ds)


def final_result(out_root, ds, act, seed):
    return os.path.join(out_dir(out_root, ds),
                        f"{NAME_PREFIX}_{ds}_{act}_seed{seed}_100ep_result.json")


def build_jobs(out_root, datasets, acts, seeds):
    jobs = []
    for ds in datasets:
        for act in acts:
            for seed in seeds:
                if os.path.exists(final_result(out_root, ds, act, seed)):
                    continue
                jobs.append((ds, act, seed))
    return jobs


def make_cmd(py, out_root, ds, act, seed, data_root, no_kernel, workers_dl):
    o = out_dir(out_root, ds)
    os.makedirs(o, exist_ok=True)
    cmd = [py, "-u", TRAIN, "--arch", "mlpmixer", "--dataset", ds,
           "--act", act, "--seed", str(seed), "--epochs", str(EPOCHS),
           "--lr", repr(LR), "--wd", repr(WD),
           "--num-workers", str(workers_dl), "--data-root", data_root,
           "--out-dir", o]
    if no_kernel:
        cmd.append("--no-cuda-kernel")
    return cmd


def promote(out_root, ds, act, seed):
    """Rename train_target's plain outputs to the study's _100ep filenames."""
    o = out_dir(out_root, ds)
    plain = f"{NAME_PREFIX}_{ds}_{act}_seed{seed}"
    src = os.path.join(o, f"{plain}_result.json")
    if os.path.exists(src):
        os.replace(src, final_result(out_root, ds, act, seed))
    ck = os.path.join(o, "checkpoints")
    for suf in ("best", "last"):
        s = os.path.join(ck, f"{plain}_{suf}.pt")
        if os.path.exists(s):
            os.replace(s, os.path.join(ck, f"{plain}_100ep_{suf}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--acts", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=4, help="concurrent train procs")
    ap.add_argument("--workers-dl", type=int, default=2, help="DataLoader workers per proc")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--no-cuda-kernel", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT,
                    help="run tree root (default: results/logs/mlpmixer_100ep)")
    args = ap.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    jobs = build_jobs(args.out_root, args.datasets, args.acts, args.seeds)
    total = len(jobs)
    print(f"[run_zoo] datasets={args.datasets} acts={args.acts} "
          f"seeds={args.seeds} gpu={args.gpu} workers={args.workers} "
          f"pending={total}", flush=True)
    if total == 0:
        print("[run_zoo] nothing to do.", flush=True)
        return

    logdir = os.path.join(args.out_root, "logs")
    os.makedirs(logdir, exist_ok=True)
    running = []  # (Popen, (ds,act,seed), logfile, t0)
    done = fail = 0
    i = 0
    t_start = time.time()

    def launch(job):
        ds, act, seed = job
        cmd = make_cmd(args.python, args.out_root, ds, act, seed, args.data_root,
                       args.no_cuda_kernel, args.workers_dl)
        lf = open(os.path.join(logdir, f"common_{ds}_{act}_seed{seed}_100ep.log"), "w")
        lf.write("CMD: " + " ".join(cmd) + "\n"); lf.flush()
        return (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env),
                job, lf, time.time())

    while i < total or running:
        while i < total and len(running) < args.workers:
            running.append(launch(jobs[i]))
            print(f"[run_zoo] >>> start {jobs[i]} ({i+1}/{total})", flush=True)
            i += 1
        time.sleep(3)
        still = []
        for proc, job, lf, t0 in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, job, lf, t0)); continue
            lf.close()
            if rc == 0:
                promote(args.out_root, *job)
                done += 1
                print(f"[run_zoo] done {job} rc=0 ({time.time()-t0:.0f}s) "
                      f"[{done+fail}/{total}]", flush=True)
            else:
                fail += 1
                print(f"[run_zoo] FAIL {job} rc={rc} -- see "
                      f"{logdir}", flush=True)
        running = still

    print(f"[run_zoo] ALL DONE done={done} fail={fail} "
          f"wall={time.time()-t_start:.0f}s", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
