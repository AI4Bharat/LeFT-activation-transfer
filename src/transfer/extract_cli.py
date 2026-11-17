"""Command-line front end for the Freeze step: pull the learned curve out of a
source checkpoint and write it as a coefficient artifact.

    python -m left.transfer.extract_cli \
        --checkpoint runs/source/vit_d6_imagenet1k_fact_k2_seed1_final.pt \
        --out artifacts/fact/imagenet1k_vitd6_seed1.json

The extractor reads *every* activation site in the checkpoint and refuses to
write anything unless they agree -- so the artifact it produces is evidence
that the source curve was globally shared, not just an average of per-layer
curves. `--print` shows the result without writing.
"""
import argparse
import json

from left.transfer.transfer_activation import extract_from_checkpoint, save_artifact


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", help="artifact JSON to write")
    ap.add_argument("--prefix", help="read one activation site only, e.g. shared_act")
    ap.add_argument("--K", type=int, help="assert this harmonic count")
    ap.add_argument("--note", help="free-text provenance note stored in the artifact")
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    prov = {"note": args.note} if args.note else None
    art = extract_from_checkpoint(args.checkpoint, prefix=args.prefix, K=args.K,
                                  provenance=prov)
    if args.show or not args.out:
        print(json.dumps(art, indent=2))
    if args.out:
        save_artifact(art, args.out)
        print(f"wrote {args.out} "
              f"({art['K']} harmonics, {2 * art['K'] + 1} scalars, "
              f"{art['n_activation_sites']} activation sites agreed)")


if __name__ == "__main__":
    main()
