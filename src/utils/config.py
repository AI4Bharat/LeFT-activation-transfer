"""YAML configs as argparse defaults.

A config file in `configs/` is the *source of truth* for a recipe; the CLI flags
override it. That ordering matters for reproducibility: a run is identified by
"this config, plus these explicit overrides", both of which end up recorded in
the checkpoint's `config` block.

A config is a shallow, sectioned mapping. Only these sections are read, and
within them only keys that correspond to a flag of the script being run:

    activation:  name, harmonics, omega, shared, trainable, artifact
    model:       architecture, embed_dim, depth, num_heads, mlp_ratio,
                 window, patch_size, stages, dropout
    data:        dataset, data_root, batch_size, num_workers
    optim:       epochs, lr, wd, seed
    transfer:    freeze_activation, transfer_weights

Unknown sections are ignored with a warning; an unknown *key* inside a known
section is an error, because a silently-ignored hyperparameter is how an
"identical recipe" stops being identical.

`include:` (a path, relative to the including file) merges another config
first, so a target config can inherit a dataset config without repeating it.
"""
import argparse
import os
import warnings

_SECTIONS = ("activation", "model", "data", "optim", "transfer")

#: config key -> argparse dest. Keys absent here are rejected.
KEY_TO_DEST = {
    # activation
    "name": "act", "harmonics": "K", "omega": "omega", "shared": "shared",
    "trainable": "trainable", "artifact": "artifact",
    # model
    "architecture": "arch", "embed_dim": "embed_dim", "depth": "depth",
    "num_heads": "num_heads", "mlp_ratio": "mlp_ratio", "window": "window",
    "patch_size": "patch_size", "stages": "stages", "dropout": "dropout",
    # data
    "dataset": "dataset", "data_root": "data_root", "batch_size": "batch_size",
    "num_workers": "num_workers",
    # optim
    "epochs": "epochs", "lr": "lr", "wd": "wd", "seed": "seed",
    "warmup_epochs": "warmup_epochs", "label_smoothing": "label_smoothing",
    # transfer
    "freeze_activation": "freeze_activation", "transfer_weights": "transfer_weights",
}


def load_config(path, _seen=None):
    """Read a config file (following `include:`) into a flat dict of argparse
    dests."""
    import yaml  # pyyaml is a light dependency and only the config path needs it

    path = os.path.abspath(path)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular include at {path}")
    _seen.add(path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    flat = {}
    for inc in ([raw["include"]] if isinstance(raw.get("include"), str)
                else raw.get("include", [])):
        flat.update(load_config(os.path.join(os.path.dirname(path), inc), _seen))

    for section, body in raw.items():
        if section in ("include", "name", "description", "notes"):
            continue
        if section not in _SECTIONS:
            warnings.warn(f"{path}: ignoring unknown section {section!r}")
            continue
        if not isinstance(body, dict):
            raise ValueError(f"{path}: section {section!r} must be a mapping")
        for key, value in body.items():
            if key not in KEY_TO_DEST:
                raise ValueError(
                    f"{path}: unknown key {section}.{key!r}. Add it to "
                    f"left.utils.config.KEY_TO_DEST if it is a real flag -- a "
                    f"silently ignored hyperparameter breaks reproducibility.")
            flat[KEY_TO_DEST[key]] = value
    return flat


def apply_config(parser, argv=None):
    """Parse `--config FILE ...` out of `argv`, fold the files into the
    parser's defaults (later files win), and return the parsed namespace.
    Explicit CLI flags still override the config, because argparse defaults are
    what we are setting."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", default=[])
    known, rest = pre.parse_known_args(argv)

    #: Declarative keys: they record the protocol in the config file (and are
    #: read by tests and docs) but are not flags of any trainer, so they must
    #: not produce an "unused key" warning on every run.
    declarative = {"K", "omega", "shared", "trainable",
                   "freeze_activation", "transfer_weights"}
    merged = {}
    for path in known.config:
        merged.update(load_config(path))
    if merged:
        valid = {a.dest for a in parser._actions}
        unused = sorted(set(merged) - valid - declarative)
        if unused:
            warnings.warn(f"config keys not used by this script: {unused}")
        parser.set_defaults(**{k: v for k, v in merged.items() if k in valid})
        # A `required` flag whose value the config just supplied is no longer
        # missing: argparse's required check looks only at argv, so without
        # this a config-driven run (scripts/train_targets.sh, which passes the
        # dataset as configs/datasets/<ds>.yaml) dies on "the following
        # arguments are required: --dataset".
        for action in parser._actions:
            if action.required and action.dest in merged:
                action.required = False
    parser.add_argument("--config", action="append", default=[],
                        help="YAML config(s) in configs/; CLI flags override them")
    args = parser.parse_args(rest)
    args.config = known.config
    return args
