"""Checkpoint I/O. The convention every trainer here follows:

    {"state_dict": ..., "config": {...}, "test_acc": float, "epoch": int}

`config` is what makes a checkpoint self-describing -- architecture, dataset,
activation, seed and epoch budget -- so a result file can be interpreted years
later without the launch script that produced it.
"""
import os

import torch


def save_checkpoint(path, model, config, **extra):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {"state_dict": model.state_dict(), "config": dict(config)}
    blob.update(extra)
    torch.save(blob, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """Return `(state_dict, config, blob)`."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"{path} is not a LeFT checkpoint")
    sd = blob.get("state_dict", blob.get("model", blob))
    return sd, blob.get("config", {}), blob


def load_into(model, path, strict=True, map_location="cpu"):
    sd, config, _ = load_checkpoint(path, map_location)
    model.load_state_dict(sd, strict=strict)
    return config
