"""Transfer -- step 3 of Learn, Freeze, Transfer.

What crosses the gap between the source network and a target network is a
*coefficient artifact*: a small JSON file holding the learned scalars plus the
provenance needed to say where they came from. Nothing else crosses. No weight,
no buffer, no optimizer state, no architectural choice.

That is why the artifact is the unit of transfer rather than a checkpoint:

* it is five numbers, so it lives in git next to the code;
* a reader can reproduce every target-network result without ever obtaining or
  training the source network;
* and it makes the negative claim auditable -- you can see for yourself that
  the only thing transferred is the nonlinearity.

Typical use:

    # once, from the source checkpoint
    art = extract_from_checkpoint("vit_d6_imagenet1k_fact_k2_seed1_final.pt")
    save_artifact(art, "artifacts/fact/imagenet1k_vitd6_seed1.json")

    # thereafter, for every target run
    act = FAct.from_artifact("artifacts/fact/imagenet1k_vitd6_seed1.json")
    model = build_vit("cifar100", act)
    assert_frozen(model)
    verify_weight_independence(model, "…seed1_final.pt")
"""
import datetime
import json
import os

import torch

from left.activations.base import SharedActivation
from left.transfer.freeze import find_activations

#: state_dict key stems that hold FAct coefficients in a source checkpoint.
#: `_a`/`_b` are the source-scale parameterisation (fourier_layers.py exposes
#: `a`/`b` as properties over them); `a`/`b` the target-scale one.
_FACT_KEYS = {"a0": ("a0",), "a": ("_a", "a"), "b": ("_b", "b")}

#: Buffers whose presence means the source module applied a transform on top of
#: the raw parameters, so the raw values are NOT the function's coefficients.
_GATING_BUFFERS = ("kfact_inv", "route_logits", "amp", "a_dir", "phi")


def extract(activation, provenance=None):
    """Extract a coefficient artifact from a live `SharedActivation`."""
    if not isinstance(activation, SharedActivation):
        raise TypeError(f"expected a SharedActivation, got {type(activation)}")
    art = dict(activation.coefficients())
    art["extracted_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if provenance:
        art.update(provenance)
    return art


def extract_from_model(model, provenance=None):
    """Extract from the single shared activation of a trained model. Raises if
    the model has more than one distinct activation instance -- a model whose
    activations are not globally shared has no one curve to transfer."""
    acts = find_activations(model)
    if len(acts) != 1:
        raise ValueError(
            f"expected exactly one shared activation instance, found {len(acts)}. "
            "LeFT transfers a globally shared curve; per-layer activations have "
            "no single curve to extract.")
    return extract(acts[0], provenance)


def extract_from_checkpoint(path, prefix=None, K=None, w=1.0, provenance=None,
                            tol=1e-6):
    """Extract a FAct coefficient artifact straight from a source checkpoint,
    without building the source model.

    Every activation site in a correctly shared source run holds the *same*
    tensors, so this reads all of them and asserts they agree to `tol` -- that
    agreement is itself the evidence that the source curve was global rather
    than per-layer. Pass `prefix` (e.g. "shared_act") to read one site only.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob.get("model", blob))
    config = blob.get("config", {}) if isinstance(blob, dict) else {}

    for bad in _GATING_BUFFERS:
        hits = [k for k in sd if k.endswith("." + bad) or k == bad]
        if hits:
            raise ValueError(
                f"checkpoint has {hits[0]!r}: the source activation applies a "
                f"transform on top of its raw parameters (harmonic annealing, "
                f"factorial scaling, routing or learnable amplitude), so the "
                f"stored tensors are not the curve's coefficients. Rebuild the "
                f"source module and use extract_from_model() instead.")

    found = {}            # field -> {site: tensor}
    for field, stems in _FACT_KEYS.items():
        for key, val in sd.items():
            if prefix and not key.startswith(prefix):
                continue
            stem = key.rsplit(".", 1)[-1]
            if stem in stems:
                found.setdefault(field, {})[key] = val
    missing = [f for f in _FACT_KEYS if f not in found]
    if missing:
        raise ValueError(f"no FAct coefficients ({missing}) in {path}; "
                         f"is this a FAct checkpoint?")

    vals = {}
    for field, sites in found.items():
        ref_key, ref = next(iter(sites.items()))
        for key, val in sites.items():
            if val.shape != ref.shape or not torch.allclose(val.float(), ref.float(),
                                                            atol=tol):
                raise ValueError(
                    f"activation site {key!r} disagrees with {ref_key!r} on "
                    f"{field!r}: this checkpoint's activations are not globally "
                    f"shared, so there is no single curve to transfer.")
        vals[field] = ref.float().flatten().tolist()

    k = K if K is not None else len(vals["a"])
    if len(vals["a"]) != k or len(vals["b"]) != k:
        raise ValueError(f"expected K={k} coefficients, got a={len(vals['a'])}, "
                         f"b={len(vals['b'])}")
    art = {
        "family": "fact",
        "K": k,
        "w": float(config.get("act_w", w)),
        "a0": float(vals["a0"][0]),
        "a": vals["a"],
        "b": vals["b"],
        "source_checkpoint": os.path.basename(path),
        "n_activation_sites": len(found["a0"]),
    }
    if config:
        art["source_run"] = {kk: config[kk] for kk in
                             ("dataset", "variant", "seed", "epochs")
                             if kk in config}
    if "test_acc" in (blob if isinstance(blob, dict) else {}):
        art.setdefault("source_run", {})["test_top1"] = blob["test_acc"]
    art["extracted_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if provenance:
        art.update(provenance)
    return art


def save_artifact(artifact, path):
    """Write a coefficient artifact as pretty JSON (so git diffs are readable
    and a reader can check the five numbers by eye)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
    return path


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def install(activation, artifact, freeze=True, strict=True):
    """Install a coefficient artifact into a target activation and (by default)
    freeze it. This is the whole of 'transfer'."""
    if isinstance(artifact, (str, os.PathLike)):
        artifact = load_artifact(artifact)
    activation.load_coefficients(artifact, strict=strict)
    if freeze:
        activation.freeze()
    return activation


# --------------------------------------------------------------------------- #
#  Verification -- the negative claims
# --------------------------------------------------------------------------- #
def verify_weight_independence(target_model, source_checkpoint, tol=1e-8):
    """Raise if any non-activation tensor of `target_model` matches the source
    checkpoint's tensor of the same shape and name.

    Constant-valued tensors are skipped (see below): a LayerNorm gain of all
    ones is identical in every network and proves nothing.

    The claim LeFT makes is a negative one -- *no source-network weights were
    transferred* -- and a negative claim is worth a test. A target network that
    accidentally inherited a source weight would still train and still report a
    number.
    """
    blob = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob.get("model", blob))
    act_ids = {id(p) for a in find_activations(target_model)
               for p in list(a.parameters(recurse=False)) + list(a.buffers(recurse=False))}
    collisions = []
    tgt = dict(target_model.state_dict())
    for name, tensor in tgt.items():
        if any(id(tensor) == i for i in act_ids):
            continue
        if name.rsplit(".", 1)[-1] in ("a0", "a", "b", "_a", "_b", "kvec", "omega"):
            continue
        src = sd.get(name)
        if src is None or src.shape != tensor.shape or tensor.numel() < 2:
            continue
        # Constant-initialised tensors (LayerNorm gains of 1, biases of 0)
        # match between ANY two independently initialised networks, so an
        # equality there carries no information and must not be reported.
        # Only tensors with actual spread can evidence a leak.
        if float(tensor.float().std()) == 0.0:
            continue
        if torch.allclose(src.float(), tensor.float(), atol=tol):
            collisions.append(name)
    if collisions:
        raise AssertionError(
            f"{len(collisions)} target tensors are identical to the source "
            f"checkpoint's (e.g. {collisions[:3]}): source weights leaked into "
            f"the target network.")
    return True


def verify_curve_matches(activation, artifact, atol=1e-6):
    """Raise unless `activation` currently computes the artifact's curve. Run
    this after target training: together with ActivationWatch it closes the
    loop on 'the transferred curve is what was evaluated'."""
    if isinstance(artifact, (str, os.PathLike)):
        artifact = load_artifact(artifact)
    live = activation.coefficients()
    for key, want in artifact.items():
        if key not in live:
            continue
        got = live[key]
        if isinstance(want, list):
            if len(want) != len(got) or any(abs(x - y) > atol for x, y in zip(want, got)):
                raise AssertionError(f"{key}: artifact {want} != live {got}")
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            if abs(float(want) - float(got)) > atol:
                raise AssertionError(f"{key}: artifact {want} != live {got}")
    return True
