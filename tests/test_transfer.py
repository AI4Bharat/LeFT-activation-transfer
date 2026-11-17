"""The LeFT protocol: freeze, transfer, and the negative claims.

Every test here corresponds to something that could silently go wrong and
still produce a plausible number -- a "frozen" curve that drifted, a target
network that inherited a source weight, a seed that did not take. Those are the
failure modes worth testing in a research repository.
"""
import json

import pytest
import torch

from left.activations import FAct
from left.models.vit import build_vit
from left.transfer import (ActivationWatch, activation_parameter_count,
                           assert_frozen, extract, extract_from_checkpoint,
                           install, load_artifact,
                           optimizer_sees_no_activation, save_artifact,
                           verify_curve_matches, verify_weight_independence)
from left.utils.seed import set_seed


def _model(act, seed=0, depth=2):
    set_seed(seed)
    return build_vit("cifar10", act, depth=depth, use_cuda_kernel=False)


def _step(model, lr=0.1):
    """One real optimizer step on only the trainable parameters."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.2)
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return opt


# --------------------------------------------------------------------------- #
#  Freeze
# --------------------------------------------------------------------------- #
def test_frozen_coefficients_receive_no_update(artifact_path):
    """The central freeze claim: training the target network does not move the
    transferred curve."""
    act = FAct.from_artifact(artifact_path)
    model = _model(act)
    before = act.coefficients()
    _step(model)
    assert act.coefficients() == before


def test_frozen_coefficients_receive_no_gradient(artifact_path):
    act = FAct.from_artifact(artifact_path)
    model = _model(act)
    loss = model(torch.randn(4, 3, 32, 32)).sum()
    loss.backward()
    assert act.a0.grad is None and act.a.grad is None and act.b.grad is None


def test_activation_watch_catches_a_curve_that_moved():
    """The guard itself must work, or it guards nothing."""
    act = FAct(K=2)                        # trainable on purpose
    with pytest.raises(AssertionError, match="frozen activation moved"):
        with ActivationWatch(act):
            with torch.no_grad():
                act.a0.add_(1e-3)


def test_optimizer_built_before_freezing_is_rejected():
    """requires_grad=False is not enough on its own: an optimizer that already
    holds the tensors would still apply weight decay to them."""
    act = FAct(K=2)
    model = _model(act)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # built too early
    act.freeze()
    with pytest.raises(AssertionError, match="optimizer holds an activation"):
        optimizer_sees_no_activation(opt, model)
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    assert optimizer_sees_no_activation(opt2, model)


def test_an_unfrozen_arm_does_train_its_curve():
    """Control: the learnable (source) arm must actually learn, otherwise the
    frozen tests above would pass for the wrong reason."""
    act = FAct(K=2)
    model = _model(act)
    before = act.coefficients()
    _step(model, lr=0.5)
    assert act.coefficients() != before
    assert activation_parameter_count(model) == 5


def test_assert_frozen_is_the_gate_a_target_run_passes(artifact_path):
    frozen = _model(FAct.from_artifact(artifact_path))
    assert assert_frozen(frozen)
    with pytest.raises(AssertionError):
        assert_frozen(_model(FAct(K=2)))


# --------------------------------------------------------------------------- #
#  Transfer
# --------------------------------------------------------------------------- #
def test_only_the_activation_crosses_over(tmp_path, artifact_path):
    """Install a transferred curve in a target network and verify that no
    source weight came with it."""
    source = _model(FAct(K=2), seed=1, depth=6)
    ckpt = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict(),
                "config": {"dataset": "imagenet1k", "seed": 1}}, ckpt)

    art = extract(find_source_act(source), provenance={"source_checkpoint": "source.pt"})
    save_artifact(art, tmp_path / "curve.json")

    target = _model(FAct(K=2, trainable=False), seed=7, depth=6)
    install(find_source_act(target), tmp_path / "curve.json")

    assert verify_curve_matches(find_source_act(target), tmp_path / "curve.json")
    assert verify_weight_independence(target, ckpt)


def test_weight_leakage_is_detected(tmp_path, artifact_path):
    """The negative claim has teeth: if source weights *were* copied, the
    verifier must say so."""
    source = _model(FAct(K=2), seed=1)
    ckpt = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, ckpt)

    leaky = _model(FAct(K=2), seed=7)
    leaky.load_state_dict(source.state_dict())        # the mistake
    with pytest.raises(AssertionError, match="leaked"):
        verify_weight_independence(leaky, ckpt)


def test_target_weights_are_independently_initialised():
    """Two seeds must give different weights -- the target network is trained
    from scratch, not fine-tuned."""
    a = _model(FAct(K=2), seed=0)
    b = _model(FAct(K=2), seed=1)
    diffs = [not torch.allclose(pa, pb)
             for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters())
             if pa.numel() > 1 and "a0" not in na and na.rsplit(".", 1)[-1] not in ("a", "b")]
    assert any(diffs)


def test_same_seed_reproduces_the_same_initialisation():
    a = _model(FAct(K=2), seed=3)
    b = _model(FAct(K=2), seed=3)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), na


def test_extraction_refuses_an_unshared_checkpoint(tmp_path):
    """If the activation sites disagree there is no single curve, and the
    extractor must refuse rather than pick one."""
    model = _model(FAct(K=2), depth=4)
    sd = dict(model.state_dict())
    # forge a per-layer curve
    sd["blocks.1.ffn.act.a0"] = sd["blocks.1.ffn.act.a0"] + 0.5
    ckpt = tmp_path / "unshared.pt"
    torch.save({"state_dict": sd}, ckpt)
    with pytest.raises(ValueError, match="not globally shared"):
        extract_from_checkpoint(ckpt)


def test_artifact_is_small_and_human_readable(artifact_path):
    """The artifact is the unit of transfer because it is five numbers a reader
    can check by eye -- not a multi-gigabyte checkpoint."""
    art = load_artifact(artifact_path)
    assert art["K"] == 2
    assert len(art["a"]) == len(art["b"]) == 2
    assert "source_checkpoint" in art and "source_run" in art
    assert len(json.dumps(art)) < 8000


def find_source_act(model):
    from left.transfer import find_activations
    return find_activations(model)[0]
