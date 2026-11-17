"""Global sharing: one curve for the whole network.

"Globally shared" is the premise that makes a learned activation transferable
at all -- if every block learned its own curve there would be no single
nonlinearity to move, and the five-scalar artifact would be meaningless. These
tests check the property on all four target architectures rather than trusting
the builders.
"""
import pytest
import torch

from left.activations import FAct
from left.activations.zoo import act_param_count
from left.models.linvit import build_linvit
from left.models.mlp_mixer import build_mlpmixer
from left.models.swin import build_swin
from left.models.vit import build_vit
from left.transfer import activation_parameter_count, find_activations

BUILDERS = {"vit": build_vit, "swin": build_swin, "linvit": build_linvit,
            "mlpmixer": build_mlpmixer}
CPU_KW = {"use_cuda_kernel": False}


def _build(arch, act, dataset="cifar10", depth=2):
    kw = dict(CPU_KW)
    if arch == "mlpmixer":
        kw["embed_dim"] = 76
    return BUILDERS[arch](dataset, act, depth=depth, **kw)


@pytest.mark.parametrize("arch", list(BUILDERS))
def test_exactly_one_activation_instance(arch):
    """Every activation site must be the *same object*, at any depth."""
    act = FAct(K=2)
    model = _build(arch, act, depth=2)
    found = find_activations(model)
    assert len(found) == 1
    assert found[0] is act


@pytest.mark.parametrize("depth", [1, 2, 4, 6])
def test_sharing_does_not_depend_on_depth(depth):
    act = FAct(K=2)
    model = _build("vit", act, depth=depth)
    assert len(find_activations(model)) == 1
    assert activation_parameter_count(model, trainable_only=False) == 5


@pytest.mark.parametrize("arch", list(BUILDERS))
def test_every_block_sees_the_same_parameter_tensor(arch):
    """Sharing by reference, not by value: a gradient step on the curve must
    move it for every block at once."""
    act = FAct(K=2)
    model = _build(arch, act)
    tensors = {id(m.a0) for m in model.modules() if isinstance(m, FAct)}
    assert len(tensors) == 1


def test_one_curve_means_five_parameters_not_five_per_block():
    """The headline capacity claim: five scalars for the whole network."""
    act = FAct(K=2)
    model = _build("vit", act, depth=6)
    total = sum(p.numel() for p in model.parameters())
    act_params = activation_parameter_count(model, trainable_only=False)
    assert act_params == 5
    assert act_params / total < 1e-4


def test_frozen_activation_has_zero_trainable_parameters():
    """Equal capacity with a conventional activation -- what makes the
    comparison in the paper's tables fair."""
    act = FAct(K=2).freeze()
    model = _build("vit", act)
    assert activation_parameter_count(model, trainable_only=True) == 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gelu_model = _build("vit", "gelu")
    gelu_trainable = sum(p.numel() for p in gelu_model.parameters()
                         if p.requires_grad)
    assert trainable == gelu_trainable


def test_published_zoo_is_parameter_free():
    """Every entry of the paper's comparison menu contributes zero activation
    parameters, so swapping the activation changes the function and nothing
    else."""
    from left.activations.zoo import ACT_KINDS
    for kind in ACT_KINDS:
        assert act_param_count(kind, False) == 0, kind
    assert act_param_count("fact_k2_global", False) == 5   # the learnable arm


@pytest.mark.parametrize("arch", list(BUILDERS))
def test_forward_pass_runs_with_a_transferred_curve(arch, artifact_path):
    act = FAct.from_artifact(artifact_path)
    model = _build(arch, act)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
