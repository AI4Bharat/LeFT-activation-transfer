"""The fused CUDA kernel against the reference path.

`fact_fixed` evaluates the transferred curve through a fused CUDA kernel by
default and through pure PyTorch under `--no-cuda-kernel`. docs/reproduction.md
tells a reader the two are numerically identical and that only their execution
times differ, so that claim is gated here rather than asserted in prose.

These tests need a GPU and a working `ninja` (the kernel is JIT-compiled on
first use); they skip otherwise, which is why the suite still runs on CPU.
"""
import pytest
import torch

from left.activations.fact_fixed import FixedFAct
from left.activations.zoo import act_param_count

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the fused kernel needs CUDA")


@pytest.fixture(scope="module")
def pair(artifact_path):
    """The same curve, once through the kernel and once through PyTorch."""
    try:
        fused = FixedFAct(coeff_path=artifact_path, use_cuda_kernel=True).cuda()
    except Exception as exc:                    # no nvcc, no ninja, no driver
        pytest.skip(f"could not build the fused kernel: {exc}")
    ref = FixedFAct(coeff_path=artifact_path, use_cuda_kernel=False).cuda()
    return fused, ref


@pytest.mark.parametrize("scale", [1.0, 3.0, 12.0])
def test_fused_kernel_equals_the_reference_path(pair, scale):
    fused, ref = pair
    x = torch.randn(4096, 512, device="cuda") * scale
    with torch.no_grad():
        assert torch.allclose(fused(x), ref(x), atol=1e-5, rtol=0)


def test_fused_kernel_is_exact_on_the_series_definition(pair):
    fused, _ = pair
    x = torch.linspace(-20, 20, 100_001, device="cuda")
    a0 = fused.a0.detach().reshape(())
    a = fused.a.detach().reshape(-1)
    b = fused.b.detach().reshape(-1)
    w = float(fused.w) if not torch.is_tensor(fused.w) else float(fused.w.reshape(()))
    series = a0.clone().expand_as(x).clone()
    for k in range(1, a.numel() + 1):
        series = series + a[k - 1] * torch.cos(k * w * x) + b[k - 1] * torch.sin(k * w * x)
    with torch.no_grad():
        assert torch.allclose(fused(x), series, atol=1e-5, rtol=0)


def test_both_paths_are_parameter_free():
    assert act_param_count("fact_fixed", True) == 0
    assert act_param_count("fact_fixed", False) == 0
