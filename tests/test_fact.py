"""FAct itself: initialisation, harmonic configuration, and agreement with the
published implementation.

These are protocol tests, not smoke tests. Each one corresponds to a claim the
manuscript makes about the activation.
"""
import json
import math

import numpy as np
import pytest
import torch

from left.activations import FAct, gelu_init_coefficients
from left.activations.fact_fixed import FixedFAct
from left.activations.fact_k2_global_trainable import FactK2GlobalTrainable
from left.activations.fourier_fit import (fourier_coeffs, fourier_series_eval,
                                          gelu_np)

#: The GELU Fourier fit at K=2, trapezoid quadrature -- what the source run
#: started from, as recorded in the artifact's `gelu_fit_init` block.
GELU_FIT_K2 = {"a0": 0.7060692947200876,
               "a": [-0.7049028873443604, 0.02611173316836357],
               "b": [0.9999998211860657, -0.49999961256980896]}


def test_gelu_initialisation_matches_the_reported_fit():
    """Claim: FAct is initialised at the Fourier approximation of GELU, so the
    network starts out computing (very nearly) the GELU baseline."""
    got = gelu_init_coefficients(K=2)
    assert got["a0"] == pytest.approx(GELU_FIT_K2["a0"], abs=1e-9)
    assert got["a"] == pytest.approx(GELU_FIT_K2["a"], abs=1e-9)
    assert got["b"] == pytest.approx(GELU_FIT_K2["b"], abs=1e-9)
    assert got["w"] == 1.0


def test_gelu_init_approximates_gelu_and_improves_with_K():
    """The initialisation is only meaningful if the truncated series really
    approximates GELU where the network's pre-activations live.

    It is a *truncated* series on [-pi, pi], so the error is not uniformly
    small: at K=2 the mean absolute error over [-2, 2] is ~0.16 and the
    endpoint error is much larger (~1.3 at |t| = 3). That is expected and is
    why the measured pre-activation band matters (docs/methodology.md). What
    must hold is that the fit improves monotonically with K."""
    t = torch.linspace(-2, 2, 801)
    gelu = torch.nn.functional.gelu(t)
    errs = []
    for K in (2, 5, 9, 21):
        errs.append((FAct(K=K)(t) - gelu).abs().mean().item())
    assert errs[0] < 0.2, errs
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 0.03, errs


def test_the_two_quadratures_differ_only_at_the_documented_magnitude():
    """Source- and target-scale runs fit GELU with different quadratures. The
    difference is real and documented (~8e-4); this pins it so neither path can
    drift without the test noticing."""
    trap = gelu_init_coefficients(K=2, quadrature="trapezoid")
    riem = gelu_init_coefficients(K=2, quadrature="riemann")
    diffs = [abs(x - y) for x, y in zip(trap["a"], riem["a"])]
    assert max(diffs) < 2e-3
    assert max(diffs) > 1e-5
    # the target-scale value the published runs used
    assert riem["a"][1] == pytest.approx(0.0253, abs=5e-4)


@pytest.mark.parametrize("K", [1, 2, 3, 5, 9, 21])
def test_harmonic_count_is_configurable(K):
    """K is a knob, not a hard-coded 2. The paper reports K=2; the other
    values are supported, not evaluated."""
    act = FAct(K=K)
    assert act.num_activation_parameters() == 2 * K + 1
    t = torch.randn(7, 11)
    assert act(t).shape == t.shape
    assert torch.isfinite(act(t)).all()


def test_forward_equals_the_series_definition():
    """phi(t) = a0 + sum_k a_k cos(k w t) + b_k sin(k w t), evaluated
    independently in numpy."""
    act = FAct(K=3)
    c = act.coefficients()
    t = np.linspace(-8, 8, 257)
    want = fourier_series_eval(t, c["a0"], c["a"], c["b"], c["w"])
    got = act(torch.from_numpy(t).float()).detach().numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_agrees_with_the_published_frozen_implementation(artifact_path):
    """The framework-facing FAct must compute exactly what the paper's
    `FixedFAct` computes, or new work built on FAct would not sit on the
    published numbers."""
    act = FAct.from_artifact(artifact_path)
    ref = FixedFAct(coeff_path=artifact_path, use_cuda_kernel=False)
    t = torch.linspace(-10, 10, 2001)
    assert torch.allclose(act(t), ref(t), atol=1e-6)


def test_agrees_with_the_published_learnable_implementation():
    """Same, for the source arm -- up to the documented quadrature choice."""
    act = FAct(K=2, quadrature="riemann")
    ref = FactK2GlobalTrainable()
    t = torch.linspace(-10, 10, 2001)
    assert torch.allclose(act(t), ref(t), atol=1e-6)


def test_artifact_round_trip(tmp_path):
    """Coefficients survive extraction and reinstallation exactly -- the
    property the whole transfer step rests on."""
    src = FAct(K=4)
    with torch.no_grad():                     # pretend it learned something
        src.a.add_(0.3)
        src.b.mul_(-0.5)
        src.a0.add_(0.1)
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(src.coefficients()))

    dst = FAct.from_artifact(path)
    t = torch.linspace(-6, 6, 501)
    assert torch.allclose(src(t), dst(t), atol=1e-7)
    assert dst.frozen


def test_strict_load_refuses_a_mismatched_artifact():
    a2, a5 = FAct(K=2), FAct(K=5)
    with pytest.raises(ValueError):
        a2.load_coefficients(a5.coefficients(), strict=True)
    with pytest.raises(ValueError):
        a2.load_coefficients({"family": "spline", "K": 2, "a0": 0.0,
                              "a": [0.0, 0.0], "b": [0.0, 0.0]}, strict=True)


def test_shipped_artifact_is_the_manuscripts_curve(artifact_path):
    """The five numbers the paper reports, in the file the repository ships."""
    art = json.loads(open(artifact_path).read())
    assert art["K"] == 2 and art["w"] == 1.0
    reported = [0.2182, 0.1193, -0.4237, 0.8315, -0.5918]
    got = [art["a0"], art["a"][0], art["a"][1], art["b"][0], art["b"][1]]
    assert got == pytest.approx(reported, abs=5e-5)
