"""The LeFT protocol: freeze a learned shared activation, move it, verify.

Nothing in this package knows what a Fourier series is -- it moves coefficient
artifacts between objects satisfying `left.activations.base.SharedActivation`.
"""
from left.transfer.freeze import (ActivationWatch, activation_parameter_count,  # noqa: F401
                                  assert_frozen, find_activations,
                                  freeze_activation, optimizer_sees_no_activation)
from left.transfer.transfer_activation import (extract, extract_from_checkpoint,  # noqa: F401
                                               extract_from_model, install,
                                               load_artifact, save_artifact,
                                               verify_curve_matches,
                                               verify_weight_independence)
