# Methodology

## The question

An activation function is ordinarily a fixed design choice, and a *learnable*
activation is ordinarily treated as part of the network in which it was
learned. This work examines whether the learned nonlinearity is separable from
that network: if one curve is learned inside a source network, does it carry
information that benefits a different, independently initialised network?

Two confounds would render a positive result uninformative, and the measurement
is designed to exclude both:

1. **Capacity.** A learnable activation introduces additional parameters. If
   the arm with the highest accuracy also has the larger parameter count, no
   conclusion follows. Freezing removes this confound: in every target run the
   activation contributes no trainable parameters, as GELU does.
2. **Weight transfer.** If any source weight reaches the target network, the
   result is ordinary transfer learning. The quantity transferred is a JSON
   file containing five scalars, and `verify_weight_independence` establishes
   that nothing further is shared.

## The three steps

### 1. Learn

A single activation module is instantiated once and installed at every
activation site of the source network, giving one curve for the entire model,
shared across neurons and across depth. For FAct at K = 2 this amounts to five
scalars:

```
phi(t) = a0 + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) + b2 sin(2wt)
```

The coefficients are initialised at the truncated Fourier fit of GELU, so that
at initialisation the network computes a close approximation of the GELU
baseline; they are then optimised jointly with the weights by the same
optimizer. The source network reported in the paper is a depth-6, approximately
3M-parameter ViT trained for 100 epochs on ImageNet-1K.

Global sharing is essential rather than incidental. It establishes the result as
a statement about a single nonlinearity rather than about a per-layer adaptation
scheme, and it reduces the transferred quantity to five scalars.

### 2. Freeze

After source training the coefficients are extracted and held fixed. Frozen
denotes all of the following: the coefficients receive no gradient, they are
absent from the optimizer's parameter groups, and they are consequently exempt
from weight decay. `left.transfer.freeze` provides the corresponding checks
(`assert_frozen`, `optimizer_sees_no_activation`, `ActivationWatch`), and
`tests/test_transfer.py` exercises each of them, including a control in which an
unfrozen curve is confirmed to change, so that the tests on the frozen
condition cannot succeed for an unrelated reason.

### 3. Transfer

The frozen curve is installed in an independently initialised target network as
a fixed activation and compared, under one common training configuration,
against ten conventional activations. No other quantity is shared with the
source network: neither its weights, nor its depth, nor its width, and in three
of the four target architectures not its attention mechanism either.

## Why a Fourier basis

A truncated Fourier series is a global basis: every coefficient affects the
entire real line. A curve expressed in such a basis cannot encode a feature of
the source network's pre-activation distribution that is local to a particular
layer, and it is this property that makes a learned shape plausible as a
transferable quantity rather than a fitted one.

The claim is evaluated rather than assumed. `src/activations/pade.py` repeats
the protocol with PAU, a rational parameterisation in a local basis. A shared
PAU learned in the same source network and frozen into the depth-2 ViT target
also exceeds GELU on all four datasets, and exceeds frozen FAct on that target
by 0.16 to 0.87 percentage points. The effect is therefore a property of
globally shared learned nonlinearities in general rather than of the Fourier
basis specifically, which is why the framework and the activation are kept
separate in this repository. This comparison covers one target architecture
only.

## Harmonic count and frequency

K = 2, corresponding to five scalars, is the setting reported in the paper and
the only one evaluated. `FAct(K=...)` accepts any harmonic count, but a result
obtained at another K constitutes new work rather than a reproduction.

The base frequency w is held at 1, the series being fit on [-pi, pi], and is
not learnable: it is registered as a buffer rather than a parameter. w is a
gauge direction, along which weight decay displaces the coefficients by orders
of magnitude without altering the represented function, so a learnable w yields
large coefficient drift of no functional consequence. No learnable-frequency arm
is reported and none is implemented.

## The two GELU-fit quadratures

The source-scale and target-scale published runs computed the GELU Fourier fit
with different numerical integration rules: the trapezoidal rule on an
endpoint-inclusive grid, and a left Riemann sum on an endpoint-excluded grid. At
K = 2 the two differ by approximately 8e-4 in a1 and a2. Both are retained, each
reproducing the initialisation of the runs at its own scale;
`left.activations.fourier_fit` documents the difference and `tests/test_fact.py`
fixes it numerically. No part of the transfer result depends on this choice,
since the transferred curve is the learned one rather than the initialisation.

## What is reported

Each reported number is a mean over five seeds with a sample standard
deviation. Each comparison between two activations is a paired t-test over the
shared seeds, with Holm correction across the four datasets. The test is paired
because a given seed presents both arms with the same data order, which makes
the per-seed difference a considerably tighter quantity than the difference of
means. `left.evaluation.metrics` implements these statistics.

## The learned curve

```
theta* = (a0, a1, a2, b1, b2)
       = (0.2182, 0.1193, -0.4237, 0.8315, -0.5918)
```

extracted from the seed-1 ImageNet-1K source checkpoint (100 epochs, top-1
65.12%). Seeds 1 to 5 of that configuration agree to approximately three decimal
places on all five coefficients, as recorded in the artifact; the learned curve
is therefore a reproducible outcome rather than an artefact of a single seed.
