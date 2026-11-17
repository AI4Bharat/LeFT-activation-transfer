"""LeFT -- Learn, Freeze, Transfer.

A framework for learning a globally shared activation function inside a source
network, freezing it, and installing it in independently initialised target
networks without transferring any source-network weight.

The framework is deliberately independent of any particular activation family:
`left.transfer` moves a *coefficient artifact* between networks and knows
nothing about Fourier series. `left.activations.fact` (truncated Fourier) and
`left.activations.pade` (Pade / PAU) are two implementations that satisfy the
`SharedActivation` protocol it expects.
"""

__version__ = "1.0.0"
