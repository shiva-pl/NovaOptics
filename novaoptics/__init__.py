"""NovaOptics: phase-first lens design with a differentiable ray tracer (JAX).

Importing the package configures JAX for this workload: 64-bit floats (optical path
differences of a fraction of a wavelength over tens of millimetres need them) and the
CPU backend.  The optimization is many small, sequential, latency-bound steps; on a
GPU it ran ~1.5x slower than on the CPU.  Set JAX_PLATFORMS yourself to override.
"""

import os

import jax

if "JAX_PLATFORMS" not in os.environ:
    jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
