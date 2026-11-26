"""NovaOptics: phase-first lens design with a differentiable ray tracer (JAX).

Importing the package configures JAX for this workload: 64-bit floats (optical path
differences of a fraction of a wavelength over tens of millimetres need them) and the
CPU backend.  The optimization is many small, sequential, latency-bound steps; on a
GPU it ran ~1.5x slower than on the CPU.  Set JAX_PLATFORMS yourself to override.
Compiled programs are cached on disk (see below).
"""

import os

import jax

if "JAX_PLATFORMS" not in os.environ:
    jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

# Persistent compilation cache: compiling the merit Jacobian takes seconds per layout,
# and the same programs recur across runs, so they are kept on disk.
# NOVAOPTICS_JAX_CACHE overrides the location; an empty value disables the cache.
_cache = os.environ.get("NOVAOPTICS_JAX_CACHE",
                        os.path.join(os.path.expanduser("~"), ".cache", "novaoptics", "jax"))
if _cache:
    jax.config.update("jax_compilation_cache_dir", _cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
