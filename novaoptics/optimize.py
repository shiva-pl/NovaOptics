"""Damped least squares (Levenberg-Marquardt) with exact JAX Jacobians.

Which parameters are free is given by a dict {name: True | list of element
indices | boolean array}; everything else is frozen.  Jitted residual/Jacobian
functions are cached per (layout, spec, free-set), so a continuation path that
only changes `cfg` values compiles once.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .merit import residuals
from .system import Layout, Spec

# Compiled residual/Jacobian pairs, least recently used first.  Every layout (stop
# position, glass model, aspheres) compiles its own pair, so the cache is bounded:
# an unbounded one ran a long multi-architecture run out of memory in LLVM.
_CACHE: OrderedDict = OrderedDict()
CACHE_SIZE = 16


def clear_compiled():
    """Drop all compiled merit functions (ours and JAX's own caches)."""
    _CACHE.clear()
    jax.clear_caches()


def free_mask(params, free: dict):
    mask = {}
    for k, v in params.items():
        m = np.zeros(np.shape(v), dtype=bool)
        spec = free.get(k, False)
        if spec is True:
            m[...] = True
        elif spec is False or spec is None:
            pass
        else:
            arr = np.asarray(spec)
            if arr.dtype == bool:
                m[...] = arr
            else:
                m[arr] = True
        mask[k] = m
    flat, _ = ravel_pytree({k: jnp.asarray(v) for k, v in mask.items()})
    return np.nonzero(np.asarray(flat) > 0.5)[0]


def _compiled(layout: Layout, spec: Spec, unravel, idx_key, transform):
    key = (layout, spec, idx_key, transform)
    if key in _CACHE:
        _CACHE.move_to_end(key)
    else:
        idx = jnp.asarray(idx_key)

        def res(x, full, cfg):
            p = unravel(full.at[idx].set(x))
            if transform is not None:
                p = transform(p, cfg, layout)
            return residuals(p, layout, spec, cfg)

        _CACHE[key] = (jax.jit(res), jax.jit(jax.jacfwd(res)))
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def optimize(params, layout: Layout, spec: Spec, cfg, free: dict, iters=40, verbose=False,
             lam=1e-3, tol=1e-9, transform=None):
    """Minimize the merit over the `free` entries of `params`.

    If `transform` (a module-level function `(z, cfg, layout) -> params`) is given,
    `params` is a design-variable pytree z that is mapped to system parameters
    before evaluating the merit; this is how continuation reparameterizations work.
    """
    full, unravel = ravel_pytree(params)
    idx = free_mask(params, free)
    if idx.size == 0:
        return params, {}
    res, jac = _compiled(layout, spec, unravel, tuple(int(i) for i in idx), transform)
    x = full[idx]
    r = res(x, full, cfg)
    cost = float(r @ r)
    history = [cost]
    t0 = time.time()
    for it in range(iters):
        J = np.asarray(jac(x, full, cfg))
        rn = np.asarray(r)
        A = J.T @ J
        g = J.T @ rn
        dA = np.diag(A).copy() + 1e-12
        improved = False
        for _ in range(12):
            try:
                dx = -np.linalg.solve(A + lam * np.diag(dA), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            x_new = x + dx
            r_new = res(x_new, full, cfg)
            c_new = float(r_new @ r_new)
            if np.isfinite(c_new) and c_new < cost:
                improved = True
                rel = (cost - c_new) / max(cost, 1e-30)
                x, r, cost = x_new, r_new, c_new
                lam = max(lam / 3, 1e-9)
                break
            lam *= 4
        history.append(cost)
        if verbose:
            print(f"    it {it:3d}  cost {cost:.6e}  lam {lam:.1e}")
        if not improved or rel < tol:
            break
    out = unravel(full.at[idx].set(x))
    return out, {"cost": cost, "history": history, "iters": it + 1, "time": time.time() - t0}
