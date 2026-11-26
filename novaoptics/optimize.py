"""Damped least squares (Levenberg-Marquardt) with exact JAX Jacobians.

Which parameters are free is given by a dict {name: True | list of element
indices | boolean array}; everything else is frozen.  Jitted residual/Jacobian
functions are cached per (layout, spec, free-set), so a continuation path that
only changes `cfg` values compiles once.
"""

from __future__ import annotations

import threading
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
_LOCK = threading.Lock()  # optimizations may run in threads (see parallel.py)
# Relative cost reduction regarded as converged.  Kept tight: the morph is a homotopy,
# and stopping its steps early (1e-7 was tried) steers it into a worse final design
# (0.108 vs 0.102 waves), as the path-gap diagnostic predicts.
STOP_TOL = 1e-9


class Stats:
    """Wall-time accounting of the optimizer (seconds), for profiling and progress."""

    def __init__(self):
        self.reset()

    def add(self, t=None, n=None):
        with _LOCK:
            for k, v in (t or {}).items():
                self.t[k] += v
            for k, v in (n or {}).items():
                self.n[k] += v

    def reset(self):
        self.t = {"compile": 0.0, "jac": 0.0, "res": 0.0, "solve": 0.0, "total": 0.0}
        self.n = {"compiles": 0, "calls": 0, "iters": 0, "jac": 0, "res": 0, "rejected": 0}
        self.shapes = []  # (n_free, n_residuals) per compiled function pair

    def report(self):
        t, n = self.t, self.n
        other = t["total"] - t["compile"] - t["jac"] - t["res"] - t["solve"]
        lines = [f"optimizer wall time {t['total']:.1f} s in {n['calls']} calls, "
                 f"{n['iters']} LM iterations ({n['rejected']} rejected trial steps)"]
        for k in ("compile", "jac", "res", "solve"):
            per = {"compile": n["compiles"], "jac": n["jac"], "res": n["res"],
                   "solve": n["iters"] + n["rejected"]}[k]
            lines.append(f"  {k:<8} {t[k]:8.1f} s  {100 * t[k] / max(t['total'], 1e-9):5.1f}%"
                         f"  ({per} x {1e3 * t[k] / max(per, 1):.2f} ms)")
        lines.append(f"  {'other':<8} {other:8.1f} s  {100 * other / max(t['total'], 1e-9):5.1f}%")
        return "\n".join(lines)


STATS = Stats()


def clear_compiled():
    """Drop all compiled merit functions (ours and JAX's own caches)."""
    with _LOCK:
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
    with _LOCK:
        return _compiled_locked(key, layout, spec, unravel, idx_key, transform)


def _compiled_locked(key, layout, spec, unravel, idx_key, transform):
    if key in _CACHE:
        _CACHE.move_to_end(key)
    else:
        idx = jnp.asarray(idx_key)

        def res(x, full, cfg):
            p = unravel(full.at[idx].set(x))
            if transform is not None:
                p = transform(p, cfg, layout)
            return residuals(p, layout, spec, cfg)

        # [residual, Jacobian, compiled?, compile lock]
        _CACHE[key] = [jax.jit(res), jax.jit(jax.jacfwd(res)), False, threading.Lock()]
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def optimize(params, layout: Layout, spec: Spec, cfg, free: dict, iters=40, verbose=False,
             lam=1e-3, tol=None, transform=None):
    """Minimize the merit over the `free` entries of `params` (Levenberg-Marquardt).

    Marquardt damping: lambda / 3 after an accepted step, x 4 after a rejected one.  The
    loop stops when the relative cost reduction falls below `tol` (default STOP_TOL)
    or no step improves the cost.  (Nielsen's rho-based damping and a looser stopping
    rule were tried: they saved few iterations and made the final designs worse, since
    the morph's homotopy path is sensitive to how each step converges.)

    If `transform` (a module-level function `(z, cfg, layout) -> params`) is given,
    `params` is a design-variable pytree z that is mapped to system parameters
    before evaluating the merit; this is how continuation reparameterizations work.
    Thread-safe: independent calls may run concurrently (see parallel.py).
    """
    full, unravel = ravel_pytree(params)
    idx = free_mask(params, free)
    if idx.size == 0:
        return params, {}
    entry = _compiled(layout, spec, unravel, tuple(int(i) for i in idx), transform)
    res, jac = entry[0], entry[1]
    x = full[idx]
    clock = time.perf_counter
    t_start = clock()
    lt = {"compile": 0.0, "jac": 0.0, "res": 0.0, "solve": 0.0}
    ln = {"calls": 1, "compiles": 0, "iters": 0, "jac": 0, "res": 0, "rejected": 0}
    if not entry[2]:  # first use: compile both functions once (other threads wait)
        with entry[3]:
            if not entry[2]:
                tc = clock()
                jax.block_until_ready(res(x, full, cfg))
                J0 = jax.block_until_ready(jac(x, full, cfg))
                lt["compile"] += clock() - tc
                ln["compiles"] += 1
                with _LOCK:
                    STATS.shapes.append((int(x.size), int(J0.shape[0])))
                entry[2] = True
    tr = clock()
    r = res(x, full, cfg)
    cost = float(r @ r)
    lt["res"] += clock() - tr
    ln["res"] += 1
    history = [cost]
    t0 = time.time()
    it = 0
    tol = STOP_TOL if tol is None else tol
    for it in range(iters):
        tj = clock()
        J = np.asarray(jac(x, full, cfg))
        lt["jac"] += clock() - tj
        ln["jac"] += 1
        ln["iters"] += 1
        rn = np.asarray(r)
        ts = clock()
        A = J.T @ J
        g = J.T @ rn
        dA = np.diag(A).copy() + 1e-12
        lt["solve"] += clock() - ts
        improved = False
        for _ in range(12):
            ts = clock()
            try:
                dx = -np.linalg.solve(A + lam * np.diag(dA), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            finally:
                lt["solve"] += clock() - ts
            x_new = x + dx
            tr = clock()
            r_new = res(x_new, full, cfg)
            c_new = float(r_new @ r_new)
            lt["res"] += clock() - tr
            ln["res"] += 1
            if np.isfinite(c_new) and c_new < cost:
                improved = True
                rel = (cost - c_new) / max(cost, 1e-30)
                x, r, cost = x_new, r_new, c_new
                lam = max(lam / 3, 1e-9)
                break
            ln["rejected"] += 1
            lam *= 4
        history.append(cost)
        if verbose:
            print(f"    it {it:3d}  cost {cost:.6e}  lam {lam:.1e}")
        if not improved or rel < tol:
            break
    out = unravel(full.at[idx].set(x))
    lt["total"] = clock() - t_start
    STATS.add(lt, ln)
    return out, {"cost": cost, "history": history, "iters": it + 1, "time": time.time() - t0}
