"""Damped least squares (Levenberg-Marquardt) with exact JAX Jacobians.

Which parameters are free is given by a dict {name: True | list of element
indices | boolean array}; everything else is frozen.  Jitted residual/Jacobian
functions are cached per (layout, spec, free-set), so a continuation path that
only changes `cfg` values compiles once.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import progress
from .merit import residuals
from .system import Layout, Spec

# Compiled residual/Jacobian pairs, least recently used first.  Every layout (stop
# position, glass model, aspheres) compiles its own pair, so the cache is bounded:
# an unbounded one ran a long multi-architecture run out of memory in LLVM.
_CACHE: OrderedDict = OrderedDict()
CACHE_SIZE = 16
_LOCK = threading.Lock()  # optimizations may run in threads (see parallel.py)
# XLA compiles of the residual Jacobian grow quickly with the number of elements (a
# 6-element program needs several GB while compiling); threads still optimize in
# parallel, but at most this many compiles run at the same time.
_COMPILE_SLOTS = threading.BoundedSemaphore(int(os.environ.get("NOVAOPTICS_COMPILES", "2")))
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
    """Drop all compiled programs (merit, summaries, drawing, JAX's own caches) and return
    the memory to the operating system.  Useful between unrelated designs."""
    release_memory(everything=True)


def _mask_tree(params, free):
    """Boolean tree shaped like `params`.  For a dict of arrays, `free` maps names to
    True / False / an index array / a boolean array; nested dicts and lists (e.g. the
    per-surface dicts of an engine system) take the same nested structure in `free`."""
    if isinstance(params, dict):
        free = free or {}
        return {k: _mask_tree(v, free.get(k, False) if isinstance(free, dict) else free)
                for k, v in params.items()}
    if isinstance(params, (list, tuple)):
        if isinstance(free, (list, tuple)):
            return [_mask_tree(v, f) for v, f in zip(params, free)]
        return [_mask_tree(v, free) for v in params]
    m = np.zeros(np.shape(params), dtype=bool)
    if free is True:
        m[...] = True
    elif free is False or free is None or isinstance(free, dict):
        pass
    else:
        arr = np.asarray(free)
        if arr.dtype == bool:
            m[...] = arr
        else:
            m[arr] = True
    return jnp.asarray(m)


def free_mask(params, free):
    flat, _ = ravel_pytree(_mask_tree(params, free))
    return np.nonzero(np.asarray(flat) > 0.5)[0]


def _compiled(layout: Layout, spec: Spec, unravel, idx_key, transform, residual_fn=None):
    key = (layout, spec, idx_key, transform, residual_fn)
    with _LOCK:
        return _compiled_locked(key, layout, spec, unravel, idx_key, transform, residual_fn)


def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def _mem_limit():
    """Bytes this process may use: the cgroup limit if there is one, else physical RAM."""
    total = None
    try:
        with open("/proc/meminfo") as f:
            total = int(f.readline().split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    cg = [_read_int("/sys/fs/cgroup/memory.max"),
          _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    vals = [v for v in [total, *cg] if v and v < 1 << 60]
    return min(vals) if vals else None


_MAP_LIMIT = _read_int("/proc/sys/vm/max_map_count")
_MEM_LIMIT = _mem_limit()
# Fractions of vm.max_map_count and of the memory limit that compiled programs may push
# the process to before least recently used programs are evicted.
MAP_BUDGET = 0.55
MEM_BUDGET = 0.5
_CLEARABLE: list = []  # other caches of compiled programs (summary, drawing geometry)


def register_cache(clear):
    """Register a function that empties a cache of compiled programs."""
    _CLEARABLE.append(clear)


def _map_count():
    try:
        with open("/proc/self/maps") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _rss():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def memory_pressure():
    """True when compiled programs should be released before compiling more.

    Every compiled XLA CPU program holds memory and thousands of memory mappings (about
    2700 for a 3-element residual + Jacobian pair, twice that for 6 elements).  Linux caps
    mappings per process (vm.max_map_count, 65530 by default; exceeding it aborts inside
    LLVM), and a 6-element study otherwise grows past 14 GB."""
    if _MAP_LIMIT and _map_count() > MAP_BUDGET * _MAP_LIMIT:
        return True
    return bool(_MEM_LIMIT) and _rss() > MEM_BUDGET * _MEM_LIMIT


def _trim():
    """Return freed heap memory to the operating system (glibc)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _register_defaults():
    if _CLEARABLE:
        return
    from . import analysis, plot

    register_cache(analysis._SUMMARY_JIT.clear)
    register_cache(plot._GEOM_JIT.clear)


def release_memory(everything=False):
    """Drop compiled programs: least recently used optimizer programs while under memory
    pressure (all of them with everything=True), then, if still needed, the other
    registered caches and JAX's own caches."""
    import gc

    _register_defaults()
    with _LOCK:
        while _CACHE and (everything or (len(_CACHE) > 1 and memory_pressure())):
            _CACHE.popitem(last=False)
            gc.collect()
    if everything or memory_pressure():
        for clear in _CLEARABLE:
            clear()
        jax.clear_caches()
        gc.collect()
    _trim()


def _compiled_locked(key, layout, spec, unravel, idx_key, transform, residual_fn=None):
    if key in _CACHE:
        _CACHE.move_to_end(key)
    else:
        if memory_pressure():
            import gc

            while len(_CACHE) > 1 and memory_pressure():
                _CACHE.popitem(last=False)
                gc.collect()
            if memory_pressure():
                _register_defaults()
                for clear in _CLEARABLE:
                    clear()
                gc.collect()
            _trim()
        idx = jnp.asarray(idx_key)

        def res(x, full, cfg):
            p = unravel(full.at[idx].set(x))
            if transform is not None:
                p = transform(p, cfg, layout)
            return (residual_fn or residuals)(p, layout, spec, cfg)

        # [residual, Jacobian, compiled?, compile lock]
        _CACHE[key] = [jax.jit(res), jax.jit(jax.jacfwd(res)), False, threading.Lock()]
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def ensure_compiled(entry, x, full, cfg):
    """Compile an entry's residual and Jacobian on first use (other threads wait for it;
    at most NOVAOPTICS_COMPILES compiles at once).  Returns the compile time, or 0."""
    if entry[2]:
        return 0.0
    with entry[3]:
        if entry[2]:
            return 0.0
        with _COMPILE_SLOTS:
            tc = time.perf_counter()
            jax.block_until_ready(entry[0](x, full, cfg))
            J0 = jax.block_until_ready(entry[1](x, full, cfg))
            tc = time.perf_counter() - tc
        with _LOCK:
            STATS.shapes.append((int(x.size), int(J0.shape[0])))
        entry[2] = True
        return tc


def optimize(params, layout: Layout, spec: Spec, cfg, free: dict, iters=40, verbose=False,
             lam=1e-3, tol=None, transform=None, residual_fn=None):
    """Minimize the merit over the `free` entries of `params` (Levenberg-Marquardt).

    Marquardt damping: lambda / 3 after an accepted step, x 4 after a rejected one.  The
    loop stops when the relative cost reduction falls below `tol` (default STOP_TOL)
    or no step improves the cost.  (Nielsen's rho-based damping and a looser stopping
    rule were tried: they saved few iterations and made the final designs worse, since
    the morph's homotopy path is sensitive to how each step converges.)

    `residual_fn(params, layout, spec, cfg)` replaces the element-model merit (the
    surface engine passes its own, with `layout` its hashable problem description).
    If `transform` (a module-level function `(z, cfg, layout) -> params`) is given,
    `params` is a design-variable pytree z that is mapped to system parameters
    before evaluating the merit; this is how continuation reparameterizations work.
    Thread-safe: independent calls may run concurrently (see parallel.py).
    """
    progress.check()
    full, unravel = ravel_pytree(params)
    idx = free_mask(params, free)
    if idx.size == 0:
        return params, {}
    entry = _compiled(layout, spec, unravel, tuple(int(i) for i in idx), transform,
                      residual_fn)
    res, jac = entry[0], entry[1]
    x = full[idx]
    clock = time.perf_counter
    t_start = clock()
    lt = {"compile": 0.0, "jac": 0.0, "res": 0.0, "solve": 0.0}
    ln = {"calls": 1, "compiles": 0, "iters": 0, "jac": 0, "res": 0, "rejected": 0}
    tc = ensure_compiled(entry, x, full, cfg)
    if tc:
        lt["compile"] += tc
        ln["compiles"] += 1
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
        progress.check()  # cooperative cancel (GUI stop button)
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
