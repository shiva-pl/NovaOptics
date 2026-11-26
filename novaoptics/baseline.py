"""Conventional design baseline, for honest comparison.

The "traditional" workflow: pick a known form (here a Cooke-type +/-/+ triplet with
SK16 / F2 / SK16), perturb thin-lens starting points, and run the *same* damped
least squares on the *same* merit function with the same constraints.  Optionally
the glasses are freed and snapped to the catalog with the same catalog stage as
the phase-first pipeline, and aspheres can be enabled, so that the only difference
between the two approaches is where the optimization starts from.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from . import analysis as A
from . import glass as G
from .continuation import Mech, Recorder, _glass_vars, stage_catalog
from .optimize import optimize
from .system import Layout, Spec, nearest_in_space, new_params, set_glass


def cooke_start(lay: Layout, spec: Spec, rng, spread=0.3):
    """Thin-lens Cooke-like start (powers ~ 2/f, -3.5/f, 2/f), randomly perturbed."""
    p = new_params(lay, gaps=[5.0, 1.0, 5.0], bfd=0.85 * spec.efl)
    phis = np.array([2.0, -3.5, 2.0]) / spec.efl * (1 + spread * rng.standard_normal(3))
    glasses = [G.CATALOG["N-SK16"], G.CATALOG["F2"], G.CATALOG["N-SK16"]]
    nd = np.array([g[0] for g in glasses])
    vd = np.array([g[1] for g in glasses])
    X = rng.uniform(-1.5, 1.5, 3)
    dc = phis / (nd - 1)
    p.update(c1=jnp.asarray(0.5 * (1 + X) * dc), c2=jnp.asarray(0.5 * (X - 1) * dc),
             t=jnp.full(3, 4.0), nd=jnp.asarray(nd), vd=jnp.asarray(vd))
    if lay.glass is not None:  # exact catalog glasses
        gs = lay.glass
        ids = [gs.find(n) if n in gs.names else nearest_in_space(gs, a, b)
               for n, (a, b) in zip(("N-SK16", "F2", "N-SK16"), zip(nd, vd))]
        set_glass(p, lay, [0, 1, 2], ids)
    return p


def conventional(spec: Spec, n_starts=12, aspheres=False, free_glass=False, mech=Mech(),
                 iters=200, seed=0, verbose=True, glass=None, stop_shift=False):
    """Best of `n_starts` Cooke-type starts; then optionally a stop shift on the glass
    system (as for the phase-first pipeline) and the catalog stage."""
    # Same glass handling as the phase-first pipeline: free glasses are continuous
    # (n_d, V_d) bounded by the catalog's glass-map hull and snapped at the end; fixed
    # glasses are the exact catalog SK16 / F2 / SK16.
    if glass is not None and free_glass:
        lay = Layout(n_elem=3, stop_index=1, aspheres=aspheres, region=glass)
    else:
        lay = Layout(n_elem=3, stop_index=1, aspheres=aspheres, glass=glass)
    rng = np.random.default_rng(seed)
    cfg = mech.cfg()
    free = {k: True for k in _glass_vars(lay, glass=free_glass)}
    runs = []
    t0 = time.time()
    for trial in range(n_starts):
        p = cooke_start(lay, spec, rng)
        p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
        s = A.summary(p, lay, spec)
        score = s["mean_wfe"] if abs(s["efl"] - spec.efl) < 0.05 * spec.efl else np.inf
        runs.append({"score": score if np.isfinite(score) else np.inf, "params": p,
                     "rms_wfe": s["rms_wfe"]})
        if verbose:
            print(f"  start {trial:2d}: mean RMS WFE {score:10.4f} waves", flush=True)
    best = min(runs, key=lambda r: r["score"])
    p = best["params"]
    rec = Recorder(spec, verbose=verbose)
    if stop_shift and np.isfinite(best["score"]):
        from .continuation import stage_stop_shift

        p, lay = stage_stop_shift(p, lay, spec, rec, mech=mech)
    if free_glass:
        p, lay = stage_catalog(p, lay, spec, rec, mech=mech)
    else:
        rec.add("conventional", 0, p, lay, 1.0, "best start")
    ok = sum(np.isfinite(r["score"]) and r["score"] < 2.0 for r in runs)
    return p, lay, {"runs": runs, "best": rec.snaps[-1].mean_wfe, "n_good": ok,
                    "n_starts": n_starts, "time": time.time() - t0, "rec": rec}
