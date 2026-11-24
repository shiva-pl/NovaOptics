"""The design pipeline: ideal phase -> distributed phase -> color -> glass -> catalog.

Stage 0  IDEAL        one achromatic phase screen at the stop: the best a single
                      ideal thin element can do.
Stage 1  DISTRIBUTE   split the power over N screens at real spacings and optimize
                      them, still achromatic, with *virtual Petzval*: each screen will
                      become glass of index n_d and then curve the image by -phi/n_d,
                      so the phase system is asked to image onto a surface pre-curved
                      by +sum(phi/n_d).  Pure phase screens have no Petzval curvature
                      of their own; without this the phase world happily converges to
                      structures that glass cannot realize.
Stage 2  COLOR        homotopy beta: 0 -> 1 in phase space.  Each screen disperses like
                      a thin lens of its assigned glass (n_d, V_d), which are now free.
                      Color correction and glass selection happen here, while the
                      elements are still ideal and cheap to reshape.
Stage 3  MATERIALIZE  homotopy s: 0 -> 1.  Element i carries total power phi_i; glass
                      takes s*phi_i (real curvatures, thickness, Petzval from the trace),
                      ideal phase keeps (1-s)*phi_i plus (1-s)*aspheric terms.  At s = 1
                      the phase is identically zero: an all-spherical glass system.
Stage 4  CATALOG      pull (n_d, V_d) to real catalog glasses, snap (trying the nearest
                      candidates per element) and re-optimize.

Every step is recorded so the morph can be inspected and plotted afterwards.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from . import analysis as A
from . import glass as G
from .merit import make_cfg, virtual_petzval_curvature
from .optimize import optimize
from .system import Layout, Spec, glass_label, nearest_in_space, new_params, set_glass


@dataclass
class Snapshot:
    stage: str
    step: int
    params: dict
    layout: Layout
    beta: float
    mean_wfe: float
    rms_wfe: np.ndarray
    efl: float
    note: str = ""
    extra: dict = field(default_factory=dict)  # e.g. morph design variables and cfg


@dataclass
class Recorder:
    spec: Spec
    snaps: list = field(default_factory=list)
    verbose: bool = True

    def add(self, stage, step, params, layout, beta, note="", virtual_petzval=0.0, extra=None):
        # Phase stages are judged on the image surface they are designed for: the
        # flat sensor, pre-curved by the Petzval their phase will acquire as glass.
        c_img = virtual_petzval * float(virtual_petzval_curvature(params, layout))
        s = A.summary(params, layout, self.spec, beta, c_img=c_img)
        snap = Snapshot(stage, step, params, layout, float(beta), s["mean_wfe"], s["rms_wfe"],
                        s["efl"], note, dict(extra or {}, virtual_petzval=virtual_petzval))
        self.snaps.append(snap)
        if self.verbose:
            wf = " ".join(f"{w:6.3f}" for w in s["rms_wfe"])
            print(f"[{stage:<11}] {step:2d}  EFL={s['efl']:7.3f}  RMS WFE [waves]: {wf}   {note}")
        return snap


@dataclass(frozen=True)
class Mech:
    """Manufacturing limits applied at the end of the morph [mm], and the
    aberration-content weight used in every stage after IDEAL."""

    tmin: float = 2.0
    emin: float = 1.0
    gmin: float = 1.0
    tmax: float = 8.0
    # Weight of the per-element aberration-content penalty (see merit.py).  Large
    # contributions that cancel between elements make tight tolerances, so this is
    # a manufacturability setting as much as an optical one.
    w_content: float = 0.0


def _powers(phase, r_norm):
    return " ".join(f"{float(v) * 1e3:+6.1f}" for v in -2.0 * phase[:, 0] / r_norm**2)


def _gvars(lay: Layout):
    """Names of the glass design variables: PCA coordinates or (n_d, V_d)."""
    return ["g"] if lay.glass is not None else ["nd", "vd"]


def _glass_vars(lay: Layout, glass=True):
    names = ["c1", "c2", "t", *(_gvars(lay) if glass else []), "gaps", "bfd"]
    return names + ["asph"] if lay.aspheres else names


def _glasses(p, lay: Layout | None = None):
    if lay is not None and lay.glass is not None:
        q = {k: np.asarray(v) for k, v in G.resolve(p, lay).items()}
        return " ".join(glass_label(q, lay, i) for i in range(lay.n_elem))
    return " ".join(f"{float(a):.3f}/{float(b):.1f}" for a, b in zip(p["nd"], p["vd"]))


# --------------------------------------------------------------------------- stages

def stage_ideal(spec: Spec, rec: Recorder, n_phase=4, r_norm=10.0, iters=60):
    lay = Layout(n_elem=1, stop_index=0, n_phase=n_phase, r_norm=r_norm)
    p = new_params(lay, gaps=[0.0], bfd=spec.efl)
    p["phase"] = p["phase"].at[0, 0].set(-(r_norm**2) / (2 * spec.efl))
    cfg = make_cfg(beta=0.0)
    p, _ = optimize(p, lay, spec, cfg, {"phase": True, "bfd": True}, iters=iters)
    rec.add("ideal", 0, p, lay, 0.0, "one ideal phase screen at the stop")
    return p, lay


def stage_distribute(p0, lay0: Layout, spec: Spec, rec: Recorder, n_elem=3, stop_index=1,
                     spacing=6.0, iters=100, mech=Mech(), virtual_petzval=1.0, pattern=None,
                     aspheres=False, glass=None):
    """Split the stage-0 power over `n_elem` screens.

    `pattern` gives relative powers (summing to 1), e.g. (2, -3, 2) for +/-/+;
    None means an equal split.  Screens of positive power start as a crown glass,
    negative ones as a flint (the usual achromat starting point); n_d and V_d are
    free from here on.
    """
    # Continuous glasses stay in (n_d, V_d), which optimizes well; with a catalog its
    # glass-map hull bounds them, and real glasses arrive by snapping after COLOR.
    lay = Layout(n_elem=n_elem, stop_index=stop_index, n_phase=lay0.n_phase, r_norm=lay0.r_norm,
                 aspheres=aspheres, region=glass)
    gaps = np.full(n_elem, spacing)
    if 0 < stop_index < n_elem:
        gaps[stop_index] = 0.5  # stop just in front of element `stop_index`
    pattern = np.ones(n_elem) / n_elem if pattern is None else np.asarray(pattern, float)
    pattern = pattern / pattern.sum()
    p = new_params(lay, gaps=gaps, bfd=float(p0["bfd"]))
    p["nd"] = jnp.asarray(np.where(pattern > 0, 1.62, 1.62))
    p["vd"] = jnp.asarray(np.where(pattern > 0, 60.0, 36.0))
    # Split by optical power only: away from the stop the beam footprint exceeds the
    # region where the stage-0 polynomial was fitted, so its high orders would
    # extrapolate wildly.  The optimizer re-grows the aspheric orders it needs.
    p["phase"] = jnp.zeros((n_elem, lay.n_phase)).at[:, 0].set(p0["phase"][0, 0] * pattern)
    rec.add("distribute", 0, p, lay, 0.0, f"split into {n_elem} screens  "
            f"powers[1/m] {_powers(p['phase'], lay.r_norm)}", virtual_petzval)
    cfg = make_cfg(beta=0.0, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval,
                   w_content=mech.w_content)
    free = {"phase": True, "gaps": True, "bfd": True, "nd": True}
    p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
    rec.add("distribute", 1, p, lay, 0.0,
            f"optimized w/ virtual Petzval  powers[1/m] {_powers(p['phase'], lay.r_norm)}",
            virtual_petzval)
    return p, lay


def stage_color(p, lay: Layout, spec: Spec, rec: Recorder, steps=5, iters=40, mech=Mech(),
                virtual_petzval=1.0, w_axial=0.0):
    free = {"phase": True, "gaps": True, "bfd": True, **{v: True for v in _gvars(lay)}}
    for k, beta in enumerate(np.linspace(0, 1, steps + 1)[1:], start=1):
        cfg = make_cfg(beta=beta, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval,
                       w_content=mech.w_content, w_axial=w_axial)
        p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
        rec.add("color", k, p, lay, beta, f"beta={beta:.2f}  glasses {_glasses(p, lay)}",
                virtual_petzval)
    return p, lay


X_MAX = 4.0


def stage_stop_shift(p, lay: Layout, spec: Spec, rec: Recorder, mech=Mech(),
                     virtual_petzval=1.0, k=3, iters=60, w_axial=0.0):
    """STOP-SHIFT: place the aperture stop once axial color is corrected.

    Stop-shift theory (see stopshift.py) predicts the third-order coma, astigmatism and
    field curvature for every stop position in closed form; with axial color corrected
    the lateral color does not depend on the stop.  The `k` best predicted positions
    and the current one are tried for real: the stop is moved there and the phase
    system re-optimized with the full real-ray merit.  The best real result is kept, so
    the stage can only improve the design.
    """
    from . import stopshift as SS
    from .merit import residuals

    sc = SS.scan(p, lay, spec, beta=1.0, virtual_petzval=virtual_petzval)
    z_now = sc["current_z"]
    i_now = int(np.argmin(np.abs(sc["z"] - z_now)))
    cands = SS.candidates(sc, k=k)
    cfg = make_cfg(beta=1.0, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval,
                   w_content=mech.w_content, w_axial=w_axial)
    free = {"phase": True, "gaps": True, "bfd": True, **{v: True for v in _gvars(lay)}}
    best = None
    for z in [z_now] + [c for c in cands if abs(c - z_now) > 0.5]:
        q, lq = SS.relayout(p, lay, z)
        q, _ = optimize(q, lq, spec, cfg, free, iters=iters)
        cost = float(np.sum(np.asarray(residuals(q, lq, spec, cfg)) ** 2))
        cost = cost if np.isfinite(cost) else np.inf
        i = int(np.argmin(np.abs(sc["z"] - z)))
        if rec.verbose:
            print(f"      stop at z={z:+7.2f} mm (third-order merit {sc['merit'][i]:9.3g}) "
                  f"-> real cost {cost:.4g}")
        if best is None or cost < best[0]:
            best = (cost, q, lq, z, i)
    _, q, lq, z, i = best
    rec.add("stop-shift", 0, q, lq, 1.0,
            f"stop {z_now:+.1f} -> {z:+.1f} mm (stop_index {lq.stop_index}); third-order "
            f"merit {sc['merit'][i_now]:.3g} -> {sc['merit'][i]:.3g}; "
            f"C_I={sc['CI'][i_now]:.2e} (lateral color stop-independent if ~0)",
            virtual_petzval, extra={"scan": sc})
    return q, lq


def morph_to_params(z, cfg, layout: Layout):
    """Homotopy between an ideal phase screen (s = 0) and a glass singlet (s = 1).

    Design variables per element: total power phi, bending xi, thickness t,
    glass (n_d, V_d) and aspheric phase b_k.  The glass carries s*phi (thin-lens
    curvatures from phi, X and n_d), the ideal phase carries (1 - s)*phi plus
    (1 - s)*b_k.  With aspheres enabled the glass also takes s*b_k as rear-surface
    aspheric sag.  Every s is a smooth deformation of the previous system, and at
    s = 1 the phase is identically zero.
    """
    s = cfg["s"]
    nd = G.nd_vec(z, layout)
    dc = s * z["phi"] / (nd - 1.0)
    # Shape factor X = (c1 + c2)/(c1 - c2), smoothly bounded.  While s is small the
    # bending barely matters, so an unbounded X can drift to absurd values that turn
    # into extreme menisci once the glass carries the power.
    X = X_MAX * jnp.tanh(z["xi"])
    c1 = 0.5 * (1.0 + X) * dc
    c2 = 0.5 * (X - 1.0) * dc
    if "dc1" in z:
        # optional curvature corrections beyond the thin-lens rule (thick-lens effects)
        c1 = c1 + s * z["dc1"]
        c2 = c2 + s * z["dc2"]
    a1 = -(1.0 - s) * z["phi"] * layout.r_norm**2 / 2.0
    phase = jnp.concatenate([a1[:, None], (1.0 - s) * z["bhi"]], axis=1)
    if layout.aspheres:
        # Aspheric phase is handed to the rear surface: a thin element adds
        # W = (n - 1) * sag, so b_k of phase corresponds to b_k/(n_d - 1) of sag.
        asph = s * z["bhi"] / (nd[:, None] - 1.0)
    else:
        asph = jnp.zeros_like(z["bhi"])
    out = {"c1": c1, "c2": c2, "t": z["t"], "nd": nd, "vd": G.vd_vec(z, layout),
           "phase": phase, "asph": asph, "gaps": z["gaps"], "bfd": z["bfd"]}
    if layout.glass is not None:
        out["g"], out["gr"] = z["g"], z["gr"]
    return out


def stage_materialize(p, lay: Layout, spec: Spec, rec: Recorder, steps=20, iters=30,
                      mech=Mech(), virtual_petzval=1.0, mu_b=1e-5, polish_iters=80,
                      free_curvature=True, glass_pull=0.0):
    """Morph phase into glass (see `morph_to_params`).

    glass_pull > 0 (continuous glasses only): the pull toward real catalog glasses
    ramps up with the morph, nu = glass_pull * s^2, so glass discreteness is turned on
    gradually while the lens takes shape and the final snap is a small step.
    """
    n = lay.n_elem
    z = {
        "phi": -2.0 * p["phase"][:, 0] / lay.r_norm**2,
        "xi": jnp.zeros(n),
        "bhi": p["phase"][:, 1:],
        "t": jnp.zeros(n),
        "nd": p["nd"],
        "vd": p["vd"],
        "gaps": p["gaps"],
        "bfd": p["bfd"],
    }
    if lay.glass is not None:
        z["g"], z["gr"] = p["g"], p["gr"]
    if free_curvature:
        # Curvature corrections on top of the thin-lens rule (scaled by s): the path-gap
        # diagnostic showed the thin-lens manifold alone costs 3-7x in merit mid-morph.
        z["dc1"] = jnp.zeros(n)
        z["dc2"] = jnp.zeros(n)
    # With a catalog the glasses were snapped in phase space and stay fixed (real
    # glasses throughout the morph); in (n_d, V_d) mode they remain free.
    fixed = ("nd", "vd", "g", "gr") if lay.glass is not None else ()
    free = {k: True for k in z if k not in fixed}
    for k, s in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:], start=1):
        # The penalty acts on the *effective* aspheric phase (1-s)*b; dividing by
        # (1-s)^2 keeps b itself bounded so it cannot blow up as s -> 1, and the
        # factor s lets it start at zero so the first steps are gentle.
        mu = mu_b * s / max(1.0 - s, 1e-3) ** 2 if s < 1 else 0.0
        cfg_kw = dict(s=s, beta=1.0, mu=mu, tmin=mech.tmin * s, emin=mech.emin * s,
                      gmin=mech.gmin, tmax=mech.tmax, virtual_petzval=virtual_petzval,
                      w_content=mech.w_content, nu=glass_pull * s**2)
        cfg = make_cfg(**cfg_kw)
        z, _ = optimize(z, lay, spec, cfg, free, iters=iters, transform=morph_to_params)
        p = morph_to_params(z, cfg, lay)
        powers = " ".join(f"{float(v) * 1e3:+6.1f}" for v in z["phi"])
        rec.add("materialize", k, p, lay, 1.0, f"s={s:.2f}  powers[1/m] {powers}",
                virtual_petzval, extra={"z": dict(z), "cfg": cfg_kw, "iters": iters})
    # polish as a plain glass system with independent curvatures
    free = {k: True for k in _glass_vars(lay, glass=lay.glass is None)}
    cfg = make_cfg(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                   w_content=mech.w_content)
    p, _ = optimize(p, lay, spec, cfg, free, iters=polish_iters)
    rec.add("materialize", steps + 1, p, lay, 1.0,
            f"all-glass polish  glasses {_glasses(p, lay)}")
    return p, lay


def stage_catalog(p, lay: Layout, spec: Spec, rec: Recorder, candidates=2, iters=100,
                  mech=Mech(), pull=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    if lay.glass is None and lay.region is not None:
        base = dict(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                    w_content=mech.w_content)
        p = pull_to_catalog(p, lay, spec, rec, base, {k: True for k in _glass_vars(lay)},
                            "catalog", pulls=pull)
        p, lay = to_glass_space(p, lay, lay.region)  # continuous (n_d, V_d) -> exact PCA
    if lay.glass is not None:
        cfg_kw = dict(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                      w_content=mech.w_content)
        return stage_snap(p, lay, spec, rec, cfg_kw, _glass_vars(lay, glass=False), "catalog",
                          iters=iters)
    free = {k: True for k in _glass_vars(lay)}
    base = dict(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                w_content=mech.w_content)
    for k, nu in enumerate(pull, start=1):
        p, _ = optimize(p, lay, spec, make_cfg(nu=nu, **base), free, iters=30)
        rec.add("catalog", k, p, lay, 1.0, f"glass pull nu={nu:.0e}  glasses {_glasses(p)}")

    options = [G.nearest_glasses(float(p["nd"][i]), float(p["vd"][i]), candidates)
               for i in range(lay.n_elem)]
    free_ng = {k: True for k in _glass_vars(lay) if k not in ("nd", "vd")}
    cfg = make_cfg(**base)
    best = None
    for combo in itertools.product(*options):
        q = dict(p)
        q["nd"] = jnp.array([g[1] for g in combo])
        q["vd"] = jnp.array([g[2] for g in combo])
        q, _ = optimize(q, lay, spec, cfg, free_ng, iters=iters)
        score = A.summary(q, lay, spec)["mean_wfe"]
        names = "/".join(g[0] for g in combo)
        if rec.verbose:
            print(f"      candidate {names:<30} mean RMS WFE {score:.4f} waves")
        if best is None or score < best[0]:
            best = (score, q, names)
    rec.add("catalog", len(pull) + 1, best[1], lay, 1.0, f"catalog glasses {best[2]}")
    return best[1], lay


def stage_snap(p, lay: Layout, spec: Spec, rec: Recorder, cfg_kw: dict, free_other, label,
               top_m=5, n_eval=4, iters=60, virtual_petzval=0.0):
    """Snap every element to a real catalog glass of `lay.glass` (PCA glass space).

    Candidates are ranked by *predicted merit*, not by distance in the glass map:
    with residuals r and their Jacobian J with respect to the glass coordinates g,
    the cost after moving to catalog coordinates g + D is predicted as |r + J D|^2.
    For each element the `top_m` best single moves and the `top_m` nearest glasses
    are kept; every combination is scored jointly with the same linear model, and the
    `n_eval` most promising combinations plus the all-nearest one are snapped exactly
    and re-optimized over `free_other`.  (The prediction is only reliable for small
    moves; the nearest glasses guard against it.)
    """
    from jax.flatten_util import ravel_pytree

    from .merit import residuals
    from .optimize import _compiled, free_mask

    gs = lay.glass
    n, k = lay.n_elem, gs.k
    cfg = make_cfg(**cfg_kw)
    full, unravel = ravel_pytree(p)
    idx = free_mask(p, {"g": True})
    _, jac = _compiled(lay, spec, unravel, tuple(int(i) for i in idx), None)
    r0 = np.asarray(residuals(p, lay, spec, cfg))
    J = np.asarray(jac(full[idx], full, cfg))  # [R, n*k], g flattened row-major
    g0 = np.asarray(p["g"])
    C = gs.coords / gs.scale  # catalog coordinates [G, k]

    top, nearest = [], []
    for i in range(n):
        Ji = J[:, i * k:(i + 1) * k]
        pred = np.nan_to_num(np.sum((r0[None, :] + (C - g0[i]) @ Ji.T) ** 2, axis=1),
                             nan=np.inf)
        # nearest = same (n_d/0.1, V_d/10) metric the catalog pull uses, so the
        # all-nearest combination is exactly where the pull ended.  (Raw index-curve
        # distance is dominated by the index level and ignores dispersion; normalized
        # PCA coordinates magnify the tiny partial-dispersion axis.)
        cur = gs.mu + gs.B @ (g0[i] * gs.scale) + np.asarray(p["gr"][i])
        jF, jd, jC = (gs.grid_index(w) for w in (G.LAMBDA_F, G.LAMBDA_D, G.LAMBDA_C))
        nd_i, vd_i = cur[jd], (cur[jd] - 1.0) / (cur[jF] - cur[jC])
        dist = ((gs.nd - nd_i) / 0.1) ** 2 + ((gs.vd - vd_i) / 10.0) ** 2
        near = list(np.argsort(dist)[:top_m])
        nearest.append(near[0])
        # The linear prediction is reliable for small moves only, so the nearest
        # glasses are always candidates alongside the best-predicted ones.
        top.append(list(dict.fromkeys(list(np.argsort(pred)[:top_m]) + near)))
    combos = list(itertools.product(*top))
    D = np.array([np.concatenate([C[j] - g0[i] for i, j in enumerate(c)]) for c in combos])
    pred = np.nan_to_num(np.sum((r0[None, :] + D @ J.T) ** 2, axis=1), nan=np.inf)
    order = list(np.argsort(pred)[:n_eval])
    o_near = combos.index(tuple(nearest))
    if o_near not in order:
        order.append(o_near)  # always try the all-nearest combination too

    free = {v: True for v in free_other}
    best = None
    for o in order:
        combo = combos[o]
        q = set_glass(dict(p), lay, list(range(n)), list(combo), exact=True)
        q, _ = optimize(q, lay, spec, cfg, free, iters=iters)
        cost = float(np.sum(np.asarray(residuals(q, lay, spec, cfg)) ** 2))
        if not np.isfinite(cost):
            cost = np.inf  # failed re-optimization (ray failures): never selected
        names = "/".join(gs.names[j] for j in combo)
        if rec.verbose:
            print(f"      {label}: {names:<34} predicted {pred[o]:.4g}  after re-opt {cost:.4g}")
        if best is None or cost < best[0]:
            best = (cost, q, names)
    if not np.isfinite(best[0]):
        # Every candidate failed to re-optimize (ray failures).  Keep the input design
        # and say so; the caller sees it in the recorder and the prescription labels.
        rec.add(label, 0, p, lay, cfg_kw.get("beta", 1.0),
                "SNAP FAILED: every candidate combination failed to re-optimize; "
                "glasses left unsnapped", virtual_petzval)
        return p, lay
    rec.add(label, 0, best[1], lay, cfg_kw.get("beta", 1.0), f"snapped to {best[2]}",
            virtual_petzval)
    return best[1], lay


def pull_to_catalog(p, lay: Layout, spec: Spec, rec: Recorder, cfg_kw: dict, free, label,
                    pulls=(1e-3, 1e-2, 1e-1), iters=30, virtual_petzval=0.0):
    """Gradually pull continuous (n_d, V_d) glasses toward the catalog's real glasses
    (softmin distance, weight ramped by `pulls`) while re-optimizing everything else,
    so that the exact snap that follows is a small step, not a jump."""
    free = dict(free, nd=True, vd=True)
    for k, nu in enumerate(pulls, start=1):
        p, _ = optimize(p, lay, spec, make_cfg(nu=nu, **cfg_kw), free, iters=iters)
        rec.add(label, k, p, lay, cfg_kw.get("beta", 1.0),
                f"glass pull nu={nu:.0e}  glasses {_glasses(p)}", virtual_petzval)
    return p


def from_glass_space(p, lay: Layout):
    """Exact PCA glasses -> continuous (n_d, V_d) normal-glass model of the same catalog."""
    q = dict(p, nd=G.nd_vec(p, lay), vd=G.vd_vec(p, lay))
    return q, dataclasses.replace(lay, glass=None, region=lay.glass)


def to_glass_space(p, lay: Layout, gs):
    """Re-express (n_d, V_d) Cauchy glasses exactly in a PCA glass space.

    Each element's Cauchy index curve on the glass-space grid is projected onto the
    principal directions (g) and the remainder kept as the fixed residual (gr), so the
    optical system is unchanged; afterwards the glasses can be snapped to the catalog.
    """
    lay_g = dataclasses.replace(lay, glass=gs)
    q = dict(p)
    g, gr = [], []
    for i in range(lay.n_elem):
        x = np.array([float(G.index_at(p, i, w, 1.0, lay)) for w in gs.grid])
        a = gs.project(x)
        g.append(a / gs.scale)
        gr.append(x - gs.mu - gs.B @ a)
    q["g"], q["gr"] = jnp.asarray(np.array(g)), jnp.asarray(np.array(gr))
    return q, lay_g


def fade_aspheres(z, cfg, layout: Layout):
    """Homotopy that removes rear-surface aspheres: asph = (1 - u) * a."""
    p = dict(z)
    p["asph"] = (1.0 - cfg["s"]) * z["asph"]
    return p


def stage_spherize(p, lay: Layout, spec: Spec, rec: Recorder, steps=10, iters=40, mech=Mech(),
                   mu_a=1e-4, polish_iters=150):
    """Aspheric design -> all-spherical design, by fading the aspheres out.

    The optimizer re-balances curvatures, thicknesses, spacings and glasses while
    the aspheric departure shrinks; a penalty on the *unscaled* coefficients keeps
    them from inflating to compensate.
    """
    if lay.glass is not None:
        # Exact catalog glasses -> continuous normal-glass (n_d, V_d) (index-curve error
        # ~1e-5), so the final catalog stage can pull and re-snap after the aspheres go.
        p, lay = from_glass_space(p, lay)
    # glasses stay fixed during the fade; the catalog stage runs afterwards
    free = {k: True for k in _glass_vars(lay, glass=False)}
    base = dict(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                w_content=mech.w_content)
    z = dict(p)
    for k, u in enumerate(np.linspace(0, 1, steps + 1)[1:], start=1):
        # penalty on the effective coefficients, scaled so it bounds the variable a
        mu = mu_a * u / max(1.0 - u, 1e-3) ** 2 if u < 1 else 0.0
        cfg = make_cfg(s=u, mu_asph=mu, **base)
        z, _ = optimize(z, lay, spec, cfg, free, iters=iters, transform=fade_aspheres)
        p = fade_aspheres(z, cfg, lay)
        rec.add("spherize", k, p, lay, 1.0, f"asphere weight {1 - u:.2f}")
    lay_s = dataclasses.replace(lay, aspheres=False)
    p = dict(p)
    p["asph"] = jnp.zeros_like(p["asph"])
    free = {k: True for k in _glass_vars(lay_s, glass=False)}
    p, _ = optimize(p, lay_s, spec, make_cfg(**base), free, iters=polish_iters)
    rec.add("spherize", steps + 1, p, lay_s, 1.0, "all-spherical polish")
    return stage_catalog(p, lay_s, spec, rec, mech=mech)


AXIAL_WEIGHT = 10.0  # w_axial of the "axial color first" ordering


PATTERNS = {
    "+-+": (2.0, -3.0, 2.0),
    "-++": (-2.0, 1.5, 1.5),
    "++-": (1.5, 1.5, -2.0),
    "+++": (1.0, 1.0, 1.0),
}


def explore_architectures(p0, lay0, spec: Spec, patterns=PATTERNS, stop_index=1, mech=Mech(),
                          virtual_petzval=1.0, verbose=True, aspheres=False, glass=None,
                          snap="early", stop_shift=False):
    """Run DISTRIBUTE + COLOR for several power patterns: cheap, all in phase space."""
    results = []
    for name, pat in patterns.items():
        rec = Recorder(spec, verbose=verbose)
        if verbose:
            print(f"--- architecture {name}")
        p, lay = stage_distribute(p0, lay0, spec, rec, n_elem=len(pat), stop_index=stop_index,
                                  mech=mech, virtual_petzval=virtual_petzval, pattern=pat,
                                  aspheres=aspheres, glass=glass)
        if not stop_shift:
            p, lay = stage_color(p, lay, spec, rec, mech=mech, virtual_petzval=virtual_petzval)
        else:
            # Two color orderings, both followed by the stop shift; keep the better.
            #  - merit color: polychromatic merit (axial color balanced against
            #    spherochromatism), then place the stop;
            #  - stop-shift ordering: correct primary axial color first (C_I -> 0, so
            #    lateral color no longer depends on the stop), then place the stop.
            # Neither wins for every architecture, and phase space is cheap.
            best = None
            p_d, lay_d, n_d = p, lay, len(rec.snaps)
            for w_ax, tag in ((0.0, "merit color"), (AXIAL_WEIGHT, "axial color first")):
                r = Recorder(spec, verbose=verbose)
                r.snaps = list(rec.snaps[:n_d])
                q, lq = stage_color(p_d, lay_d, spec, r, mech=mech,
                                    virtual_petzval=virtual_petzval, w_axial=w_ax)
                q, lq = stage_stop_shift(q, lq, spec, r, mech=mech,
                                         virtual_petzval=virtual_petzval, w_axial=w_ax)
                r.snaps[-1].note = f"[{tag}] " + r.snaps[-1].note
                score = r.snaps[-1].mean_wfe
                score = score if np.isfinite(score) else np.inf
                if best is None or score < best[0]:
                    best = (score, q, lq, r)
            _, p, lay, rec = best
        if glass is not None and snap == "early":
            # snap to real glasses while still in phase space: cheap, and the
            # architecture ranking then reflects glasses that actually exist
            cfg_kw = dict(beta=1.0, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval,
                          w_content=mech.w_content)
            p = pull_to_catalog(p, lay, spec, rec, cfg_kw,
                                {"phase": True, "gaps": True, "bfd": True}, "glass-pull",
                                virtual_petzval=virtual_petzval)
            p, lay = to_glass_space(p, lay, glass)
            p, lay = stage_snap(p, lay, spec, rec, cfg_kw, ("phase", "gaps", "bfd"),
                                "glass-snap", virtual_petzval=virtual_petzval)
        results.append({"name": name, "params": p, "layout": lay, "rec": rec,
                        "score": rec.snaps[-1].mean_wfe})
    results.sort(key=lambda r: r["score"] if np.isfinite(r["score"]) else np.inf)
    return results


def design_from_phase(spec: Spec, stop_index=1, mech=Mech(), virtual_petzval=1.0,
                      patterns=PATTERNS, top_k=2, aspheres=False, glass=None, snap="early",
                      morph_iters=30, glass_pull=0.1, stop_shift=False, verbose=True):
    """Full pipeline.  Returns (params, layout, recorder, info).

    `glass`: a catalog.GlassSpace of real catalog glasses.  Continuous stages use
    (n_d, V_d) bounded by the catalog's glass-map hull.  snap="early" snaps to exact
    catalog glasses in phase space after COLOR and keeps them fixed through the morph
    (re-checked at the end); snap="late" keeps glasses continuous through the morph and
    snaps once at the end; snap="gradual" is "late" plus a pull toward real glasses
    that ramps up during the morph.  glass=None: the built-in small (n_d, V_d) list.
    """
    rec0 = Recorder(spec, verbose=verbose)
    p0, lay0 = stage_ideal(spec, rec0)
    archs = explore_architectures(p0, lay0, spec, patterns, stop_index, mech, virtual_petzval,
                                  verbose, aspheres, glass, snap, stop_shift)
    if verbose:
        print("phase-space ranking:", ", ".join(f"{a['name']} {a['score']:.3f}" for a in archs))
    finals = []
    for a in archs[:top_k]:
        if verbose:
            print(f"=== materializing architecture {a['name']}")
        rec = a["rec"]
        rec.snaps[:0] = rec0.snaps
        p, lay = stage_materialize(a["params"], a["layout"], spec, rec, mech=mech,
                                   virtual_petzval=virtual_petzval, iters=morph_iters,
                                   glass_pull=glass_pull if snap == "gradual" else 0.0)
        p, lay = stage_catalog(p, lay, spec, rec, mech=mech)
        finals.append((rec.snaps[-1].mean_wfe, a["name"], p, lay, rec))
    finals.sort(key=lambda f: f[0] if np.isfinite(f[0]) else np.inf)
    score, name, p, lay, rec = finals[0]
    info = {"architectures": [(a["name"], a["score"]) for a in archs],
            "finals": [(f[1], f[0]) for f in finals], "chosen": name,
            "candidates": [{"name": f[1], "score": f[0], "params": f[2], "layout": f[3],
                            "rec": f[4]} for f in finals]}
    return p, lay, rec, info
