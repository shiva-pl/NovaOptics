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
(GROW)                optional: split screens in phase space until the target number of
                      elements is reached (grow.py).
Stage 4  CATALOG      pull (n_d, V_d) to real catalog glasses, snap (candidates ranked by
                      predicted merit) and re-optimize.  With a catalog the pull already
                      ramps up during the morph, so the snap is a small step.

Optional glass-system stages before CATALOG: STOP-SHIFT (stop placed by stop-shift
theory) and BENDING HOPS (basin hopping over element bendings).

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
from . import progress
from .merit import make_cfg, virtual_petzval_curvature
from .optimize import optimize
from .parallel import pmap
from .system import Layout, Spec, glass_label, new_params, set_glass


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
    label: str = ""  # e.g. the architecture, for progress listeners

    def add(self, stage, step, params, layout, beta, note="", virtual_petzval=0.0, extra=None):
        # Phase stages are judged on the image surface they are designed for: the
        # flat sensor, pre-curved by the Petzval their phase will acquire as glass.
        c_img = virtual_petzval * float(virtual_petzval_curvature(params, layout))
        s = A.summary(params, layout, self.spec, beta, c_img=c_img)
        snap = Snapshot(stage, step, params, layout, float(beta), s["mean_wfe"], s["rms_wfe"],
                        s["efl"], note, dict(extra or {}, virtual_petzval=virtual_petzval))
        self.snaps.append(snap)
        progress.emit(self, snap)
        if self.verbose:
            wf = " ".join(f"{w:6.3f}" for w in s["rms_wfe"])
            print(f"[{stage:<11}] {step:2d}  EFL={s['efl']:7.3f}  RMS WFE [waves]: {wf}   {note}")
        return snap


@dataclass(frozen=True)
class Mech:
    """Manufacturing limits applied at the end of the morph [mm]."""

    tmin: float = 2.0
    emin: float = 1.0
    gmin: float = 1.0
    tmax: float = 8.0

    def base(self):
        """Merit configuration keywords of a finished glass system."""
        return dict(beta=1.0, tmin=self.tmin, emin=self.emin, gmin=self.gmin, tmax=self.tmax)

    def cfg(self, **kw):
        return make_cfg(**self.base(), **kw)


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


# --------------------------------------------------------------------------- phase space

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
    # glass-map hull bounds them, and real glasses arrive by snapping at the end.
    lay = Layout(n_elem=n_elem, stop_index=stop_index, n_phase=lay0.n_phase, r_norm=lay0.r_norm,
                 aspheres=aspheres, region=glass)
    gaps = np.full(n_elem, spacing)
    if 0 < stop_index < n_elem:
        gaps[stop_index] = 0.5  # stop just in front of element `stop_index`
    pattern = np.ones(n_elem) / n_elem if pattern is None else np.asarray(pattern, float)
    pattern = pattern / pattern.sum()
    p = new_params(lay, gaps=gaps, bfd=float(p0["bfd"]))
    p["nd"] = jnp.full(n_elem, 1.62)
    p["vd"] = jnp.asarray(np.where(pattern > 0, 60.0, 36.0))
    # Split by optical power only: away from the stop the beam footprint exceeds the
    # region where the stage-0 polynomial was fitted, so its high orders would
    # extrapolate wildly.  The optimizer re-grows the aspheric orders it needs.
    p["phase"] = jnp.zeros((n_elem, lay.n_phase)).at[:, 0].set(p0["phase"][0, 0] * pattern)
    rec.add("distribute", 0, p, lay, 0.0, f"split into {n_elem} screens  "
            f"powers[1/m] {_powers(p['phase'], lay.r_norm)}", virtual_petzval)
    cfg = make_cfg(beta=0.0, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval)
    free = {"phase": True, "gaps": True, "bfd": True, "nd": True}
    p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
    rec.add("distribute", 1, p, lay, 0.0,
            f"optimized w/ virtual Petzval  powers[1/m] {_powers(p['phase'], lay.r_norm)}",
            virtual_petzval)
    return p, lay


def stage_color(p, lay: Layout, spec: Spec, rec: Recorder, steps=5, iters=40, mech=Mech(),
                virtual_petzval=1.0):
    free = {"phase": True, "gaps": True, "bfd": True, **{v: True for v in _gvars(lay)}}
    for k, beta in enumerate(np.linspace(0, 1, steps + 1)[1:], start=1):
        cfg = make_cfg(beta=beta, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval)
        p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
        rec.add("color", k, p, lay, beta, f"beta={beta:.2f}  glasses {_glasses(p, lay)}",
                virtual_petzval)
    return p, lay


# --------------------------------------------------------------------------- morph

X_MAX = 4.0


def morph_to_params(z, cfg, layout: Layout):
    """Homotopy between an ideal phase screen (s = 0) and a glass singlet (s = 1).

    Design variables per element: total power phi, bending xi, curvature corrections
    dc1/dc2, thickness t, glass (n_d, V_d) and aspheric phase b_k.  The glass carries
    s*phi (thin-lens curvatures from phi, X and n_d, plus s*dc), the ideal phase carries
    (1 - s)*phi plus (1 - s)*b_k.  With aspheres enabled the glass also takes s*b_k as
    rear-surface aspheric sag.  Every s is a smooth deformation of the previous system,
    and at s = 1 the phase is identically zero.
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
        # curvature corrections beyond the thin-lens rule (thick-lens effects)
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
    return {"c1": c1, "c2": c2, "t": z["t"], "nd": nd, "vd": G.vd_vec(z, layout),
            "phase": phase, "asph": asph, "gaps": z["gaps"], "bfd": z["bfd"]}


def stage_materialize(p, lay: Layout, spec: Spec, rec: Recorder, steps=20, iters=30,
                      mech=Mech(), virtual_petzval=1.0, mu_b=1e-5, polish_iters=80,
                      glass_pull=0.0, xi0=None):
    """Morph phase into glass (see `morph_to_params`).

    xi0: initial bending variable per element (shape factor X = 4 tanh(xi)); default 0,
    i.e. equiconvex/equiconcave.
    glass_pull > 0 (catalog region set): the pull toward real catalog glasses ramps up
    with the morph, nu = glass_pull * s^2, so glass discreteness is turned on gradually
    while the lens takes shape and the final snap is a small step.
    """
    n = lay.n_elem
    z = {
        "phi": -2.0 * p["phase"][:, 0] / lay.r_norm**2,
        "xi": jnp.zeros(n) if xi0 is None else jnp.asarray(xi0, dtype=float),
        "bhi": p["phase"][:, 1:],
        "t": jnp.zeros(n),
        "nd": p["nd"],
        "vd": p["vd"],
        "gaps": p["gaps"],
        "bfd": p["bfd"],
        # Curvature corrections on top of the thin-lens rule (scaled by s): the path-gap
        # diagnostic showed the thin-lens manifold alone costs 3-7x in merit mid-morph.
        "dc1": jnp.zeros(n),
        "dc2": jnp.zeros(n),
    }
    free = {k: True for k in z}
    for k, s in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:], start=1):
        # The penalty acts on the *effective* aspheric phase (1-s)*b; dividing by
        # (1-s)^2 keeps b itself bounded so it cannot blow up as s -> 1, and the
        # factor s lets it start at zero so the first steps are gentle.
        mu = mu_b * s / max(1.0 - s, 1e-3) ** 2 if s < 1 else 0.0
        cfg_kw = dict(s=s, beta=1.0, mu=mu, tmin=mech.tmin * s, emin=mech.emin * s,
                      gmin=mech.gmin, tmax=mech.tmax, virtual_petzval=virtual_petzval,
                      nu=glass_pull * s**2)
        cfg = make_cfg(**cfg_kw)
        z, _ = optimize(z, lay, spec, cfg, free, iters=iters, transform=morph_to_params)
        p = morph_to_params(z, cfg, lay)
        powers = " ".join(f"{float(v) * 1e3:+6.1f}" for v in z["phi"])
        rec.add("materialize", k, p, lay, 1.0, f"s={s:.2f}  powers[1/m] {powers}",
                virtual_petzval, extra={"z": dict(z), "cfg": cfg_kw, "iters": iters})
    # polish as a plain glass system with independent curvatures
    free = {k: True for k in _glass_vars(lay)}
    p, _ = optimize(p, lay, spec, mech.cfg(), free, iters=polish_iters)
    rec.add("materialize", steps + 1, p, lay, 1.0,
            f"all-glass polish  glasses {_glasses(p, lay)}")
    return p, lay


# --------------------------------------------------------------------------- glass system

def stage_stop_shift(p, lay: Layout, spec: Spec, rec: Recorder, mech=Mech(), k=3, iters=60):
    """STOP-SHIFT on a glass system: place the aperture stop with stop-shift theory.

    Stop-shift theory (see stopshift.py) predicts the third-order coma, astigmatism and
    field curvature for every stop position in air in closed form.  The `k` best
    predicted positions and the current one are tried for real: the stop is moved
    there and the system re-optimized with the full real-ray merit (curvatures,
    thicknesses, spacings, aspheres, continuous glasses).  The best real result is
    kept, so the stage can only improve the design.

    (Placing the stop in phase space was tried and misleads: thin, freely aspheric
    screens want a different stop than thick glass elements.)
    """
    from . import stopshift as SS

    sc = SS.scan(p, lay, spec, beta=1.0, virtual_petzval=0.0)
    z_now = sc["current_z"]
    i_now = int(np.argmin(np.abs(sc["z"] - z_now)))
    cands = SS.candidates(sc, k=k)
    cfg = mech.cfg()
    free = {v: True for v in _glass_vars(lay, glass=lay.glass is None)}
    def try_position(z):
        # keep the current layout for the current position: a relayout could put a stop
        # that sits exactly at a vertex on the other side of that element
        q, lq = (p, lay) if z == z_now else SS.relayout(p, lay, z)
        q, info = optimize(q, lq, spec, cfg, free, iters=iters)
        cost = info.get("cost", np.inf)
        return (cost if np.isfinite(cost) else np.inf), q, lq, z

    best = None
    for cost, q, lq, z in pmap(try_position,
                               [z_now] + [c for c in cands if abs(c - z_now) > 0.5]):
        i = int(np.argmin(np.abs(sc["z"] - z)))
        if rec.verbose:
            print(f"      stop at z={z:+7.2f} mm (third-order merit {sc['merit'][i]:9.3g}) "
                  f"-> real cost {cost:.4g}")
        if best is None or cost < best[0]:
            best = (cost, q, lq, z, i)
    _, q, lq, z, i = best
    rec.add("stop-shift", 0, q, lq, 1.0,
            f"stop {z_now:+.1f} -> {z:+.1f} mm (stop_index {lq.stop_index}); third-order "
            f"merit {sc['merit'][i_now]:.3g} -> {sc['merit'][i]:.3g}", extra={"scan": sc})
    return q, lq


def stage_bending_hops(p, lay: Layout, spec: Spec, rec: Recorder, mech=Mech(),
                       X_alt=(-3.0, -1.5, 0.0, 1.5, 3.0), iters=100, label="bending hops"):
    """Basin hopping over element bendings on a glass system.

    The morph picks each element's bending basin early, when bending barely matters,
    and cannot leave it later.  Both basins can be fully corrected in third order (they
    differ in fifth-order residuals), so third-order theory cannot choose between them,
    but the real-ray merit can.  For each element in turn, every shape factor in
    `X_alt` is tried from the same starting design (power c1 - c2 kept, system
    re-polished), and the best improvement of the merit is kept.
    """
    from .bending import rebend

    cfg = mech.cfg()
    free = {v: True for v in _glass_vars(lay, glass=lay.glass is None)}

    def polish(q, n):
        q, info = optimize(q, lay, spec, cfg, free, iters=n)  # n = 0: compiled cost only
        c = info.get("cost", np.inf)
        return q, (c if np.isfinite(c) else np.inf)

    def shape(q, i):
        c1, c2 = float(q["c1"][i]), float(q["c2"][i])
        return (c1 + c2) / (c1 - c2) if c1 != c2 else 0.0

    best, best_cost = polish(p, 0)
    rec.add(label, 0, p, lay, 1.0, "before")
    moves = []
    for i in range(lay.n_elem):
        base, X_now, found = best, shape(best, i), None
        alts = [X for X in X_alt if abs(X - X_now) >= 0.75]
        # the alternatives all start from the same design: independent, so concurrent
        tried = pmap(lambda X: polish(rebend(base, i, X), iters), alts)
        for q, c in tried:
            if c < best_cost:
                best, best_cost, found = q, c, shape(q, i)
        if found is not None:
            moves.append(f"E{i} {X_now:+.1f}->{found:+.1f}")
    rec.add(label, 1, best, lay, 1.0,
            "kept: " + (", ".join(moves) if moves else "no improving hop"))
    return best, lay


# --------------------------------------------------------------------------- glasses

def stage_catalog(p, lay: Layout, spec: Spec, rec: Recorder, candidates=2, iters=100,
                  mech=Mech(), pull=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Pull continuous glasses to the catalog and snap to real glasses.

    With a catalog region (`lay.region`): pull in (n_d, V_d), convert exactly to the PCA
    glass space and snap with `stage_snap`.  With exact catalog glasses (`lay.glass`):
    re-check neighbours with `stage_snap`.  Otherwise the built-in (n_d, V_d) list.
    """
    base = mech.base()
    if lay.glass is None and lay.region is not None:
        p = pull_to_catalog(p, lay, spec, rec, base, {k: True for k in _glass_vars(lay)},
                            "catalog", pulls=pull)
        p, lay = to_glass_space(p, lay, lay.region)  # continuous (n_d, V_d) -> exact PCA
    if lay.glass is not None:
        return stage_snap(p, lay, spec, rec, base, _glass_vars(lay, glass=False), "catalog",
                          iters=iters)
    free = {k: True for k in _glass_vars(lay)}
    for k, nu in enumerate(pull, start=1):
        p, _ = optimize(p, lay, spec, make_cfg(nu=nu, **base), free, iters=30)
        rec.add("catalog", k, p, lay, 1.0, f"glass pull nu={nu:.0e}  glasses {_glasses(p)}")

    options = [G.nearest_glasses(float(p["nd"][i]), float(p["vd"][i]), candidates)
               for i in range(lay.n_elem)]
    free_ng = {k: True for k in _glass_vars(lay) if k not in ("nd", "vd")}
    cfg = make_cfg(**base)
    combos = list(itertools.product(*options))
    if len(combos) > 16:  # many elements: all-nearest plus one alternative at a time
        near = tuple(o[0] for o in options)
        combos = [near] + [near[:i] + (o,) + near[i + 1:] for i in range(lay.n_elem)
                           for o in options[i][1:]]
    best = None
    for combo in combos:
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

    from .optimize import _compiled, ensure_compiled, free_mask

    gs = lay.glass
    n, k = lay.n_elem, gs.k
    cfg = make_cfg(**cfg_kw)
    full, unravel = ravel_pytree(p)
    idx = free_mask(p, {"g": True})
    entry = _compiled(lay, spec, unravel, tuple(int(i) for i in idx), None)
    ensure_compiled(entry, full[idx], full, cfg)
    res, jac = entry[:2]
    r0 = np.asarray(res(full[idx], full, cfg))
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
    combos, pred = _rank_combos(r0, J, C, g0, top, k)
    order = list(range(min(n_eval, len(combos))))
    if tuple(nearest) not in combos:
        combos.append(tuple(nearest))
        pred = np.r_[pred, _predict(r0, J, C, g0, [tuple(nearest)], k)]
    o_near = combos.index(tuple(nearest))
    if o_near not in order:
        order.append(o_near)  # always try the all-nearest combination too

    free = {v: True for v in free_other}

    def try_combo(o):
        q = set_glass(dict(p), lay, list(range(n)), list(combos[o]), exact=True)
        q, info = optimize(q, lay, spec, cfg, free, iters=iters)
        cost = info.get("cost", np.inf)
        # a failed re-optimization (ray failures) is never selected
        return o, q, (cost if np.isfinite(cost) else np.inf)

    best = None
    for o, q, cost in pmap(try_combo, order):
        combo = combos[o]
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


def _predict(r0, J, C, g0, combos, k):
    D = np.array([np.concatenate([C[j] - g0[i] for i, j in enumerate(c)]) for c in combos])
    return np.nan_to_num(np.sum((r0[None, :] + D @ J.T) ** 2, axis=1), nan=np.inf)


def _rank_combos(r0, J, C, g0, top, k, exhaustive=20000, beam=400):
    """Glass combinations (one candidate per element) sorted by predicted cost
    |r0 + J D|^2.  All combinations while there are at most `exhaustive`; beyond
    that (many elements) a beam search over elements on the same model, keeping the
    `beam` best partial combinations (unassigned elements stay where they are)."""
    n = len(top)
    if np.prod([len(t) for t in top], dtype=float) <= exhaustive:
        combos = list(itertools.product(*top))
        pred = _predict(r0, J, C, g0, combos, k)
    else:
        # per-element residual changes u[i][j] = J_i (C_j - g0_i)
        u = [{j: J[:, i * k:(i + 1) * k] @ (C[j] - g0[i]) for j in top[i]} for i in range(n)]
        partial = [((), r0)]
        for i in range(n):
            ext = [(c + (j,), v + u[i][j]) for c, v in partial for j in top[i]]
            cost = np.nan_to_num([float(v @ v) for _, v in ext], nan=np.inf)
            partial = [ext[m] for m in np.argsort(cost)[:beam]]
        combos = [c for c, _ in partial]
        pred = np.nan_to_num([float(v @ v) for _, v in partial], nan=np.inf)
    order = np.argsort(pred)
    return [combos[o] for o in order], np.asarray(pred)[order]


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
    """Re-express continuous (n_d, V_d) glasses exactly in a PCA glass space.

    Each element's index curve on the glass-space grid is projected onto the principal
    directions (g) and the remainder kept as the fixed residual (gr), so the optical
    system is unchanged; afterwards the glasses can be snapped to the catalog.
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


# --------------------------------------------------------------------------- spherical

def fade_aspheres(z, cfg, layout: Layout):
    """Homotopy that removes rear-surface aspheres: asph = (1 - u) * a."""
    p = dict(z)
    p["asph"] = (1.0 - cfg["s"]) * z["asph"]
    return p


def stage_spherize(p, lay: Layout, spec: Spec, rec: Recorder, steps=10, iters=40, mech=Mech(),
                   mu_a=1e-4, polish_iters=150, bending_hops=False):
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
    base = mech.base()
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
    if bending_hops:
        p, lay_s = stage_bending_hops(p, lay_s, spec, rec, mech=mech)
    return stage_catalog(p, lay_s, spec, rec, mech=mech)


# --------------------------------------------------------------------------- pipeline

PATTERNS = {
    "+-+": (2.0, -3.0, 2.0),
    "-++": (-2.0, 1.5, 1.5),
    "++-": (1.5, 1.5, -2.0),
    "+++": (1.0, 1.0, 1.0),
}


def explore_architectures(p0, lay0, spec: Spec, patterns=PATTERNS, stop_index=1, mech=Mech(),
                          virtual_petzval=1.0, verbose=True, aspheres=False, glass=None):
    """Run DISTRIBUTE + COLOR for several power patterns: cheap, all in phase space."""
    def explore(item):
        name, pat = item
        rec = Recorder(spec, verbose=verbose, label=name)
        if verbose:
            print(f"--- architecture {name}")
        p, lay = stage_distribute(p0, lay0, spec, rec, n_elem=len(pat), stop_index=stop_index,
                                  mech=mech, virtual_petzval=virtual_petzval, pattern=pat,
                                  aspheres=aspheres, glass=glass)
        p, lay = stage_color(p, lay, spec, rec, mech=mech, virtual_petzval=virtual_petzval)
        return {"name": name, "params": p, "layout": lay, "rec": rec,
                "score": rec.snaps[-1].mean_wfe}

    results = pmap(explore, patterns.items())  # architectures are independent
    results.sort(key=lambda r: r["score"] if np.isfinite(r["score"]) else np.inf)
    return results


def design_from_phase(spec: Spec, stop_index=1, mech=Mech(), virtual_petzval=1.0,
                      patterns=PATTERNS, top_k=2, aspheres=False, glass=None, morph_iters=30,
                      glass_pull=0.1, stop_shift=False, bending_hops=False, verbose=True,
                      n_elem=None):
    """Full pipeline.  Returns (params, layout, recorder, info).

    glass: a catalog.GlassSpace of real catalog glasses.  Glasses stay continuous
    (n_d, V_d), bounded by the catalog's glass-map hull, with the catalog's normal-glass
    dispersion; a pull toward real glasses ramps up during the morph (`glass_pull`), and
    the CATALOG stage snaps to exact catalog glasses.  None: the built-in (n_d, V_d) list.
    stop_shift: place the stop on the materialized glass system (`stage_stop_shift`).
    bending_hops: re-bend each element to alternative shapes after the morph and keep
    real-merit improvements (`stage_bending_hops`).
    n_elem: number of elements of the final design.  The architectures (`patterns`)
    start with len(pattern) screens; GROW splits screens in phase space up to n_elem
    (see grow.py) before the morph.
    """
    rec0 = Recorder(spec, verbose=verbose, label="ideal")
    p0, lay0 = stage_ideal(spec, rec0)
    archs = explore_architectures(p0, lay0, spec, patterns, stop_index, mech, virtual_petzval,
                                  verbose, aspheres, glass)
    if verbose:
        print("phase-space ranking:", ", ".join(f"{a['name']} {a['score']:.3f}" for a in archs))
    def finish(a):
        if verbose:
            print(f"=== materializing architecture {a['name']}")
        rec = a["rec"]
        rec.snaps[:0] = rec0.snaps
        p, lay = a["params"], a["layout"]
        if n_elem and n_elem > lay.n_elem:
            from .grow import stage_grow

            p, lay = stage_grow(p, lay, spec, rec, n_elem, mech=mech,
                                virtual_petzval=virtual_petzval)
        p, lay = stage_materialize(p, lay, spec, rec, mech=mech,
                                   virtual_petzval=virtual_petzval, iters=morph_iters,
                                   glass_pull=glass_pull if glass is not None else 0.0)
        if bending_hops:
            p, lay = stage_bending_hops(p, lay, spec, rec, mech=mech)
        if stop_shift:
            p, lay = stage_stop_shift(p, lay, spec, rec, mech=mech)
        p, lay = stage_catalog(p, lay, spec, rec, mech=mech)
        return (rec.snaps[-1].mean_wfe, a["name"], p, lay, rec)

    finals = pmap(finish, archs[:top_k])  # architectures are independent
    finals.sort(key=lambda f: f[0] if np.isfinite(f[0]) else np.inf)
    score, name, p, lay, rec = finals[0]
    info = {"architectures": [(a["name"], a["score"]) for a in archs],
            "finals": [(f[1], f[0]) for f in finals], "chosen": name,
            "candidates": [{"name": f[1], "score": f[0], "params": f[2], "layout": f[3],
                            "rec": f[4]} for f in finals]}
    return p, lay, rec, info
