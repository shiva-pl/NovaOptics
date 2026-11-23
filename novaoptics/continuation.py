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

import itertools
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from . import analysis as A
from . import glass as G
from .merit import make_cfg, virtual_petzval_curvature
from .optimize import optimize
from .system import Layout, Spec, new_params


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


def _glass_vars(lay: Layout):
    names = ["c1", "c2", "t", "nd", "vd", "gaps", "bfd"]
    return names + ["asph"] if lay.aspheres else names


def _glasses(p):
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
                     aspheres=False):
    """Split the stage-0 power over `n_elem` screens.

    `pattern` gives relative powers (summing to 1), e.g. (2, -3, 2) for +/-/+;
    None means an equal split.  Screens of positive power start as a crown glass,
    negative ones as a flint (the usual achromat starting point); n_d and V_d are
    free from here on.
    """
    lay = Layout(n_elem=n_elem, stop_index=stop_index, n_phase=lay0.n_phase, r_norm=lay0.r_norm,
                 aspheres=aspheres)
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
                virtual_petzval=1.0):
    free = {"phase": True, "gaps": True, "bfd": True, "nd": True, "vd": True}
    for k, beta in enumerate(np.linspace(0, 1, steps + 1)[1:], start=1):
        cfg = make_cfg(beta=beta, gmin=2.0 * mech.gmin, virtual_petzval=virtual_petzval,
                       w_content=mech.w_content)
        p, _ = optimize(p, lay, spec, cfg, free, iters=iters)
        rec.add("color", k, p, lay, beta, f"beta={beta:.2f}  glasses {_glasses(p)}",
                virtual_petzval)
    return p, lay


X_MAX = 4.0


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
    dc = s * z["phi"] / (z["nd"] - 1.0)
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
        asph = s * z["bhi"] / (z["nd"][:, None] - 1.0)
    else:
        asph = jnp.zeros_like(z["bhi"])
    return {"c1": c1, "c2": c2, "t": z["t"], "nd": z["nd"], "vd": z["vd"], "phase": phase,
            "asph": asph, "gaps": z["gaps"], "bfd": z["bfd"]}


def stage_materialize(p, lay: Layout, spec: Spec, rec: Recorder, steps=20, iters=30,
                      mech=Mech(), virtual_petzval=1.0, mu_b=1e-5, polish_iters=80,
                      free_curvature=True):
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
    if free_curvature:
        # Curvature corrections on top of the thin-lens rule (scaled by s): the path-gap
        # diagnostic showed the thin-lens manifold alone costs 3-7x in merit mid-morph.
        z["dc1"] = jnp.zeros(n)
        z["dc2"] = jnp.zeros(n)
    free = {k: True for k in z}
    for k, s in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:], start=1):
        # The penalty acts on the *effective* aspheric phase (1-s)*b; dividing by
        # (1-s)^2 keeps b itself bounded so it cannot blow up as s -> 1, and the
        # factor s lets it start at zero so the first steps are gentle.
        mu = mu_b * s / max(1.0 - s, 1e-3) ** 2 if s < 1 else 0.0
        cfg_kw = dict(s=s, beta=1.0, mu=mu, tmin=mech.tmin * s, emin=mech.emin * s,
                      gmin=mech.gmin, tmax=mech.tmax, virtual_petzval=virtual_petzval,
                      w_content=mech.w_content)
        cfg = make_cfg(**cfg_kw)
        z, _ = optimize(z, lay, spec, cfg, free, iters=iters, transform=morph_to_params)
        p = morph_to_params(z, cfg, lay)
        powers = " ".join(f"{float(v) * 1e3:+6.1f}" for v in z["phi"])
        rec.add("materialize", k, p, lay, 1.0, f"s={s:.2f}  powers[1/m] {powers}",
                virtual_petzval, extra={"z": dict(z), "cfg": cfg_kw, "iters": iters})
    # polish as a plain glass system with independent curvatures
    free = {k: True for k in _glass_vars(lay)}
    cfg = make_cfg(beta=1.0, tmin=mech.tmin, emin=mech.emin, gmin=mech.gmin, tmax=mech.tmax,
                   w_content=mech.w_content)
    p, _ = optimize(p, lay, spec, cfg, free, iters=polish_iters)
    rec.add("materialize", steps + 1, p, lay, 1.0, f"all-glass polish  glasses {_glasses(p)}")
    return p, lay


def stage_catalog(p, lay: Layout, spec: Spec, rec: Recorder, candidates=2, iters=100,
                  mech=Mech(), pull=(1e-3, 1e-2, 1e-1)):
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
    # glasses stay at their catalog values here; the catalog stage runs afterwards
    free = {k: True for k in _glass_vars(lay) if k not in ("nd", "vd")}
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
    lay_s = Layout(n_elem=lay.n_elem, stop_index=lay.stop_index, n_phase=lay.n_phase,
                   r_norm=lay.r_norm, aspheres=False)
    p = dict(p)
    p["asph"] = jnp.zeros_like(p["asph"])
    free = {k: True for k in _glass_vars(lay_s) if k not in ("nd", "vd")}
    p, _ = optimize(p, lay_s, spec, make_cfg(**base), free, iters=polish_iters)
    rec.add("spherize", steps + 1, p, lay_s, 1.0, "all-spherical polish")
    return stage_catalog(p, lay_s, spec, rec, mech=mech)


PATTERNS = {
    "+-+": (2.0, -3.0, 2.0),
    "-++": (-2.0, 1.5, 1.5),
    "++-": (1.5, 1.5, -2.0),
    "+++": (1.0, 1.0, 1.0),
}


def explore_architectures(p0, lay0, spec: Spec, patterns=PATTERNS, stop_index=1, mech=Mech(),
                          virtual_petzval=1.0, verbose=True, aspheres=False):
    """Run DISTRIBUTE + COLOR for several power patterns: cheap, all in phase space."""
    results = []
    for name, pat in patterns.items():
        rec = Recorder(spec, verbose=verbose)
        if verbose:
            print(f"--- architecture {name}")
        p, lay = stage_distribute(p0, lay0, spec, rec, n_elem=len(pat), stop_index=stop_index,
                                  mech=mech, virtual_petzval=virtual_petzval, pattern=pat,
                                  aspheres=aspheres)
        p, lay = stage_color(p, lay, spec, rec, mech=mech, virtual_petzval=virtual_petzval)
        results.append({"name": name, "params": p, "layout": lay, "rec": rec,
                        "score": rec.snaps[-1].mean_wfe})
    results.sort(key=lambda r: r["score"] if np.isfinite(r["score"]) else np.inf)
    return results


def design_from_phase(spec: Spec, stop_index=1, mech=Mech(), virtual_petzval=1.0,
                      patterns=PATTERNS, top_k=2, aspheres=False, verbose=True):
    """Full pipeline.  Returns (params, layout, recorder, info)."""
    rec0 = Recorder(spec, verbose=verbose)
    p0, lay0 = stage_ideal(spec, rec0)
    archs = explore_architectures(p0, lay0, spec, patterns, stop_index, mech, virtual_petzval,
                                  verbose, aspheres)
    if verbose:
        print("phase-space ranking:", ", ".join(f"{a['name']} {a['score']:.3f}" for a in archs))
    finals = []
    for a in archs[:top_k]:
        if verbose:
            print(f"=== materializing architecture {a['name']}")
        rec = a["rec"]
        rec.snaps[:0] = rec0.snaps
        p, lay = stage_materialize(a["params"], a["layout"], spec, rec, mech=mech,
                                   virtual_petzval=virtual_petzval)
        p, lay = stage_catalog(p, lay, spec, rec, mech=mech)
        finals.append((rec.snaps[-1].mean_wfe, a["name"], p, lay, rec))
    finals.sort(key=lambda f: f[0])
    score, name, p, lay, rec = finals[0]
    info = {"architectures": [(a["name"], a["score"]) for a in archs],
            "finals": [(f[1], f[0]) for f in finals], "chosen": name,
            "candidates": [{"name": f[1], "score": f[0], "params": f[2], "layout": f[3],
                            "rec": f[4]} for f in finals]}
    return p, lay, rec, info
