"""Design operations on surface-engine systems.

The phase-first pipeline (continuation.py) creates designs from nothing in the element
model; here they are continued as general surface systems:

* `from_element`        an element-model design as an engine Problem + params
* `variables`           the usual variable set (curvatures, thicknesses, glasses, ...)
* `optimize_design`     damped least squares on the engine merit (all configurations)
* `add_element`         one more air-spaced element: a zero-power phase screen is
                        inserted into an air space (exactly no change), optimized, and
                        morphed into a glass singlet; every air space is tried
* `cement`              a zero-thickness layer of a new glass is inserted behind an
                        element (exactly no change) and grown into a cemented doublet
* `snap_glasses`        model (n_d, V_d) glasses -> nearest preferred catalog glasses,
                        each combination re-optimized
* `per_wavelength_focus` configurations with one wavelength each and its own image
                        distance (lenses with intentional axial color)

Every structural change starts exactly continuous; the optimizer then uses the new
freedom, the same principle as the phase-to-glass morph.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax.numpy as jnp
import numpy as np

from .. import glass as G
from ..optimize import optimize
from ..parallel import pmap
from . import merit as EM
from .system import Spec, Surface, System, surface_params

AIR = ("air", "", "vacuum")
X_MAX = 4.0


# --------------------------------------------------------------------------- basics

def from_element(params, layout, spec, region=None):
    """(Problem, params, engine Spec) for an element-model design."""
    from .convert import from_layout

    system, prm, espec = from_layout(params, layout, spec.epd,
                                     [f * spec.half_field_deg for f in spec.field_fracs],
                                     spec.wavelengths)
    targets = EM.Targets(efl=spec.efl, max_distortion=spec.max_distortion,
                         max_track=spec.max_track, region=region)
    problem = EM.Problem(system, targets=targets)
    return problem, EM.make_params(prm, problem), espec


def _overridden_everywhere(problem):
    sets = [set(c.overrides) for c in problem.configs]
    return set.intersection(*sets) if sets else set()


def variables(problem: EM.Problem, curvatures=True, thicknesses=True, glasses=True,
              aspheres=True, conics=False, configs=True, exclude=()):
    """Free-variable tree: curvatures, thicknesses (every space but the image's),
    model glasses, aspheric coefficients and configuration values.  Base values that
    every configuration overrides are not free (they would have no effect)."""
    S = problem.system.surfaces
    over = _overridden_everywhere(problem)
    surf = []
    for i, s in enumerate(S):
        d = {}
        if i not in exclude:
            if s.interact in ("refract", "reflect"):
                if curvatures:
                    d["c"] = True
                if conics and s.shape != "qbfs":
                    d["k"] = True
                if aspheres and s.n_terms:
                    d["coef"] = True
            if glasses and s.material == "model":
                d["nd"] = d["vd"] = True
            if thicknesses and i < len(S) - 1:
                d["t"] = True
        for k in list(d):
            if (i, k) in over:
                del d[k]
        surf.append(d)
    mc = [{EM.key_name(i, k): True for i, k in c.overrides} if configs else {}
          for c in problem.configs]
    return {"s": surf, "mc": mc}


def optimize_design(problem: EM.Problem, params, spec: Spec, free=None, iters=100,
                    transform=None, **cfg_kw):
    cfg = EM.make_cfg(**cfg_kw)
    return optimize(params, problem, spec, cfg, free if free is not None else variables(problem),
                    iters=iters, transform=transform, residual_fn=EM.residuals)


def cost(problem, params, spec, **cfg_kw):
    return optimize_design(problem, params, spec, iters=0, **cfg_kw)[1].get("cost", np.inf)


def mean_wfe(problem, params, spec):
    return float(np.mean(np.concatenate(EM.rms_wfe(params, problem, spec))))


def clear_semi_diameters(problem, params, spec):
    """Largest ray height on every surface over all configurations [mm]."""
    sd = None
    for c, conf in enumerate(problem.configs):
        ev = EM.evaluate_config(EM.config_surfaces(params, problem, c), problem.system,
                                EM.config_spec(spec, conf), problem.targets)
        h = np.sqrt(np.max(np.sum(np.asarray(ev["local"])[..., :2] ** 2, -1), axis=(0, 1, 2)))
        sd = h if sd is None else np.maximum(sd, h)
    return sd


def insert_surfaces(problem: EM.Problem, params, at, surfaces, values):
    """Insert surfaces (with their parameter dicts) before index `at`; configuration
    overrides and morphing indices are shifted."""
    S = list(problem.system.surfaces)
    n = len(surfaces)
    S[at:at] = surfaces
    sp = [dict(p) for p in params["s"]]
    sp[at:at] = [surface_params(s, **v) for s, v in zip(surfaces, values)]

    def sh(i):
        return i + n if i >= at else i

    configs = tuple(dataclasses.replace(c, overrides=tuple((sh(i), k) for i, k in c.overrides))
                    for c in problem.configs)
    mc = [{EM.key_name(sh(i), k): params["mc"][ci][EM.key_name(i, k)] for i, k in c.overrides}
          for ci, c in enumerate(problem.configs)]
    morphing = tuple((sh(a), sh(b)) for a, b in problem.morphing)
    prob = dataclasses.replace(problem, system=dataclasses.replace(problem.system,
                                                                   surfaces=tuple(S)),
                               configs=configs, morphing=morphing)
    return prob, {"s": sp, "mc": mc}


def replace_surface(problem, params, i, surface, **values):
    S = list(problem.system.surfaces)
    S[i] = surface
    sp = [dict(p) for p in params["s"]]
    old = sp[i]
    new = surface_params(surface, **values)
    for k in new:
        if k not in values and k in old and np.shape(old[k]) == np.shape(new[k]):
            new[k] = old[k]
    sp[i] = new
    prob = dataclasses.replace(problem, system=dataclasses.replace(problem.system,
                                                                   surfaces=tuple(S)))
    return prob, dict(params, s=sp)


# --------------------------------------------------------------------------- morph

def morph_element(z, cfg, problem: EM.Problem):
    """Homotopy of one element (problem.morphing[0] = (front, rear)) from an ideal phase
    screen on its rear surface (s = 0) into a glass singlet (s = 1); see
    continuation.morph_to_params.  Everything else is taken from z unchanged."""
    s = cfg["s"]
    f, r = problem.morphing[0]
    m = z["m"]
    rn = problem.system.surfaces[r].r_norm
    dc = s * m["phi"] / (m["nd"] - 1.0)
    X = X_MAX * jnp.tanh(m["xi"])
    surfs = [dict(p) for p in z["s"]]
    surfs[f] = dict(surfs[f], c=0.5 * (1.0 + X) * dc + s * m["dc1"], t=m["t"], nd=m["nd"],
                    vd=m["vd"])
    a1 = -(1.0 - s) * m["phi"] * rn**2 / 2.0
    surfs[r] = dict(surfs[r], c=0.5 * (X - 1.0) * dc + s * m["dc2"],
                    pcoef=jnp.concatenate([a1[None], (1.0 - s) * m["bhi"]]))
    return {"s": surfs, "mc": z["mc"]}


def _morph(problem, params, spec, steps=20, iters=30, mu_b=1e-5, jump=1.5, polish_iters=80,
           log=None):
    """Morph problem.morphing[0] into glass (adaptive homotopy steps, as in MATERIALIZE)."""
    f, r = problem.morphing[0]
    rn = problem.system.surfaces[r].r_norm
    pr = params["s"][r]
    z = {"s": params["s"], "mc": params["mc"],
         "m": {"phi": -2.0 * pr["pcoef"][0] / rn**2, "xi": jnp.asarray(0.0),
               "dc1": jnp.asarray(0.0), "dc2": jnp.asarray(0.0), "t": params["s"][f]["t"],
               "bhi": pr["pcoef"][1:], "nd": params["s"][f]["nd"], "vd": params["s"][f]["vd"]}}
    free = variables(problem)
    free["s"][f] = {}
    free["s"][r] = {"t": True}
    free["m"] = {k: True for k in z["m"]}

    def step(z, s):
        mu = mu_b * s / max(1.0 - s, 1e-3) ** 2 if s < 1 else 0.0
        z, _ = optimize_design(problem, z, spec, free, iters=iters, transform=morph_element,
                               s=s, mu=mu)
        p = morph_element(z, EM.make_cfg(s=s), problem)
        return z, p, mean_wfe(problem, p, spec)

    ds0 = 1.0 / steps
    s_prev, ds, prev = 0.0, ds0, None
    while s_prev < 1.0 - 1e-12:
        s = min(1.0, s_prev + ds)
        z_new, p, w = step(z, s)
        if (not np.isfinite(w) or (prev is not None and w > jump * prev + 0.02)) and \
                ds > ds0 / 8 + 1e-12:
            ds /= 2
            continue
        z, s_prev, prev = z_new, s, w
        ds = min(ds0, 2.0 * ds)
        if log:
            log(f"      morph s={s:.2f}: mean RMS WFE {w:.3f} waves")
    p = morph_element(z, EM.make_cfg(s=1.0), problem)
    # the rear surface is plain glass now: drop its phase
    s_r = dataclasses.replace(problem.system.surfaces[r], phase="", n_phase=0)
    prob, p = replace_surface(problem, p, r, s_r, pcoef=np.zeros(0))
    prob = dataclasses.replace(prob, morphing=())
    p, _ = optimize_design(prob, p, spec, iters=polish_iters)
    return prob, p


def insert_screen(problem: EM.Problem, params, i, r_norm, glass=(1.62, 50.0), n_phase=4,
                  front_gap=4.0):
    """Insert a zero-power ideal phase screen on the rear face of a zero-thickness flat
    slab of model glass, in the middle of the air space after surface i (i = -1: in
    front of the first surface, `front_gap` ahead of it).  The system is unchanged;
    the new pair is marked as morphing (problem.morphing)."""
    front = Surface(material="model", label="new front")
    rear = Surface(phase="ideal", n_phase=n_phase, r_norm=r_norm, label="new rear")
    p = dict(params, s=[dict(q) for q in params["s"]])
    if i >= 0:
        t = float(params["s"][i]["t"])
        p["s"][i] = dict(p["s"][i], t=jnp.asarray(t / 2))
        gap = t / 2
    else:
        gap = front_gap
    prob, p = insert_surfaces(problem, p, i + 1, [front, rear],
                              [dict(t=0.0, nd=glass[0], vd=glass[1]),
                               dict(t=gap, pcoef=np.zeros(n_phase))])
    return dataclasses.replace(prob, morphing=((i + 1, i + 2),)), p


def add_element(problem: EM.Problem, params, spec: Spec, positions=None, glass=(1.62, 50.0),
                n_phase=4, min_space=2.0, front_gap=4.0, snap=True, steps=20, log=None):
    """Add one air-spaced singlet; returns (problem, params, report).

    Every air space at least `min_space` thick, and the space in front of the first
    surface (position -1, object at infinity), gets a zero-power ideal phase screen in
    a zero-thickness slab of model glass: exactly the old system.  `positions` (surface
    indices whose following space is used, -1 = in front) restricts the candidates.
    Each candidate is optimized in phase space, morphed into glass, polished and
    (snap=True) given catalog glasses; the best real merit wins.
    """
    S = problem.system.surfaces
    over = {(i, k) for c in problem.configs for i, k in c.overrides}
    if positions is None:
        positions = [-1] + [i for i in range(len(S) - 1)
                            if S[i].material in AIR and S[i].shape != "cb"
                            and (i, "t") not in over
                            and float(params["s"][i]["t"]) >= min_space]
    sd = clear_semi_diameters(problem, params, spec)

    def candidate(i):
        rn = float(max(sd[max(i, 0)], sd[min(i + 1, len(S) - 1)], 1.0))
        prob, p = insert_screen(problem, params, i, rn, glass, n_phase, front_gap)
        f, r = i + 1, i + 2
        # phase space: the screen takes power and aspheric phase, the rest re-balances
        free = variables(prob)
        free["s"][f] = {}
        free["s"][r] = {"t": True, "pcoef": True}
        p, _ = optimize_design(prob, p, spec, free, iters=80, s=0.0)
        phase_wfe = mean_wfe(prob, p, spec)
        prob, p = _morph(prob, p, spec, steps=steps)
        glass_wfe = mean_wfe(prob, p, spec)
        if snap:
            prob, p = snap_glasses(prob, p, spec)
        c = cost(prob, p, spec)
        return {"after": i, "cost": c if np.isfinite(c) else np.inf, "problem": prob,
                "params": p, "phase_wfe": phase_wfe, "glass_wfe": glass_wfe,
                "wfe": mean_wfe(prob, p, spec)}

    results = pmap(candidate, positions)
    for r in results:
        if log:
            where = "in front" if r["after"] < 0 else f"after surface {r['after']}"
            log(f"   new element {where}: phase {r['phase_wfe']:.3f} -> "
                f"glass {r['glass_wfe']:.3f} -> final {r['wfe']:.3f} waves")
    best = min(results, key=lambda r: r["cost"])
    return best["problem"], best["params"], results


def insert_cement_layer(problem: EM.Problem, params, rear, glass=(1.72, 30.0)):
    """Insert, in front of `rear` (the rear surface of a glass element), a surface of the
    same shape with a zero-thickness layer of model glass behind it: the system is
    unchanged.  The layer is marked as morphing, so its minimum thickness can be ramped."""
    S = problem.system.surfaces
    if S[rear - 1].material in AIR or S[rear].interact != "refract":
        raise ValueError(f"surface {rear} is not the rear surface of a glass element")
    mid = dataclasses.replace(S[rear], material="model", label="cemented", stop=False,
                              phase="", n_phase=0)
    vals = {k: params["s"][rear][k] for k in ("c", "k", "coef")}
    prob, p = insert_surfaces(problem, params, rear, [mid],
                              [dict(vals, t=0.0, nd=glass[0], vd=glass[1])])
    return dataclasses.replace(prob, morphing=((rear, rear + 1),)), p


def cement(problem: EM.Problem, params, spec: Spec, rear, glass=(1.72, 30.0), steps=4,
           iters=60, snap=True):
    """Cement a new glass behind the element whose rear surface is `rear`.

    A surface conformal with `rear` (same shape) is inserted in front of it with a
    zero-thickness layer of model glass: exactly the old system.  The layer is then
    grown (its minimum thickness ramped in over `steps`) with everything free."""
    prob, p = insert_cement_layer(problem, params, rear, glass)
    for s in np.linspace(0, 1, steps + 1)[1:]:
        p, _ = optimize_design(prob, p, spec, iters=iters, s=float(s))
    prob = dataclasses.replace(prob, morphing=())
    p, _ = optimize_design(prob, p, spec, iters=80)
    if snap:
        prob, p = snap_glasses(prob, p, spec)
    return prob, p


# --------------------------------------------------------------------------- glasses

_PREFERRED = None


def preferred_glasses():
    """(name, n_d, V_d) of the preferred glasses of the default catalog."""
    global _PREFERRED
    if _PREFERRED is None:
        from ..catalog import read_agf
        from .materials import DEFAULT_AGF

        gl = [g for g in read_agf(DEFAULT_AGF) if g.status == 1
              and "augmented reality" not in g.comment.lower()]
        _PREFERRED = [(g.name, g.nd, g.vd) for g in gl]
    return _PREFERRED


def nearest_glasses(nd, vd, k=3):
    cat = preferred_glasses()
    d = [((n - nd) / 0.1) ** 2 + ((v - vd) / 10.0) ** 2 for _, n, v in cat]
    return [cat[j][0] for j in np.argsort(d)[:k]]


def snap_glasses(problem: EM.Problem, params, spec: Spec, k=3, iters=60):
    """Replace every model glass by a catalog glass: the `k` nearest preferred glasses
    per surface in the (n_d/0.1, V_d/10) glass map, every combination (up to 27, else
    nearest plus one alternative at a time) re-optimized; the best merit wins."""
    S = problem.system.surfaces
    model = [i for i, s in enumerate(S) if s.material == "model"]
    if not model:
        return problem, params
    opts = [nearest_glasses(float(params["s"][i]["nd"]), float(params["s"][i]["vd"]), k)
            for i in model]
    combos = list(itertools.product(*opts))
    if len(combos) > 27:
        near = tuple(o[0] for o in opts)
        combos = [near] + [near[:j] + (g,) + near[j + 1:] for j in range(len(model))
                           for g in opts[j][1:]]

    def run(combo):
        prob, p = problem, params
        for i, name in zip(model, combo):
            prob, p = replace_surface(prob, p, i, dataclasses.replace(S[i], material=name))
        p, info = optimize_design(prob, p, spec, iters=iters)
        c = info.get("cost", np.inf)
        return (c if np.isfinite(c) else np.inf), prob, p

    best = min(pmap(run, combos), key=lambda r: r[0])
    return best[1], best[2]


# --------------------------------------------------------------------------- configurations

def per_wavelength_focus(problem: EM.Problem, params, spec: Spec, weights=None):
    """One configuration per wavelength, each evaluated alone in its own image plane:
    the distance to the image surface becomes a variable of each configuration.  For
    lenses with intentional axial color (e.g. spectrum from defocus).  The focal-length
    target applies to the reference wavelength only (the focal length differs by design
    at the others)."""
    S = problem.system.surfaces
    last = len(S) - 2  # the space in front of the image surface
    wls = spec.wavelengths
    weights = weights or (1.0,) * len(wls)
    configs = tuple(EM.Config(name=f"{wl * 1e3:.0f} nm", overrides=((last, "t"),),
                              wavelengths=(wl,), wl_weights=(1.0,), weight=w,
                              efl=None if j == spec.iref else 0.0)
                    for j, (wl, w) in enumerate(zip(wls, weights)))
    prob = dataclasses.replace(problem, configs=configs)
    return prob, EM.make_params(params["s"], prob)


def dense_report(problem: EM.Problem, params, spec: Spec, n_field=21, n_rings=8, n_arms=16):
    """RMS wavefront error on a dense field grid (0 .. the largest field, along the
    direction of the largest field) and a denser pupil, per configuration."""
    fmax = max(spec.fields, key=lambda f: np.hypot(*f))
    r = np.hypot(*fmax)
    u = np.asarray(fmax) / r if r > 0 else np.array([0.0, 1.0])
    fields = tuple(tuple(float(v) for v in u * a) for a in np.linspace(0, r, n_field))
    configs = tuple(dataclasses.replace(c, fields=fields) for c in problem.configs)
    prob = dataclasses.replace(problem, configs=configs,
                               targets=dataclasses.replace(problem.targets, n_rings=n_rings,
                                                           n_arms=n_arms))
    per = EM.rms_wfe(params, prob, spec)
    allw = np.concatenate(per)
    return {"per_config": per, "mean_wfe": float(allw.mean()), "max_wfe": float(allw.max()),
            "field_deg": np.linspace(0, r, n_field)}
