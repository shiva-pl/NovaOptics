"""Merit function and multi-configuration for the surface engine.

A design is a `Problem` (hashable, static): the `System`, its configurations and the
merit targets, plus a parameter tree

    params = {"s": [surface dicts], "mc": [per-configuration dicts]}

A `Config` overrides surface parameters with its own values (variables of that
configuration): zoom spacings, a focus distance, or a separate image distance per
wavelength for a lens with intentional axial color.  It can also evaluate its own
wavelengths and fields and have its own focal-length target.  The merit is the
concatenation of every configuration's residuals:

* image quality: wavefront samples (Gaussian quadrature over the pupil, piston
  removed per field and wavelength, tilt kept so lateral color counts) and transverse
  ray errors about the reference-wavelength chief ray, as in the element model;
* first order: focal length target, distortion bound;
* mechanics: center / edge thickness of glass, center / edge air spaces, total track,
  back focal distance (coaxial refracting runs only);
* model glasses (n_d, V_d) kept inside the glass map;
* continuation penalties set through `cfg` (as in the element model): `mu` on the
  aspheric phase of ideal-phase surfaces.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .. import glass as G
from .system import Spec, System
from .trace import bundle, focal_length

# mechanical defaults as in the element model (continuation.Mech)
DEFAULT_CFG = dict(mu=0.0, s=1.0, w_spot=1.0, w_efl=300.0, w_dist=50.0, w_constraint=10.0,
                   tmin=2.0, tmax=8.0, emin=1.0, gmin=1.0, bfd_min=5.0, w_glass=1.0,
                   virtual_petzval=0.0)

FUTURE_GLASS_ND = 1.62  # index assumed for a bare screen in air (no slab yet)


def make_cfg(**kw):
    c = dict(DEFAULT_CFG)
    c.update(kw)
    return {k: jnp.asarray(v, float) for k, v in c.items()}


@dataclass(frozen=True)
class Config:
    """One configuration.

    overrides: ((surface index, parameter name), ...) whose values come from this
    configuration's entry in params["mc"] (named "<surface>:<parameter>").
    wavelengths / fields: subsets evaluated in this configuration (None: the Spec's).
    efl: focal-length target (None: Targets.efl).
    """

    name: str = ""
    overrides: tuple = ()
    wavelengths: tuple = None
    wl_weights: tuple = None
    fields: tuple = None
    efl: float = None
    weight: float = 1.0


@dataclass(frozen=True)
class Targets:
    efl: float = None
    max_distortion: float = None
    max_track: float = None
    n_rings: int = 6
    n_arms: int = 12
    region: object = None  # catalog.GlassSpace bounding model glasses (glass map hull)


@dataclass(frozen=True)
class Problem:
    system: System
    configs: tuple = (Config(),)
    targets: Targets = field(default_factory=Targets)
    # (front, rear) surface pairs of elements being morphed from phase into glass: their
    # thickness and edge constraints ramp with the homotopy parameter cfg["s"]
    morphing: tuple = ()


def key_name(i, k):
    return f"{i}:{k}"


def make_params(surfaces, problem: Problem, values=None):
    """Parameter tree; configuration values default to the base surface values.
    values: optional {(config index, surface, key): value}."""
    mc = []
    for c, conf in enumerate(problem.configs):
        d = {}
        for i, k in conf.overrides:
            v = (values or {}).get((c, i, k), surfaces[i][k])
            d[key_name(i, k)] = jnp.asarray(v, float)
        mc.append(d)
    return {"s": list(surfaces), "mc": mc}


def config_surfaces(params, problem: Problem, c):
    """Surface dicts of configuration c (overrides applied)."""
    out = [dict(p) for p in params["s"]]
    for i, k in problem.configs[c].overrides:
        out[i][k] = params["mc"][c][key_name(i, k)]
    return out


def config_spec(spec: Spec, conf: Config):
    kw = {}
    if conf.wavelengths is not None:
        kw["wavelengths"] = tuple(conf.wavelengths)
        kw["wl_weights"] = tuple(conf.wl_weights or (1.0,) * len(conf.wavelengths))
        kw["ref"] = None
    if conf.fields is not None:
        kw["fields"] = tuple(conf.fields)
        kw["field_weights"] = None
    return dataclasses.replace(spec, **kw) if kw else spec


def pupil_quadrature(n_rings, n_arms):
    u, w = np.polynomial.legendre.leggauss(n_rings)
    u, w = (u + 1) / 2, w / 2
    r = np.sqrt(u)
    th = (np.arange(n_arms) + 0.5) * 2 * np.pi / n_arms
    px = (r[:, None] * np.cos(th)[None, :]).ravel()
    py = (r[:, None] * np.sin(th)[None, :]).ravel()
    wt = (w[:, None] * np.ones(n_arms)[None, :] / n_arms).ravel()
    return px, py, wt


def _hinge(x):
    return jnp.maximum(x, 0.0)


def evaluate_config(surfs, system: System, spec: Spec, targets: Targets):
    """Wavefronts and ray data of one configuration: dict of arrays [F, W, P, ...]."""
    px, py, wt = (jnp.asarray(a) for a in pupil_quadrature(targets.n_rings, targets.n_arms))
    fields = jnp.asarray(spec.fields, float)
    wls = jnp.asarray(spec.wavelengths, float)
    ref = spec.ref_wavelength

    def per_field(f):
        def per_wl(wl):
            b = bundle(surfs, system, spec, f, wl, px, py, ref_wl=ref)
            return b["opd"], b["img"], b["chief_ref_img"], b["local"], b["chief_img"]

        return jax.vmap(per_wl)(wls)

    opd, img, ref_img, local, chief = jax.vmap(per_field)(fields)
    return {"opd": opd, "img": img, "ref_img": ref_img[:, 0], "local": local,
            "chief": chief, "wt": wt}


def _runs(system: System):
    """Coaxial runs of consecutive interacting surfaces (for thickness constraints):
    pairs (i, j, between) where j follows i with only non-interacting, non-break
    surfaces `between`; broken at coordinate breaks and mirrors."""
    S = system.surfaces
    act = [k for k, s in enumerate(S) if s.interact == "refract"]
    pairs = []
    for a, b in zip(act[:-1], act[1:]):
        mid = list(range(a + 1, b))
        if any(S[k].shape == "cb" or S[k].interact == "reflect" for k in mid):
            continue
        pairs.append((a, b, tuple(mid)))
    return pairs


def virtual_petzval(surfs, system: System, weight):
    """Image surface pre-curved by the Petzval curvature the ideal phase screens will
    acquire as glass: sum(phi_i / n_i) (see merit.py of the element model).  A phase
    screen has no Petzval curvature of its own, so without this a phase system is
    judged against the wrong field curvature and screens look far worse (or better)
    than the glass they stand for."""
    from . import materials as M

    c = 0.0
    S = system.surfaces
    for i, s in enumerate(S):
        if s.phase in ("ideal", "glass") and s.n_phase:
            phi = -2.0 * surfs[i]["pcoef"][0] / s.r_norm**2
            prev = S[i - 1].material if i > 0 else "air"
            n = FUTURE_GLASS_ND if prev in ("air", "", "vacuum") else \
                M.index(prev, 0.5875618, surfs[i - 1])
            c = c + phi / n
    out = list(surfs)
    out[-1] = dict(out[-1], c=out[-1]["c"] + weight * c)
    return out


def config_residuals(surfs, problem: Problem, spec: Spec, conf: Config, cfg):
    from .trace import sag

    system, targets = problem.system, problem.targets
    surfs = virtual_petzval(surfs, system, cfg["virtual_petzval"])
    ev = evaluate_config(surfs, system, spec, targets)
    F, W = len(spec.fields), len(spec.wavelengths)
    fw = jnp.asarray(spec.fweights) / np.sum(spec.fweights)
    lw = jnp.asarray(spec.wl_weights) / np.sum(spec.wl_weights)
    wl_mm = jnp.asarray(spec.wavelengths)[None, :, None] * 1e-3
    w = jnp.sqrt(conf.weight * fw[:, None, None] * lw[None, :, None] * ev["wt"][None, None, :])
    opd = ev["opd"] / wl_mm  # waves
    opd = opd - jnp.sum(opd * ev["wt"], -1, keepdims=True)  # piston
    r_wfe = (w * opd).ravel()
    efl_t = conf.efl if conf.efl is not None else targets.efl
    fno = (efl_t / spec.epd) if efl_t else 4.0
    scale = 8.0 * fno * wl_mm[..., None]
    eps = (ev["img"] - ev["ref_img"][:, None, None, :]) / scale
    r_spot = (jnp.sqrt(cfg["w_spot"] / 2.0) * w[..., None] * eps).ravel()
    out = [r_wfe, r_spot]

    if efl_t:
        efl = focal_length(surfs, system, spec)
        out.append(jnp.atleast_1d(cfg["w_efl"] * (efl - efl_t) / efl_t))
        if targets.max_distortion is not None:
            th = jnp.deg2rad(jnp.linalg.norm(jnp.asarray(spec.fields, float), axis=-1))
            h = jnp.linalg.norm(ev["chief"][:, spec.iref], axis=-1)
            ideal = efl * jnp.tan(th)
            d = jnp.where(ideal > 1e-9, (h - ideal) / jnp.where(ideal > 1e-9, ideal, 1.0), 0.0)
            out.append(cfg["w_dist"] * _hinge(jnp.abs(d) - targets.max_distortion))

    # mechanics: clear semi-diameters from all rays of this configuration
    sd = jnp.sqrt(jnp.max(jnp.sum(ev["local"][..., :2] ** 2, -1), axis=(0, 1, 2)) + 1e-12)
    cons = []
    S = system.surfaces
    for a, b, mid in _runs(system):
        t = surfs[a]["t"] + sum((surfs[k]["t"] for k in mid), 0.0)
        h = jnp.maximum(sd[a], sd[b])
        edge = t + sag(S[b], surfs[b], 0.0, h) - sag(S[a], surfs[a], 0.0, h)
        if S[a].material not in ("air", "", "vacuum"):
            ramp = cfg["s"] if any(a == f for f, _ in problem.morphing) else 1.0
            cons += [_hinge(ramp * cfg["tmin"] - t), _hinge(t - cfg["tmax"]),
                     _hinge(ramp * cfg["emin"] - edge)]
        else:
            cons += [_hinge(cfg["gmin"] - t), _hinge(cfg["gmin"] - edge)]
            # a stop / dummy surface in this air space must stay in air: neither lens
            # surface may cross its plane within its semi-diameter
            d1 = surfs[a]["t"]
            for k in mid:
                d2 = t - d1
                cons += [_hinge(sag(S[a], surfs[a], 0.0, sd[k]) - d1),
                         _hinge(-sag(S[b], surfs[b], 0.0, sd[k]) - d2)]
                d1 = d1 + surfs[k]["t"]
        for k in (a, b):  # no hemispheres
            cons.append(_hinge(jnp.abs(surfs[k]["c"]) * sd[k] - 0.8))
    last = max(k for k, s in enumerate(S) if s.interact == "refract" or s.interact == "reflect")
    bfd = sum((surfs[k]["t"] for k in range(last, len(S) - 1)), 0.0)
    cons.append(_hinge(cfg["bfd_min"] - jnp.abs(bfd)))
    if targets.max_track is not None:
        track = sum((surfs[k]["t"] for k in range(len(S) - 1)), 0.0)
        cons.append(_hinge(track - targets.max_track))
    out.append(cfg["w_constraint"] * jnp.stack([jnp.asarray(c).ravel()[0] for c in cons]))

    # model glasses inside the glass map; aspheric ideal phase penalty
    for k, s in enumerate(S):
        p = surfs[k]
        if s.material == "model":
            nd, vd = p["nd"], p["vd"]
            g = [_hinge(G.ND_BOUNDS[0] - nd) * 10, _hinge(nd - G.ND_BOUNDS[1]) * 10,
                 _hinge(G.VD_BOUNDS[0] - vd) / 10, _hinge(vd - G.VD_BOUNDS[1]) / 10]
            if targets.region is not None:
                g.append(10.0 * _hinge(G.ndvd_hull_violation(nd, vd, targets.region)))
            out.append(cfg["w_glass"] * cfg["w_constraint"] * jnp.stack(g))
        if s.phase in ("ideal", "glass") and s.n_phase > 1:
            lam = spec.ref_wavelength * 1e-3
            out.append(jnp.sqrt(cfg["mu"]) * p["pcoef"][1:] / lam)
    del F, W
    return jnp.concatenate([jnp.atleast_1d(o).ravel() for o in out])


def residuals(params, problem: Problem, spec: Spec, cfg):
    """Residual vector of all configurations (the `residual_fn` for optimize())."""
    out = []
    for c, conf in enumerate(problem.configs):
        surfs = config_surfaces(params, problem, c)
        out.append(config_residuals(surfs, problem, config_spec(spec, conf), conf, cfg))
    return jnp.concatenate(out)


_JIT: OrderedDict = OrderedDict()  # (name, problem, spec) -> compiled helper, LRU
_JIT_LOCK = threading.Lock()


def compiled(name, problem, spec, fn):
    """fn(params) compiled once per (name, problem, spec).  Eager evaluation would
    compile every primitive separately into JAX's own caches, which grow without bound
    (each program holds memory mappings; a long design session ran out of them)."""
    from ..optimize import memory_pressure, register_cache, release_memory

    key = (name, problem, spec)
    with _JIT_LOCK:
        if key in _JIT:
            _JIT.move_to_end(key)
            return _JIT[key]
        if not getattr(compiled, "registered", False):
            register_cache(_JIT.clear)
            compiled.registered = True
    if memory_pressure():
        release_memory()
    f = jax.jit(fn)
    with _JIT_LOCK:
        _JIT[key] = f
        while len(_JIT) > 24:
            _JIT.popitem(last=False)
    return f


def rms_wfe(params, problem: Problem, spec: Spec, virtual_petzval_weight=0.0):
    """Polychromatic RMS wavefront error [waves] per configuration and field (with
    virtual_petzval_weight = 1: on the image surface the phase screens' glass will
    curve, as the pipeline judges phase systems)."""
    vp = float(virtual_petzval_weight)
    f = compiled(f"rms_wfe {vp}", problem, spec, lambda p: _rms_wfe(p, problem, spec, vp))
    return [np.asarray(a) for a in f(params)]


def _rms_wfe(params, problem: Problem, spec: Spec, vp=0.0):
    res = []
    for c, conf in enumerate(problem.configs):
        sp = config_spec(spec, conf)
        surfs = virtual_petzval(config_surfaces(params, problem, c), problem.system, vp)
        ev = evaluate_config(surfs, problem.system, sp, problem.targets)
        wl_mm = jnp.asarray(sp.wavelengths)[None, :, None] * 1e-3
        opd = ev["opd"] / wl_mm
        opd = opd - jnp.sum(opd * ev["wt"], -1, keepdims=True)
        lw = jnp.asarray(sp.wl_weights) / np.sum(sp.wl_weights)
        ms = jnp.sum(opd**2 * ev["wt"], -1)  # [F, W]
        res.append(jnp.sqrt(jnp.sum(ms * lw[None, :], -1)))
    return res
