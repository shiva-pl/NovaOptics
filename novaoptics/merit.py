"""Merit function as a residual vector (for damped least squares).

All the knobs that change along a continuation path live in a `cfg` dict of
floats, passed to the jitted residual as data, so moving along the path never
triggers recompilation:

    s        homotopy parameter of a reparameterized morph (0 = phase, 1 = glass)
    beta     dispersion switch (0 = dispersion-free, 1 = real glass)
    mu       weight pulling ideal phase W toward zero (phase -> glass transfer)
    mu_asph  weight on aspheric surface coefficients (in waves of sag x (n-1))
    w_spot   weight of transverse ray errors, scaled by 8*N*lambda (a defocus spot then
             counts about as much as its wavefront error).  Wavefront samples alone
             let high-order aspheres hide steep walls between pupil rings.
    nu       weight pulling (n_d, V_d) toward the nearest catalog glass
    tmin     minimum center thickness of elements      [mm]
    tmax     maximum center thickness of elements      [mm]
    emin     minimum edge thickness of elements        [mm]
    gmin     minimum center/edge air space between elements [mm]
    w_content  weight of per-element Seidel spherical aberration, coma and astigmatism
             (d-line waves).  Pushes toward elements that are corrected on their own
             instead of large contributions cancelling between elements.
    virtual_petzval  1 = account for the Petzval curvature that the ideal phase
             will acquire once it becomes glass.  A screen of power phi destined for
             glass of index n_d will add -phi/n_d of image-surface curvature, so the
             image is evaluated on a surface pre-curved by +sum(phi_i/n_d_i).  The
             correction fades out by itself as power migrates from phase to glass,
             whose Petzval curvature then comes from the real ray trace.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import glass as G
from .analysis import clear_semi_diameters, distortion, evaluate, paraxial_efl
from .system import Layout, Spec
from .trace import asphere_poly

DEFAULT_CFG = dict(
    s=0.0,
    beta=1.0,
    mu=0.0,
    mu_asph=0.0,
    nu=0.0,
    tmin=0.0,
    emin=0.0,
    tmax=1e3,
    gmin=0.0,
    w_spot=1.0,
    w_efl=300.0,
    w_dist=50.0,
    w_constraint=10.0,
    virtual_petzval=0.0,
    w_content=0.0,
    bfd_min=5.0,
)


def region_violation(params, layout):
    """Hinges keeping (n_d, V_d) glasses where real glasses exist.

    With `layout.region` (a GlassSpace) this is the convex hull of that catalog in the
    glass map; otherwise the built-in approximate outline (`glass.glass_region_violation`).
    """
    if layout.region is not None:
        # Convex hull of the catalog in the glass map.  It slightly over-includes (the
        # populated map is not convex), but it is smooth; the non-convex per-bin
        # envelope (GlassSpace.env_*) traps the continuous optimization at its kinks.
        hv = jnp.stack([G.ndvd_hull_violation(params["nd"][i], params["vd"][i], layout.region)
                        for i in range(layout.n_elem)])
        return (10.0 * _hinge(hv),)
    return tuple(v / 5 for v in G.glass_region_violation(params["nd"], params["vd"]))


def gs_radius(gs):
    """Feasibility radius of a glass space (normalized PCA units).

    A radius of one median neighbour spacing turns the feasible set into a scatter of
    small balls around individual glasses and traps the continuous optimization;
    the default lets continuous glasses interpolate between neighbouring catalog glasses.
    """
    from .catalog import FEAS_FACTOR

    if gs.feas_radius is not None:
        return gs.feas_radius
    if not hasattr(gs, "_radius"):
        c = gs.coords / gs.scale
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        gs._radius = FEAS_FACTOR * float(np.median(d.min(1)))
    return gs._radius


def make_cfg(**kw):
    cfg = dict(DEFAULT_CFG)
    cfg.update(kw)
    return {k: jnp.asarray(v, dtype=float) for k, v in cfg.items()}


def _hinge(x):
    return jnp.maximum(x, 0.0)


def sag(c, r):
    cr2 = jnp.minimum((c * r) ** 2, 0.98)
    return c * r * r / (1.0 + jnp.sqrt(1.0 - cr2))


def rear_sag(params, layout: Layout, i, r):
    out = sag(params["c2"][i], r)
    if layout.aspheres:
        out = out + asphere_poly(params["asph"][i], r * r, layout.r_norm)[0]
    return out


def virtual_petzval_curvature(params, layout: Layout):
    phi_phase = -2.0 * params["phase"][:, 0] / layout.r_norm**2
    return jnp.sum(phi_phase / G.nd_vec(params, layout))


def residuals(params, layout: Layout, spec: Spec, cfg):
    params = G.resolve(params, layout)  # n_d, V_d derived from PCA glass coordinates
    c_img = cfg["virtual_petzval"] * virtual_petzval_curvature(params, layout)
    ev = evaluate(params, layout, spec, cfg["beta"], c_img=c_img)

    # --- image quality: weighted wavefront samples (sum of squares = mean RMS^2) ---
    fw = jnp.asarray(spec.field_weights) / np.sum(spec.field_weights)
    lw = jnp.asarray(spec.wl_weights) / np.sum(spec.wl_weights)
    w = jnp.sqrt(lw[:, None, None] * fw[None, :, None] * ev["wt"][None, None, :])
    r_wfe = (w * ev["opd"]).ravel()
    # transverse ray errors relative to the reference-wavelength chief ray
    scale = 8.0 * spec.fnum * jnp.asarray(spec.wavelengths)[:, None, None, None] * 1e-3
    eps = (ev["img"][..., :2] - ev["chief_img"][None, :, None, :2]) / scale
    r_spot = (jnp.sqrt(cfg["w_spot"] / 2.0) * w[..., None] * eps).ravel()

    # --- first order ---
    efl = paraxial_efl(params, layout, spec, cfg["beta"])
    r_efl = cfg["w_efl"] * (efl - spec.efl) / spec.efl
    dist = distortion(ev, efl)
    r_dist = cfg["w_dist"] * _hinge(jnp.abs(dist) - spec.max_distortion)

    # --- mechanical constraints ---
    wc = cfg["w_constraint"]
    sd = clear_semi_diameters(ev)  # [2 * n_elem]
    stop_sd = jnp.sqrt(jnp.max(jnp.sum(ev["stop_xy"] ** 2, -1)) + 1e-12)
    cons = []
    t = params["t"]
    c1, c2 = params["c1"], params["c2"]
    for i in range(layout.n_elem):
        sd_e = jnp.maximum(sd[2 * i], sd[2 * i + 1])
        edge = t[i] + rear_sag(params, layout, i, sd_e) - sag(c1[i], sd_e)
        cons += [_hinge(cfg["tmin"] - t[i]), _hinge(cfg["emin"] - edge), _hinge(-t[i]),
                 _hinge(t[i] - cfg["tmax"]), _hinge(edge - cfg["tmax"])]
        cons += [_hinge(jnp.abs(c1[i]) * sd_e - 0.8), _hinge(jnp.abs(c2[i]) * sd_e - 0.8)]

    items = layout.items()
    gaps = params["gaps"]
    cons += [_hinge(-gaps)]
    # air spaces between consecutive elements (possibly through the stop)
    elem_pos = [k for k, it in enumerate(items) if it != "stop"]
    for a, b in zip(elem_pos[:-1], elem_pos[1:]):
        i, j = items[a], items[b]
        g = jnp.sum(gaps[a:b])
        sd_ij = jnp.maximum(sd[2 * i + 1], sd[2 * j])
        edge_air = g + sag(c1[j], sd_ij) - rear_sag(params, layout, i, sd_ij)
        cons += [_hinge(cfg["gmin"] - g), _hinge(cfg["gmin"] - edge_air)]
    # surfaces must not cross the stop plane
    ks = items.index("stop")
    if ks > 0:
        i = items[ks - 1]
        cons += [_hinge(rear_sag(params, layout, i, stop_sd) - gaps[ks - 1])]
    if ks < len(items) - 1:
        j = items[ks + 1]
        cons += [_hinge(-sag(c1[j], stop_sd) - gaps[ks])]
    track = jnp.sum(t) + jnp.sum(gaps) + params["bfd"]
    cons += [_hinge(track - spec.max_track), _hinge(cfg["bfd_min"] - params["bfd"])]
    r_cons = wc * jnp.concatenate([jnp.atleast_1d(c).ravel() for c in cons])

    # --- continuation penalties ---
    lam_mm = spec.ref_wavelength * 1e-3
    # only aspheric (k >= 2) phase is penalized; the quadratic term is optical power
    r_phase = jnp.sqrt(cfg["mu"]) * (params["phase"][:, 1:] / lam_mm).ravel()
    r_asph = jnp.sqrt(cfg["mu_asph"]) * (
        params["asph"] * (params["nd"][:, None] - 1.0) / lam_mm).ravel()

    # aberration content: per-element W040, W131, W222 (Seidel, d-line waves)
    if layout.n_elem > 0:
        from .diagnostics import element_waves

        ew = element_waves(params, layout, spec, cfg["beta"])[:, :3]
        r_content = jnp.sqrt(cfg["w_content"]) * ew.ravel()
    if layout.glass is None:
        pts = None if layout.region is None else (layout.region.nd, layout.region.vd)
        gdist = jnp.stack([G.catalog_softmin_distance(params["nd"][i], params["vd"][i],
                                                      points=pts)
                           for i in range(layout.n_elem)])
        r_gbounds = wc * jnp.concatenate(
            [
                _hinge(G.ND_BOUNDS[0] - params["nd"]) * 10,
                _hinge(params["nd"] - G.ND_BOUNDS[1]) * 10,
                _hinge(G.VD_BOUNDS[0] - params["vd"]) / 10,
                _hinge(params["vd"] - G.VD_BOUNDS[1]) / 10,
                *(region_violation(params, layout)),
            ]
        )
    else:
        # PCA glass space: stay within the catalog's footprint.  Distances are in
        # normalized PCA units; beyond the feasibility radius the penalty acts.
        gs = layout.glass
        rad = gs_radius(gs)
        gdist = jnp.stack([G.space_distance(params["g"][i], gs, temperature=0.25 * rad)
                           for i in range(layout.n_elem)])
        if gs.feas_radius is not None:
            r_gbounds = wc * _hinge(gdist - rad)
        else:
            # The pipeline never frees PCA coordinates (continuous glasses live in
            # (n_d, V_d)); snapped glasses are real, so no feasibility term is needed.
            r_gbounds = jnp.zeros(0)
    r_glass = jnp.sqrt(cfg["nu"]) * gdist

    return jnp.concatenate(
        [r_wfe, r_spot, jnp.atleast_1d(r_efl), r_dist, r_cons, r_phase, r_asph, r_content, r_glass,
         r_gbounds]
    )


