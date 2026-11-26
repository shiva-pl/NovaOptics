"""Image-quality analysis: wavefront error, first-order properties, and the
"wavefront-carrying ray" diagnostics obtained from forward-mode autodiff.

Wavefront error is computed exactly from real rays: optical path to a reference
sphere centered on the reference-wavelength chief-ray image point and passing
through the chief ray at the last surface.  Piston is removed per wavelength;
tilt is *not* removed, so lateral color and per-wavelength image shifts count.

Generalized (differential) ray tracing comes for free from JAX: the Jacobian of
a ray's image-space state with respect to its pupil coordinates is exactly what
classic Coddington / generalized ray traces propagate by hand.  `focus_shifts`
uses it to get tangential and sagittal focal positions for every field point
from a single ray each.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .system import Layout, Spec
from .trace import aim_chief, launch, trace_ray


def pupil_grid(n_rings=4, n_arms=12):
    """Gaussian quadrature over the unit disk: rings at Gauss-Legendre nodes in r^2."""
    u, w = np.polynomial.legendre.leggauss(n_rings)
    u = (u + 1) / 2
    w = w / 2
    r = np.sqrt(u)
    th = (np.arange(n_arms) + 0.5) * 2 * np.pi / n_arms
    px = (r[:, None] * np.cos(th)[None, :]).ravel()
    py = (r[:, None] * np.sin(th)[None, :]).ravel()
    wt = (w[:, None] * np.ones(n_arms)[None, :] / n_arms).ravel()
    # chief ray first, with zero weight
    return np.r_[0.0, px], np.r_[0.0, py], np.r_[0.0, wt]


def field_angles(spec: Spec, fracs=None):
    fracs = spec.field_fracs if fracs is None else fracs
    return jnp.deg2rad(spec.half_field_deg) * jnp.asarray(fracs, dtype=float)


def evaluate(params, layout: Layout, spec: Spec, beta=1.0, fracs=None, n_rings=6, n_arms=12,
             c_img=0.0):
    """Trace all fields x wavelengths x pupil samples; return wavefronts and ray data.

    `c_img` is the curvature of the image surface (0 for the usual flat sensor).
    """
    thetas = field_angles(spec, fracs)
    wls = jnp.asarray(spec.wavelengths)
    iref = len(spec.wavelengths) // 2
    px, py, wt = (jnp.asarray(a) for a in pupil_grid(n_rings, n_arms))
    radius = spec.epd / 2

    y0 = jax.vmap(lambda th: aim_chief(params, layout, th, wls[iref], beta))(thetas)

    def one(wl, th, y, a, b):
        o, d = launch(th, y, a, b, radius, layout)
        return trace_ray(params, layout, wl, beta, o, d, c_img)

    f = jax.vmap(one, in_axes=(None, None, None, 0, 0))  # pupil
    f = jax.vmap(f, in_axes=(None, 0, 0, None, None))  # field
    f = jax.vmap(f, in_axes=(0, None, None, None, None))  # wavelength
    r = f(wls, thetas, y0, px, py)  # leaves: [W, F, P, ...]

    P = r["img"][iref, :, 0]  # [F, 3] chief image points (reference wavelength)
    Q = r["pos_last"][iref, :, 0]
    Rs = jnp.linalg.norm(Q - P, axis=-1)  # [F]
    w = r["pos_last"] - P[None, :, None, :]
    wd = jnp.sum(w * r["dir"], -1)
    ww = jnp.sum(w * w, -1)
    disc = jnp.maximum(wd**2 - (ww - Rs[None, :, None] ** 2), 0.0)
    s = -wd - jnp.sqrt(disc)
    opl = r["opl_last"] + s
    mean = jnp.sum(opl * wt, -1, keepdims=True)
    opd_waves = -(opl - mean) / (wls[:, None, None] * 1e-3)

    return {
        "opd": opd_waves,  # [W, F, P] in waves of each wavelength
        "wt": wt,
        "img": r["img"],
        "heights": r["heights"],  # [W, F, P, S]
        "stop_xy": r["stop_xy"],
        "chief_img": P,
        "thetas": thetas,
        "y0": y0,
        "iref": iref,
    }


def rms_wfe(ev, spec: Spec):
    """Polychromatic RMS wavefront error per field [waves], and the field average."""
    lw = jnp.asarray(spec.wl_weights) / np.sum(spec.wl_weights)
    ms = jnp.sum(ev["opd"] ** 2 * ev["wt"], -1)  # [W, F]
    per_field = jnp.sqrt(jnp.sum(ms * lw[:, None], 0))
    return per_field


def rms_spot(ev, spec: Spec):
    """Polychromatic RMS spot radius per field [um], about the polychromatic centroid."""
    wt = ev["wt"]
    lw = jnp.asarray(spec.wl_weights) / np.sum(spec.wl_weights)
    xy = ev["img"][..., :2]  # [W, F, P, 2]
    w = lw[:, None, None] * wt[None, None, :]  # [W, 1, P]
    w = w / jnp.sum(w, axis=(0, 2), keepdims=True)
    c = jnp.sum(xy * w[..., None], axis=(0, 2), keepdims=True)
    d2 = jnp.sum((xy - c) ** 2, -1)
    return jnp.sqrt(jnp.sum(d2 * w, axis=(0, 2))) * 1e3


def paraxial_efl(params, layout: Layout, spec: Spec, beta=1.0, h=1e-3):
    o = jnp.array([0.0, h, layout.z_launch])
    d = jnp.array([0.0, 0.0, 1.0])
    r = trace_ray(params, layout, spec.ref_wavelength, beta, o, d)
    return -h * r["dir"][2] / r["dir"][1]


def distortion(ev, efl):
    """Chief-ray distortion relative to f*tan(theta) for every non-zero field."""
    y = ev["chief_img"][:, 1]
    ideal = efl * jnp.tan(ev["thetas"])
    return jnp.where(ideal != 0, (y - ideal) / jnp.where(ideal != 0, ideal, 1.0), 0.0)


def focus_shifts(params, layout: Layout, spec: Spec, fracs, beta=1.0, wavelength=None):
    """Tangential and sagittal focus position relative to the image plane [mm].

    Uses forward-mode AD on the chief ray: d(image position)/d(pupil) and
    d(image slope)/d(pupil) define where neighbouring rays cross the chief ray.
    """
    wl = spec.ref_wavelength if wavelength is None else wavelength
    radius = spec.epd / 2
    thetas = field_angles(spec, fracs)

    def per_field(th):
        y0 = aim_chief(params, layout, th, wl, beta)

        def f(pp):
            o, d = launch(th, y0, pp[0], pp[1], radius, layout)
            r = trace_ray(params, layout, wl, beta, o, d)
            return jnp.array(
                [r["img"][0], r["img"][1], r["dir"][0] / r["dir"][2], r["dir"][1] / r["dir"][2]]
            )

        J = jax.jacfwd(f)(jnp.zeros(2))
        s_tan = -J[1, 1] / J[3, 1]
        s_sag = -J[0, 0] / J[2, 0]
        return s_tan, s_sag

    return jax.vmap(per_field)(thetas)


def clear_semi_diameters(ev, margin=1.02):
    """Clear semi-diameter for every refracting surface, from the traced rays."""
    return jnp.max(ev["heights"], axis=(0, 1, 2)) * margin


_SUMMARY_JIT: dict = {}


def _summary_core(params, beta, c_img, layout, spec):
    ev = evaluate(params, layout, spec, beta, c_img=c_img)
    efl = paraxial_efl(params, layout, spec, beta)
    return efl, rms_wfe(ev, spec), rms_spot(ev, spec), distortion(ev, efl)


def summary(params, layout: Layout, spec: Spec, beta=1.0, c_img=0.0):
    """EFL, per-field RMS wavefront error and spot, distortion (compiled per layout)."""
    key = (layout, spec)
    if key not in _SUMMARY_JIT:
        if len(_SUMMARY_JIT) > 32:
            _SUMMARY_JIT.clear()
        _SUMMARY_JIT[key] = jax.jit(lambda p, b, c: _summary_core(p, b, c, layout, spec))
    efl, wfe, spot, dist = _SUMMARY_JIT[key](params, jnp.asarray(beta, float),
                                             jnp.asarray(c_img, float))
    return {
        "efl": float(efl),
        "rms_wfe": np.asarray(wfe),
        "rms_spot_um": np.asarray(spot),
        "distortion": np.asarray(dist),
        "mean_wfe": float(jnp.mean(wfe)),
    }


def dense_report(params, layout: Layout, spec: Spec, n_field=21, n_rings=8, n_arms=16):
    """Image quality on a dense field and pupil grid the merit function never sampled."""
    fracs = tuple(np.linspace(0, 1, n_field))
    ev = evaluate(params, layout, spec, fracs=fracs, n_rings=n_rings, n_arms=n_arms)
    wfe = np.asarray(rms_wfe(ev, spec))
    spot = np.asarray(rms_spot(ev, spec))
    efl = float(paraxial_efl(params, layout, spec))
    dist = np.asarray(distortion(ev, efl))
    return {"fracs": np.array(fracs), "rms_wfe": wfe, "mean_wfe": float(wfe.mean()),
            "max_wfe": float(wfe.max()), "rms_spot_um": spot, "efl": efl,
            "max_distortion": float(np.max(np.abs(dist)))}
