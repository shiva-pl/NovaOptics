"""Image analysis on the surface engine: OPD maps, spot diagrams, FFT PSF and MTF.

* **OPD map**: wavefront error over a square pupil grid (unit-disk mask), in waves,
  relative to the reference sphere centered on the reference-wavelength chief image
  point (so lateral color shows up as tilt).  Piston is removed.
* **Spot diagram**: image-surface intercepts of a hexapolar pupil sample per
  wavelength, relative to the reference chief image point; RMS radius about the
  polychromatic centroid, geometric (max) radius, and the Airy radius for comparison.
* **PSF** (Fraunhofer, FFT): pupil function A exp(2 pi i W) zero-padded `pad` times.
  The image-plane sampling comes from the real rays: the image-space direction cosines
  across the pupil (in the image surface's frame) are fitted linearly, and
  dx = lambda / (N dL), so tilted / anamorphic image space is handled to first order.
  Polychromatic PSFs are resampled onto the reference-wavelength grid and summed with
  the spectral weights.  Strehl ratio = peak / peak of the same pupil without aberration.
* **MTF**: modulus of the Fourier transform of the (polychromatic) PSF, sagittal (x)
  and tangential (y) cuts, with the diffraction-limited MTF of the same pupil.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

from .system import Spec, System
from .trace import bundle


@lru_cache(maxsize=32)
def _grid_fn(system: System, spec: Spec, n: int):
    """Compiled trace of an n x n pupil grid (all points, masked later)."""
    g = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    px, py = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())

    def f(params, field, wl):
        b = bundle(params, system, spec, field, wl, px, py)
        return b["opd"], b["img"], b["dir"], b["ok"], b["chief_ref_img"]

    return jax.jit(f)


@lru_cache(maxsize=32)
def _points_fn(system: System, spec: Spec, n_pts: int):
    def f(params, field, wl, px, py):
        b = bundle(params, system, spec, field, wl, px, py)
        return b["img"], b["ok"], b["chief_ref_img"]

    return jax.jit(f)


def _wl_list(spec, wl):
    if wl is None:
        w = np.asarray(spec.wl_weights, float)
        return list(spec.wavelengths), list(w / w.sum())
    return [wl], [1.0]


def opd_map(params, system: System, spec: Spec, field, wl=None, n=65, remove_tilt=False):
    """Wavefront error map [waves] on an n x n grid (NaN outside the pupil)."""
    wl = spec.ref_wavelength if wl is None else wl
    f = _grid_fn(system, spec, n)
    opd, _, _, ok, _ = f(params, jnp.asarray(field, float), jnp.asarray(wl, float))
    g = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    mask = (X**2 + Y**2 <= 1.0) & np.asarray(ok).reshape(n, n)
    W = np.asarray(opd).reshape(n, n) / (wl * 1e-3)
    A = np.c_[np.ones(mask.sum()), X[mask], Y[mask]] if remove_tilt else np.ones((mask.sum(), 1))
    coef, *_ = np.linalg.lstsq(A, W[mask], rcond=None)
    W = W - (coef[0] + (coef[1] * X + coef[2] * Y if remove_tilt else 0.0))
    W[~mask] = np.nan
    return {"x": g, "y": g, "W": W, "rms": float(np.sqrt(np.nanmean(W**2))),
            "pv": float(np.nanmax(W) - np.nanmin(W)), "wavelength": wl,
            "vignetted": int((X**2 + Y**2 <= 1.0).sum() - mask.sum())}


def hexapolar(rings=6):
    px, py = [0.0], [0.0]
    for k in range(1, rings + 1):
        r = k / rings
        for j in range(6 * k):
            a = 2 * np.pi * j / (6 * k)
            px.append(r * np.cos(a))
            py.append(r * np.sin(a))
    return np.array(px), np.array(py)


def spot_diagram(params, system: System, spec: Spec, field, rings=8):
    """Spot diagram (image-surface local coordinates relative to the reference chief
    image point, um) for every wavelength, plus RMS / geometric radius [um]."""
    px, py = hexapolar(rings)
    f = _points_fn(system, spec, px.size)
    spots, centroid_acc, wsum = [], np.zeros(2), 0.0
    wls, wts = _wl_list(spec, None)
    for wl, w in zip(wls, wts):
        img, ok, ref = f(params, jnp.asarray(field, float), jnp.asarray(wl, float),
                         jnp.asarray(px), jnp.asarray(py))
        xy = (np.asarray(img) - np.asarray(ref)) * 1e3
        xy = xy[np.asarray(ok)]
        spots.append(xy)
        centroid_acc += w * xy.mean(0)
        wsum += w
    c = centroid_acc / wsum
    r2 = sum(w * np.mean(np.sum((s - c) ** 2, 1)) for s, w in zip(spots, wts)) / wsum
    rmax = max(float(np.max(np.linalg.norm(s - c, axis=1))) for s in spots)
    fno = working_fnumber(params, system, spec, field)
    return {"spots": spots, "wavelengths": wls, "centroid": c, "rms_um": float(np.sqrt(r2)),
            "geo_um": rmax, "airy_um": 1.22 * spec.ref_wavelength * fno,
            "fno_working": fno}


def _dircos_fit(dirs, R_img, X, Y, mask):
    """Linear fit of image-space direction cosines (in the image frame) over the pupil."""
    d = np.asarray(dirs) @ np.asarray(R_img)  # rows: R^T d
    A = np.c_[np.ones(mask.sum()), X[mask], Y[mask]]
    cL, *_ = np.linalg.lstsq(A, d[mask.ravel(), 0], rcond=None)
    cM, *_ = np.linalg.lstsq(A, d[mask.ravel(), 1], rcond=None)
    return cL, cM  # L = cL0 + cL1 px + cL2 py, M = ...


def working_fnumber(params, system: System, spec: Spec, field):
    """1 / (2 sin U'): half-angle of the image-space cone, from the pupil-edge rays."""
    px, py = hexapolar(1)
    f = _dirs_fn(system, spec, px.size)
    d = np.asarray(f(params, jnp.asarray(field, float), jnp.asarray(spec.ref_wavelength, float),
                     jnp.asarray(px), jnp.asarray(py)))
    sinU = np.sqrt(1.0 - np.clip(d[1:] @ d[0], -1.0, 1.0) ** 2)
    return float(1.0 / (2.0 * np.mean(sinU)))


@lru_cache(maxsize=32)
def _dirs_fn(system: System, spec: Spec, n_pts: int):
    return jax.jit(lambda params, field, wl, px, py: bundle(params, system, spec, field, wl,
                                                            px, py)["dir"])


def _psf_one(params, system, spec, field, wl, n, pad, aberrated=True):
    from .system import frames

    f = _grid_fn(system, spec, n)
    opd, img, dirs, ok, ref = f(params, jnp.asarray(field, float), jnp.asarray(wl, float))
    g = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    mask = (X**2 + Y**2 <= 1.0) & np.asarray(ok).reshape(n, n)
    W = np.asarray(opd).reshape(n, n) / (wl * 1e-3) if aberrated else np.zeros((n, n))
    W = W - W[mask].mean()
    P = np.zeros((n, n), complex)
    P[mask] = np.exp(2j * np.pi * W[mask])
    N = n * pad
    Pp = np.zeros((N, N), complex)
    Pp[:n, :n] = P
    I = np.abs(np.fft.fftshift(np.fft.fft2(Pp))) ** 2
    R_img = np.asarray(frames(params, system)[-1][1])
    cL, cM = _dircos_fit(dirs, R_img, X, Y, mask)
    dpx = g[1] - g[0]
    # pupil sample step in direction cosines along the grid axes (x <- columns); where
    # the direction cosine decreases across the pupil (a real image) the transform runs
    # the other way: flip that axis about the center sample
    dL, dM = abs(cL[1]) * dpx, abs(cM[2]) * dpx
    if cL[1] < 0:
        I = np.roll(I[:, ::-1], 1, axis=1)
    if cM[2] < 0:
        I = np.roll(I[::-1, :], 1, axis=0)
    lam = wl * 1e-3
    return I, lam / (N * dL), lam / (N * dM), np.asarray(ref)


def psf(params, system: System, spec: Spec, field, wl=None, n=64, pad=4):
    """PSF on a grid centered on the reference chief image point.

    Returns intensity normalized to unit sum, the grid coordinates [um] (x, y), the
    Strehl ratio and the per-axis sampling.  wl=None: polychromatic (spectral weights).
    """
    wls, wts = _wl_list(spec, wl)
    ref_wl = spec.ref_wavelength if wl is None else wl
    I_ref, dx0, dy0, _ = _psf_one(params, system, spec, field, ref_wl, n, pad, False)
    N = n * pad
    total = np.zeros((N, N))
    ideal = np.zeros((N, N))
    for w_l, w in zip(wls, wts):
        I, dx, dy, _ = _psf_one(params, system, spec, field, w_l, n, pad)
        I0, *_ = _psf_one(params, system, spec, field, w_l, n, pad, False)
        s = I0.sum()
        I, I0 = I / s, I0 / s  # same normalization: Strehl = peak ratio
        if (dx, dy) != (dx0, dy0):
            I = _resample(I, dx, dy, dx0, dy0) * (dx0 * dy0) / (dx * dy)
            I0 = _resample(I0, dx, dy, dx0, dy0) * (dx0 * dy0) / (dx * dy)
        total += w * I
        ideal += w * I0
    strehl = float(total.max() / ideal.max())
    c = np.arange(N) - N // 2
    norm = total.sum()
    return {"I": total / norm, "I_ideal": ideal / ideal.sum(), "x_um": c * dx0 * 1e3,
            "y_um": c * dy0 * 1e3, "strehl": strehl, "dx_um": dx0 * 1e3, "dy_um": dy0 * 1e3}


def _resample(I, dx, dy, dx0, dy0):
    """Resample a centered PSF from sampling (dx, dy) to (dx0, dy0)."""
    N = I.shape[0]
    c = np.arange(N) - N // 2
    Yt, Xt = np.meshgrid(c * dy0 / dy + N // 2, c * dx0 / dx + N // 2, indexing="ij")
    return map_coordinates(I, [Yt, Xt], order=1, mode="constant", cval=0.0)


def mtf(params, system: System, spec: Spec, field, wl=None, n=64, pad=4, p=None):
    """Sagittal (x) and tangential (y) MTF [cycles/mm] with the diffraction limit.

    `p`: a `psf` result to reuse."""
    p = psf(params, system, spec, field, wl, n, pad) if p is None else p
    out = {}
    for key, I in (("mtf", p["I"]), ("ideal", p["I_ideal"])):
        O = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I))))
        O /= O.max()
        N = I.shape[0]
        out[key + "_x"] = O[N // 2, N // 2:]
        out[key + "_y"] = O[N // 2:, N // 2]
    N = p["I"].shape[0]
    out["fx"] = np.arange(N // 2) / (N * p["dx_um"] * 1e-3)
    out["fy"] = np.arange(N // 2) / (N * p["dy_um"] * 1e-3)
    out["strehl"] = p["strehl"]
    return out


def analyze(params, system: System, spec: Spec, field, n=64, pad=4):
    """Spot diagram, reference-wavelength OPD map, polychromatic PSF and MTF."""
    sp = spot_diagram(params, system, spec, field)
    o = opd_map(params, system, spec, field)
    p = psf(params, system, spec, field, n=n, pad=pad)
    return {"spot": sp, "opd": o, "psf": p, "mtf": mtf(params, system, spec, field, p=p)}


def focus_shifts(params, system: System, spec: Spec, fields, wl=None):
    """Tangential and sagittal focus positions relative to the image surface [mm] for
    each field, from forward-mode AD of rays around the chief ray (generalized ray
    trace): d(image position)/d(pupil) and d(image slope)/d(pupil), in the image
    surface's frame."""
    from . import merit as EM
    from .system import frames
    from .trace import aim_chief, launch, trace_ray

    wl = spec.ref_wavelength if wl is None else wl
    fields = jnp.asarray(fields, float)

    def fn(p):
        R_img = frames(p, system)[-1][1]

        def per_field(f):
            xy = aim_chief(p, system, f, wl)

            def ray(pp):
                o, d = launch(xy, f, pp[0], pp[1], spec.epd, system)
                r = trace_ray(p, system, wl, o, d)
                dl = R_img.T @ r["dir"]
                img = r["local"][-1]
                return jnp.array([img[0], img[1], dl[0] / dl[2], dl[1] / dl[2]])

            J = jax.jacfwd(ray)(jnp.zeros(2))
            return -J[1, 1] / J[3, 1], -J[0, 0] / J[2, 0]

        return jax.vmap(per_field)(fields)

    st, ss = EM.compiled(f"focus {fields.shape[0]} {wl}", system, spec, fn)(params)
    return np.asarray(st), np.asarray(ss)
