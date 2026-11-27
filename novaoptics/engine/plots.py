"""Matplotlib drawings of the engine analyses (spot diagram, OPD map, PSF, MTF)."""

from __future__ import annotations

import numpy as np

from ..plot import GRID, INK, MUTED, SERIES, _style

# wavelength colors: short -> long
WL_COLORS = [SERIES[0], SERIES[2], SERIES[1], SERIES[3], SERIES[4]]


def _label(ax, title):
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


def draw_spots(ax, sp):
    _style(ax)
    for j, (s, wl) in enumerate(zip(sp["spots"], sp["wavelengths"])):
        ax.scatter(s[:, 0] - sp["centroid"][0], s[:, 1] - sp["centroid"][1], s=3, lw=0,
                   color=WL_COLORS[j % len(WL_COLORS)], label=f"{wl * 1e3:.0f} nm")
    a = sp["airy_um"]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(a * np.cos(t), a * np.sin(t), color=MUTED, lw=0.8, ls="--")
    lim = max(sp["geo_um"], a) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("x [µm] (about centroid)", fontsize=8, color=INK)
    ax.set_ylabel("y [µm]", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper right", markerscale=3)
    _label(ax, f"spot  RMS {sp['rms_um']:.2f} µm  GEO {sp['geo_um']:.2f} µm  "
               f"Airy {a:.2f} µm")


def draw_opd(fig, ax, o):
    W = o["W"]
    v = np.nanmax(np.abs(W)) or 1e-3
    im = ax.imshow(W, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("waves", fontsize=7.5)
    ax.set_xlabel("pupil x", fontsize=8, color=INK)
    ax.set_ylabel("pupil y", fontsize=8, color=INK)
    _label(ax, f"OPD at {o['wavelength'] * 1e3:.0f} nm  RMS {o['rms']:.3f}  PV {o['pv']:.2f} waves")


def draw_psf(fig, ax, p, half_width_um=None, log=True):
    I = p["I"] / p["I"].max() * p["strehl"]
    x, y = p["x_um"], p["y_um"]
    if half_width_um is None:  # show where the light is
        prof = I.sum(0)
        keep = np.where(prof > prof.max() * 1e-3)[0]
        half_width_um = max(abs(x[keep[0]]), abs(x[keep[-1]]), 4 * p["dx_um"]) * 1.1
    mx = np.abs(x) <= half_width_um
    my = np.abs(y) <= half_width_um
    img = I[np.ix_(my, mx)]
    ext = (x[mx][0], x[mx][-1], y[my][0], y[my][-1])
    if log:
        img = np.log10(np.maximum(img, 1e-6))
    im = ax.imshow(img, origin="lower", extent=ext, cmap="magma",
                   vmin=-4 if log else 0, vmax=0 if log else None)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("log10 (relative to perfect peak)" if log else "relative to perfect peak",
                 fontsize=7)
    ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    ax.set_xlabel("x [µm]", fontsize=8, color=INK)
    ax.set_ylabel("y [µm]", fontsize=8, color=INK)
    _label(ax, f"PSF (FFT, polychromatic)  Strehl {p['strehl']:.3f}")


def draw_mtf(ax, m, max_freq=None):
    _style(ax)
    cut = max_freq or max(m["fx"][m["ideal_x"] > 1e-2][-1], m["fy"][m["ideal_y"] > 1e-2][-1])
    ax.plot(m["fx"], m["ideal_x"], color=MUTED, lw=1, label="diffraction limit")
    ax.plot(m["fx"], m["mtf_x"], color=SERIES[0], lw=2, label="sagittal (x)")
    ax.plot(m["fy"], m["mtf_y"], color=SERIES[1], lw=2, ls="--", label="tangential (y)")
    ax.set_xlim(0, cut)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("spatial frequency [cycles/mm]", fontsize=8, color=INK)
    ax.set_ylabel("MTF", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    _label(ax, "MTF (from the PSF)")


def analysis_figure(fig, sp, o, p, m, title=""):
    fig.clear()
    a = fig.subplots(2, 2)
    draw_spots(a[0, 0], sp)
    draw_opd(fig, a[0, 1], o)
    draw_psf(fig, a[1, 0], p)
    draw_mtf(a[1, 1], m)
    if title:
        fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")


_GEOM: dict = {}


def system_geometry(surfs, system, spec, n_rays=7, n_pts=61):
    """Ray paths, clear semi-diameters and global surface curves (y-z) for drawing;
    compiled once per (system, spec)."""
    import jax
    import jax.numpy as jnp

    from .system import frames
    from .trace import aim_chief, launch, sag, trace_ray

    key = (system, spec, n_rays, n_pts)
    if key not in _GEOM:
        if len(_GEOM) > 16:
            _GEOM.clear()
        wl = spec.ref_wavelength
        py = jnp.linspace(-1, 1, n_rays)
        fields = jnp.asarray(spec.fields, float)

        def geom(surfs):
            fr = frames(surfs, system)

            def per_field(f):
                xy = aim_chief(surfs, system, f, wl)
                one = lambda b: trace_ray(surfs, system, wl,
                                          *launch(xy, f, 0.0, b, spec.epd, system), fr)
                r = jax.vmap(one)(py)
                return r["pts"], r["local"]

            pts, loc = jax.vmap(per_field)(fields)
            sd = jnp.sqrt(jnp.max(loc[..., 0] ** 2 + loc[..., 1] ** 2, axis=(0, 1))) * 1.02
            curves = []
            for i, s in enumerate(system.surfaces):
                O, R = fr[i]
                y = jnp.linspace(-1.0, 1.0, n_pts) * jnp.maximum(sd[i], 1e-3)
                if s.shape == "cb":
                    z = jnp.zeros_like(y)
                else:
                    z = jax.vmap(lambda yy: sag(s, surfs[i], 0.0, yy))(y)
                curves.append(O[None, :] + jnp.stack([jnp.zeros_like(y), y, z], 1) @ R.T)
            return pts, sd, jnp.stack(curves), fr[-1][0], fr[-1][1]

        _GEOM[key] = jax.jit(geom)
    out = _GEOM[key](surfs)
    return tuple(np.asarray(a) for a in out)


def draw_system(ax, params, system, spec, n_rays=7, title=None, config=None, problem=None,
                geometry=None):
    """Meridional (y-z) drawing of an engine system: surfaces as sag curves placed with
    their coordinate frames, glass shaded between consecutive refracting surfaces,
    mirrors as thick lines, rays of every field at the reference wavelength.

    `params` is a list of surface dicts, or a design tree {"s", "mc"} with `problem`
    and `config` (configuration index) to draw one configuration.  `geometry`: a
    precomputed system_geometry() result (e.g. from a worker thread)."""
    from matplotlib.patches import Polygon

    from ..plot import GLASS, GLASS_EDGE, PHASE

    if isinstance(params, dict):
        from .merit import config_spec, config_surfaces

        c = config or 0
        surfs = config_surfaces(params, problem, c)
        spec = config_spec(spec, problem.configs[c])
    else:
        surfs = params
    pts, sd, curves, O_img, R_img = geometry or system_geometry(surfs, system, spec, n_rays)
    for j in range(pts.shape[0]):
        for k in range(pts.shape[1]):
            ax.plot(pts[j, k, :, 2], pts[j, k, :, 1], color=SERIES[j % len(SERIES)], lw=0.6,
                    alpha=0.85)
    S = system.surfaces
    drawn = [s.shape != "cb" and not (i == system.n - 1 and s.interact == "none")
             for i, s in enumerate(S)]
    for i, s in enumerate(S):
        if not drawn[i]:
            continue
        c = curves[i]
        if s.interact == "refract" and s.material not in ("air", "", "vacuum"):
            j = next((k for k in range(i + 1, len(S)) if drawn[k] and S[k].interact == "refract"),
                     None)
            if j is not None:
                poly = np.r_[c[:, [2, 1]], curves[j][::-1][:, [2, 1]]]
                ax.add_patch(Polygon(poly, closed=True, fc=GLASS, ec=GLASS_EDGE, lw=0.8,
                                     zorder=0))
        if s.interact == "reflect":
            ax.plot(c[:, 2], c[:, 1], color=GLASS_EDGE, lw=2.2)
        elif s.stop:
            ax.plot(c[[0, 0], 2], [c[0, 1] * 1.05, c[0, 1] * 1.6], color=INK, lw=1.5)
            ax.plot(c[[-1, -1], 2], [c[-1, 1] * 1.05, c[-1, 1] * 1.6], color=INK, lw=1.5)
        if s.phase and np.any(np.abs(np.asarray(surfs[i]["pcoef"])) > 1e-9):
            ax.plot(c[:, 2], c[:, 1], color=PHASE, lw=1.8, ls=(0, (2, 1.5)))
    hi = np.max(np.abs(pts[:, :, -1, 1])) * 1.15 + 1e-3
    yy = np.array([-hi, hi])
    ax.plot(O_img[2] + R_img[2, 1] * yy, O_img[1] + R_img[1, 1] * yy, color=INK, lw=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    return sd
