"""Matplotlib figures: system layouts (glass + residual phase), morph film strips,
merit trajectories and AD-based field curves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from . import analysis as A
from .merit import sag
from .system import Layout, Spec, vertex_positions
from .trace import asphere_poly, launch, trace_ray

# Reference categorical palette (light mode), slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#2b2b29"
MUTED = "#8a8980"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"
GLASS = "#cde2fb"
GLASS_EDGE = "#256abf"
PHASE = "#d55181"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def meridional_rays(params, layout: Layout, spec: Spec, beta=1.0, n_rays=7, fracs=None):
    """Ray paths [field, ray, point, 3] at the reference wavelength."""
    thetas = A.field_angles(spec, fracs)
    wl = spec.ref_wavelength
    py = jnp.linspace(-1, 1, n_rays)

    def per_field(th):
        y0 = A.aim_chief(params, layout, th, wl, beta)

        def one(b):
            o, d = launch(th, y0, 0.0, b, spec.epd / 2, layout)
            return trace_ray(params, layout, wl, beta, o, d)["path"]

        return jax.vmap(one)(py)

    return np.asarray(jax.vmap(per_field)(thetas))


def draw_layout(ax, params, layout: Layout, spec: Spec, beta=1.0, title=None, sd=None):
    from .glass import resolve

    p = {k: np.asarray(v) for k, v in resolve(params, layout).items()}
    if sd is None:
        ev = A.evaluate(params, layout, spec, beta)
        sd = np.asarray(A.clear_semi_diameters(ev))
    vp = vertex_positions(params, layout)
    paths = meridional_rays(params, layout, spec, beta)
    for f in range(paths.shape[0]):
        for r in range(paths.shape[1]):
            ax.plot(paths[f, r, :, 2], paths[f, r, :, 1], color=SERIES[f], lw=0.6, alpha=0.85)
    for i in range(layout.n_elem):
        z1, z2 = float(vp["elem_front"][i]), float(vp["elem_back"][i])
        h = max(sd[2 * i], sd[2 * i + 1])
        y = np.linspace(-h, h, 81)
        s1 = np.asarray(sag(p["c1"][i], np.abs(y)))
        s2 = np.asarray(sag(p["c2"][i], np.abs(y)))
        if layout.aspheres:
            s2 = s2 + np.array([float(asphere_poly(jnp.asarray(p["asph"][i]), yy * yy,
                                                   layout.r_norm)[0]) for yy in y])
        glassy = abs(p["c1"][i]) + abs(p["c2"][i]) > 1e-6 or p["t"][i] > 1e-3
        if glassy:
            poly = np.c_[np.r_[z1 + s1, (z2 + s2)[::-1]], np.r_[y, y[::-1]]]
            ax.add_patch(Polygon(poly, closed=True, fc=GLASS, ec=GLASS_EDGE, lw=0.8, zorder=0))
        # residual ideal phase: dashed screen whose opacity is its share of the power
        phi_phase = -2 * p["phase"][i, 0] / layout.r_norm**2
        phi_tot = abs(phi_phase) + abs((p["nd"][i] - 1) * (p["c1"][i] - p["c2"][i]))
        share = abs(phi_phase) / phi_tot if phi_tot > 0 else 0.0
        if share > 0.02 or (not glassy and np.any(np.abs(p["phase"][i]) > 1e-6)):
            ax.plot(z2 + s2, y, color=PHASE, lw=1.8, ls=(0, (2, 1.5)),
                    alpha=float(np.clip(0.25 + 0.75 * share, 0, 1)))
    stop = float(vp["stop"])
    sh = spec.epd / 2
    ax.plot([stop, stop], [sh * 1.05, sh * 1.6], color=INK, lw=1.5)
    ax.plot([stop, stop], [-sh * 1.05, -sh * 1.6], color=INK, lw=1.5)
    zi = float(vp["image"])
    hi = spec.efl * np.tan(np.deg2rad(spec.half_field_deg)) * 1.15
    ax.plot([zi, zi], [-hi, hi], color=INK, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-12, zi + 3)
    ax.set_ylim(-max(hi, 14), max(hi, 14))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def film_strip(snaps, spec: Spec, path, ncols=2):
    n = len(snaps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.7 * nrows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, sn) in zip(axes, snaps):
        draw_layout(ax, sn.params, sn.layout, spec, sn.beta,
                    title=f"{label}   RMS WFE {sn.mean_wfe:.2f} waves")
    for ax in axes[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=SERIES[i], lw=1.5,
                          label=f"{f * spec.half_field_deg:.0f}° field")
               for i, f in enumerate(spec.field_fracs)]
    handles += [plt.Line2D([], [], color=PHASE, lw=1.8, ls=(0, (2, 1.5)), label="ideal phase"),
                plt.Rectangle((0, 0), 1, 1, fc=GLASS, ec=GLASS_EDGE, label="glass")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def trajectory(recs, labels, path):
    """Mean RMS WFE along the pipeline, one line per recorder, stages shaded."""
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    _style(ax)
    for j, (rec, lab) in enumerate(zip(recs, labels)):
        y = [s.mean_wfe for s in rec.snaps]
        ax.plot(range(len(y)), y, color=SERIES[j], lw=2, marker="o", ms=3.5, label=lab)
    rec = recs[0]
    stages = [s.stage for s in rec.snaps]
    start = 0
    for k in range(1, len(stages) + 1):
        if k == len(stages) or stages[k] != stages[start]:
            if (len(set(stages[:start])) % 2) == 1:  # shade every other stage
                ax.axvspan(start - 0.5, k - 0.5, color=GRID, alpha=0.5, lw=0)
            nth = len(set(stages[:start]))
            ax.text((start + k - 1) / 2, 0.97 - 0.07 * (nth % 2), stages[start],
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
                    color=MUTED)
            start = k
    ax.set_yscale("log")
    ax.set_ylabel("mean RMS wavefront error [waves]", fontsize=8, color=INK)
    ax.set_xlabel("pipeline step", fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    if len(recs) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def performance(designs, spec: Spec, path, n_field=11):
    """RMS WFE vs field and AD field curves (tangential/sagittal) for each design."""
    fracs = tuple(np.linspace(0, 1, n_field))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
        ax.grid(color=GRID, lw=0.6)
    deg = np.array(fracs) * spec.half_field_deg
    for j, (lab, p, lay) in enumerate(designs):
        ev = A.evaluate(p, lay, spec, fracs=fracs)
        a1.plot(deg, np.asarray(A.rms_wfe(ev, spec)), color=SERIES[j], lw=2, label=lab)
        st, ss = A.focus_shifts(p, lay, spec, fracs)
        a2.plot(np.asarray(st), deg, color=SERIES[j], lw=2, label=f"{lab}: tangential")
        a2.plot(np.asarray(ss), deg, color=SERIES[j], lw=2, ls="--", label=f"{lab}: sagittal")
    a1.axhline(0.07, color=MUTED, lw=1, ls=":")
    a1.text(0.3, 0.075, "Maréchal 0.07", fontsize=7.5, color=MUTED, va="bottom")
    a1.set_xlabel("field angle [deg]", fontsize=8, color=INK)
    a1.set_ylabel("polychromatic RMS WFE [waves]", fontsize=8, color=INK)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK)
    a2.axvline(0, color=MUTED, lw=0.8)
    a2.set_xlabel("focus shift from image plane [mm]", fontsize=8, color=INK)
    a2.set_ylabel("field angle [deg]", fontsize=8, color=INK)
    a2.set_title("field curves from forward-mode AD (one ray per field)", fontsize=8,
                 color=INK, loc="left")
    a2.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
