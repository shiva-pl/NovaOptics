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
