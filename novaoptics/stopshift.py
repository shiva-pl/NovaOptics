"""Stop-shift theory: place the aperture stop after color correction.

For a stop moved to axial position z_s, the new chief ray is the old one plus E times
the marginal ray, with E = -ybar(z_s) / y(z_s) so that it passes through the new stop
center.  The object-space field angle and the entrance-pupil height are unchanged
(the marginal ray enters parallel to the axis), so the Lagrange invariant is too, and
the third-order sums change in closed form (Welford):

    S_I*   = S_I                          C_I*  = C_I
    S_II*  = S_II  + E S_I                C_II* = C_II + E C_I
    S_III* = S_III + 2E S_II + E^2 S_I
    S_IV*  = S_IV
    S_V*   = S_V + E (3 S_III + S_IV) + 3 E^2 S_II + E^3 S_I

Hence the classic rules: with axial color present (C_I != 0) lateral color is linear in
the stop position and vanishes at one position; once axial color is corrected (C_I = 0)
lateral color does not depend on the stop at all, and the stop is free to be used for
coma, astigmatism and field curvature.  That is the order used here: the COLOR stage
corrects axial color, then STOP-SHIFT places the stop.

In phase space the elements are thin screens, so the stop can be moved continuously
across them; `relayout` turns a stop position into the item order the tracer needs.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from .diagnostics import _seidel_core
from .merit import virtual_petzval_curvature
from .system import Layout, Spec, vertex_positions


def shifted_sums(S, E):
    """Third-order and primary-color totals after a stop shift with eccentricity E.

    S: totals in SEIDEL_NAMES order (S_I..S_V, C_I, C_II); E scalar or array.
    """
    SI, SII, SIII, SIV, SV, CI, CII = S
    E = np.asarray(E, dtype=float)
    return np.stack([
        SI + 0 * E,
        SII + E * SI,
        SIII + 2 * E * SII + E**2 * SI,
        SIV + 0 * E,
        SV + E * (3 * SIII + SIV) + 3 * E**2 * SII + E**3 * SI,
        CI + 0 * E,
        CII + E * CI,
    ], axis=-1)


def _air_intervals(params, layout: Layout, margin=0.5, reach=30.0):
    """Axial intervals (z0, z1) in air where the stop may sit, in system coordinates."""
    vp = vertex_positions(params, layout)
    fronts = [float(v) for v in vp["elem_front"]]
    backs = [float(v) for v in vp["elem_back"]]
    order = np.argsort(fronts)
    out = [(fronts[order[0]] - reach, fronts[order[0]] - margin)]
    for a, b in zip(order[:-1], order[1:]):
        if fronts[b] - backs[a] > 2 * margin:
            out.append((backs[a] + margin, fronts[b] - margin))
    z_img = float(vp["image"])
    out.append((backs[order[-1]] + margin, min(backs[order[-1]] + reach, z_img - 5.0)))
    return [(a, b) for a, b in out if b > a]


def scan(params, layout: Layout, spec: Spec, beta=1.0, virtual_petzval=1.0, n=600,
         weights=(1.0, 1.0, 1.0, 0.0)):
    """Third-order merit of the stop at every candidate axial position.

    Merit: weighted sum of squares of coma W131, astigmatism W222, sagittal field
    curvature W220 (with the screens' virtual Petzval, i.e. the curvature their glass
    will add) and distortion W311, in d-line waves.  Returns positions, merit, the
    shifted coefficients and E.
    """
    surfs, S, tr = _seidel_core(params, layout, spec, beta)
    S = np.asarray(S)
    tot = S.sum(0)
    lam = spec.ref_wavelength * 1e-3
    # virtual Petzval of the ideal phase: the screens will become glass of index n_d
    if virtual_petzval:
        tot = tot.copy()
        tot[3] += float(tr["H"]) ** 2 * float(virtual_petzval_curvature(params, layout))
    z = np.array([float(s.z) for s in surfs])
    ym, upm = np.asarray(tr["ym"]), np.asarray(tr["upm"])
    yc, upc = np.asarray(tr["yc"]), np.asarray(tr["upc"])

    zs, E = [], []
    for a, b in _air_intervals(params, layout):
        for zz in np.linspace(a, b, max(int(n * (b - a) / 60.0), 8)):
            j = np.searchsorted(z, zz, side="right") - 1
            if j < 0:  # in front of the first surface: collimated marginal ray
                y, yb = ym[0], yc[0] + (zz - z[0]) * float(tr["ubar"])
            else:
                y = ym[j] + (zz - z[j]) * upm[j]
                yb = yc[j] + (zz - z[j]) * upc[j]
            if abs(y) < 1e-9:
                continue
            zs.append(zz)
            E.append(-yb / y)
    zs, E = np.array(zs), np.array(E)
    Sh = shifted_sums(tot, E)
    W131 = Sh[:, 1] / 2 / lam
    W222 = Sh[:, 2] / 2 / lam
    W220 = (Sh[:, 2] + Sh[:, 3]) / 4 / lam
    W311 = Sh[:, 4] / 2 / lam
    w = weights
    merit = w[0] * W131**2 + w[1] * W222**2 + w[2] * W220**2 + w[3] * W311**2
    return {"z": zs, "merit": merit, "E": E, "W131": W131, "W222": W222, "W220": W220,
            "W311": W311, "CI": Sh[:, 5], "CII": Sh[:, 6], "current_z":
            float(vertex_positions(params, layout)["stop"])}


def candidates(sc, k=3, min_sep=2.0):
    """The k best local minima of the scan merit, at least `min_sep` mm apart."""
    m, z = sc["merit"], sc["z"]
    idx = [i for i in range(len(m))
           if (i == 0 or m[i] <= m[i - 1]) and (i == len(m) - 1 or m[i] <= m[i + 1])]
    idx.sort(key=lambda i: m[i])
    out = []
    for i in idx:
        if all(abs(z[i] - z[j]) >= min_sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return [float(z[i]) for i in out]


def relayout(params, layout: Layout, z_stop):
    """Move the stop to axial position z_stop (in air); returns (params, layout).

    Elements keep their absolute positions and the image plane stays put; the item
    order, gaps and back focal distance are rebuilt around the new stop.
    """
    vp = vertex_positions(params, layout)
    fronts = [float(v) for v in vp["elem_front"]]
    backs = [float(v) for v in vp["elem_back"]]
    z_img = float(vp["image"])
    k = int(sum(f < z_stop for f in fronts))
    lay = dataclasses.replace(layout, stop_index=k)
    pos = []  # (front, back) of each item in the new order
    for item in lay.items():
        pos.append((z_stop, z_stop) if item == "stop" else (fronts[item], backs[item]))
    gaps = [pos[i + 1][0] - pos[i][1] for i in range(len(pos) - 1)]
    q = dict(params)
    q["gaps"] = jnp.asarray(gaps, dtype=float)
    q["bfd"] = jnp.asarray(z_img - pos[-1][1], dtype=float)
    return q, lay
