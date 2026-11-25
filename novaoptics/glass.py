"""Glass model.

Glasses are described by the continuous pair (n_d, V_d) so that the optimizer can
move through "glass space" smoothly.  Dispersion uses a two-term Cauchy model
anchored at the d line with the F-C dispersion set by the Abbe number:

    n(lambda) = n_d + beta * (n_d - 1)/V_d * g(lambda)
    g(lambda) = (1/lambda^2 - 1/lambda_d^2) / (1/lambda_F^2 - 1/lambda_C^2)

`beta` is a continuation knob: beta = 0 switches dispersion off entirely, beta = 1
is the physical glass.  The Cauchy model is accurate to a few 1e-4 in index across
the visible for ordinary glasses, which is ample for a proof of concept; a
Sellmeier catalog can replace it later without touching the rest of the code.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

LAMBDA_F = 0.4861327  # um
LAMBDA_D = 0.5875618  # um
LAMBDA_C = 0.6562725  # um
VISIBLE = (LAMBDA_F, LAMBDA_D, LAMBDA_C)

# Common optical glasses (Schott nomenclature), (n_d, V_d).
CATALOG: dict[str, tuple[float, float]] = {
    "N-BK7": (1.5168, 64.17),
    "N-K5": (1.5225, 59.48),
    "N-BAK4": (1.5688, 55.98),
    "N-SK16": (1.6204, 60.32),
    "N-SSK8": (1.6177, 49.83),
    "N-LAK9": (1.6910, 54.71),
    "N-LAK22": (1.6511, 55.89),
    "N-BAF10": (1.6700, 47.11),
    "F2": (1.6200, 36.37),
    "N-SF2": (1.6477, 33.82),
    "N-SF5": (1.6727, 32.25),
    "N-SF11": (1.7847, 25.68),
    "N-LASF44": (1.8042, 46.50),
    "N-FK51A": (1.4866, 84.47),
    "N-PK52A": (1.4970, 81.61),
    "N-PSK53A": (1.6180, 63.39),
    "N-BALF4": (1.5796, 53.87),
    "N-KF9": (1.5234, 51.54),
    "LLF1": (1.5481, 45.75),
    "N-LAF2": (1.7440, 44.85),
    "N-SF1": (1.7174, 29.62),
    "N-SF6": (1.8052, 25.36),
    "N-SF57": (1.8467, 23.78),
    "N-LASF9": (1.8503, 32.17),
    "N-LASF31A": (1.8830, 40.76),
    "N-LASF43": (1.8061, 40.61),
    "N-LAF21": (1.7880, 47.49),
}
# (n_d, V_d values are nominal catalog numbers, rounded.)

# Approximate outline of the populated region of the glass map: for a given n_d,
# real glasses have V_min(n_d) <= V_d <= V_max(n_d).
_GLASS_NMAP = np.array([1.45, 1.52, 1.62, 1.70, 1.80, 1.90, 1.95])
_GLASS_VMAX = np.array([95.0, 72.0, 64.0, 56.0, 47.0, 38.0, 33.0])
_GLASS_VMIN = np.array([62.0, 50.0, 35.0, 28.0, 24.0, 19.0, 17.0])


def glass_region_violation(nd, vd):
    """Hinge distances (in Abbe units) outside the populated glass-map region."""
    vmax = jnp.interp(nd, _GLASS_NMAP, _GLASS_VMAX)
    vmin = jnp.interp(nd, _GLASS_NMAP, _GLASS_VMIN)
    return jnp.maximum(vd - vmax, 0.0), jnp.maximum(vmin - vd, 0.0)

ND_BOUNDS = (1.47, 1.92)
VD_BOUNDS = (20.0, 90.0)


def g_factor(wavelength_um):
    w = jnp.asarray(wavelength_um)
    return (1.0 / w**2 - 1.0 / LAMBDA_D**2) / (1.0 / LAMBDA_F**2 - 1.0 / LAMBDA_C**2)


def index(nd, vd, wavelength_um, beta=1.0):
    """Refractive index of a (n_d, V_d) glass at `wavelength_um`."""
    return nd + beta * (nd - 1.0) / vd * g_factor(wavelength_um)


def _catalog_arrays():
    names = list(CATALOG)
    arr = np.array([CATALOG[k] for k in names])
    return names, arr


def glass_distance(nd, vd, nd_c, vd_c):
    """Normalized distance in the glass map (one unit ~ 0.1 in n_d or 10 in V_d)."""
    return jnp.sqrt(((nd - nd_c) / 0.1) ** 2 + ((vd - vd_c) / 10.0) ** 2 + 1e-12)


def catalog_softmin_distance(nd, vd, temperature=0.05, points=None):
    """Smooth distance from (nd, vd) to the nearest catalog glass.

    `points`: (n_d array, V_d array) of another catalog; default the built-in list.
    """
    if points is None:
        _, arr = _catalog_arrays()
        points = (arr[:, 0], arr[:, 1])
    d = glass_distance(nd, vd, jnp.asarray(points[0]), jnp.asarray(points[1]))
    return -temperature * jnp.log(jnp.sum(jnp.exp(-d / temperature)))


def nearest_glasses(nd, vd, k=2):
    """Return the k nearest catalog glasses as [(name, nd, vd, distance), ...]."""
    names, arr = _catalog_arrays()
    d = np.sqrt(((nd - arr[:, 0]) / 0.1) ** 2 + ((vd - arr[:, 1]) / 10.0) ** 2)
    order = np.argsort(d)[:k]
    return [(names[i], float(arr[i, 0]), float(arr[i, 1]), float(d[i])) for i in order]


def glass_name(nd, vd, tol=1e-6):
    for name, (n, v) in CATALOG.items():
        if abs(n - nd) < tol and abs(v - vd) < tol:
            return name
    return f"model({nd:.4f}/{vd:.1f})"


# --------------------------------------------------------------------------------
# Glass model dispatch.  Without a glass space on the layout, an element's glass is
# the Cauchy (n_d, V_d) model above.  With `layout.glass` (a catalog.GlassSpace) it is
#
#     n(lambda_k) = mu_k + sum_j B_kj (g_j * scale_j) + gr_k
#
# with normalized PCA coordinates g (free) and a fixed residual gr (zero for a
# continuous glass, the exact off-span part of a snapped catalog glass).  beta keeps
# its meaning: n_beta = n_d + beta (n - n_d).

def _pca_curve(params, i, gs):
    return (jnp.asarray(gs.mu) + jnp.asarray(gs.B) @ (params["g"][i] * jnp.asarray(gs.scale))
            + params["gr"][i])


def index_at(params, i, wavelength, beta, layout):
    gs = getattr(layout, "glass", None)
    if gs is None:
        reg = getattr(layout, "region", None)
        if reg is not None and reg.h_coef is not None:
            return normal_index(params["nd"][i], params["vd"][i], wavelength, beta, reg)
        return index(params["nd"][i], params["vd"][i], wavelength, beta)
    x = _pca_curve(params, i, gs)
    # wavelength may be traced (vmapped in the ray trace): interpolate on the grid,
    # which is exact at grid points (all design wavelengths are on the grid)
    n = jnp.interp(wavelength, jnp.asarray(gs.grid), x)
    nd = x[gs.grid_index(LAMBDA_D)]
    return nd + beta * (n - nd)


def normal_index(nd, vd, wavelength, beta, gs):
    """Continuous (n_d, V_d) glass with the catalog's normal-glass dispersion shape."""
    from .catalog import normal_features

    h = normal_features(jnp.atleast_1d(nd), jnp.atleast_1d(vd))[0] @ jnp.asarray(gs.h_coef)
    hw = jnp.interp(wavelength, jnp.asarray(gs.grid), h)
    return nd + beta * (nd - 1.0) / vd * hw


def nd_vec(params, layout):
    gs = getattr(layout, "glass", None)
    if gs is None:
        return params["nd"]
    return jnp.stack([_pca_curve(params, i, gs)[gs.grid_index(LAMBDA_D)]
                      for i in range(layout.n_elem)])


def vd_vec(params, layout):
    gs = getattr(layout, "glass", None)
    if gs is None:
        return params["vd"]
    out = []
    for i in range(layout.n_elem):
        x = _pca_curve(params, i, gs)
        nF, nd, nC = (x[gs.grid_index(w)] for w in (LAMBDA_F, LAMBDA_D, LAMBDA_C))
        out.append((nd - 1.0) / (nF - nC))
    return jnp.stack(out)


def resolve(params, layout):
    """Params with `nd`/`vd` filled in from the glass model (no-op for Cauchy)."""
    if getattr(layout, "glass", None) is None:
        return params
    return dict(params, nd=nd_vec(params, layout), vd=vd_vec(params, layout))


def space_distance(g, gs, temperature=0.05):
    """Smooth distance (normalized PCA units) from coordinates g to the nearest glass."""
    import jax

    c = jnp.asarray(gs.coords / gs.scale)
    d = jnp.sqrt(jnp.sum((g[None, :] - c) ** 2, -1) + 1e-12)
    return -temperature * jax.nn.logsumexp(-d / temperature)


def ndvd_hull_violation(nd, vd, gs, temperature=0.02):
    """Smooth distance outside the catalog's (n_d/0.1, V_d/10) convex hull."""
    import jax

    H = jnp.asarray(gs.hull_ndvd)
    v = H[:, 0] * (nd / 0.1) + H[:, 1] * (vd / 10.0) + H[:, 2]
    return temperature * jax.nn.logsumexp(v / temperature)


def hull_violation(g, gs, temperature=0.02):
    """Smooth distance outside the catalog's convex hull (normalized PCA units).

    Inside the hull every facet distance n.x + b is negative; outside, the largest one
    is the distance to the hull (exact when the nearest point lies on a facet).  A
    soft maximum keeps it differentiable.
    """
    import jax

    H = jnp.asarray(gs.hull)
    v = H[:, :-1] @ g + H[:, -1]
    return temperature * jax.nn.logsumexp(v / temperature)
