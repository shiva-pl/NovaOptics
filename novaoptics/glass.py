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


def catalog_softmin_distance(nd, vd, temperature=0.05):
    """Smooth distance from (nd, vd) to the nearest catalog glass."""
    _, arr = _catalog_arrays()
    d = glass_distance(nd, vd, arr[:, 0], arr[:, 1])
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
