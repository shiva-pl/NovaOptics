"""Media for the surface engine: air, catalog glasses (AGF dispersion formulas in JAX)
and a "model" glass from (n_d, V_d) design variables.

Indices are relative to air (as in most lens design programs).  Catalog glasses are
looked up by name in a registry; the repository's SCHOTT catalog is loaded on first
use when a name is not registered yet.
"""

from __future__ import annotations

import threading
from pathlib import Path

import jax.numpy as jnp

from .. import glass as G

_REGISTRY: dict = {}  # name -> (formula, coefficients)
DEFAULT_AGF = Path(__file__).resolve().parents[2] / "schott2025.AGF"
AIR = ("air", "", "vacuum")


def register(glasses):
    """Register catalog.Glass objects (e.g. from catalog.read_agf) by name."""
    for g in glasses:
        _REGISTRY[g.name] = (g.formula, tuple(float(c) for c in g.coef))


def register_agf(path):
    from ..catalog import read_agf

    register(read_agf(path))


def known(name) -> bool:
    _ensure(name)
    return name in AIR or name == "model" or name in _REGISTRY


_LOAD_LOCK = threading.Lock()


def _ensure(name):
    if name in AIR or name == "model" or name in _REGISTRY:
        return
    with _LOAD_LOCK:  # threads may trace systems concurrently
        if DEFAULT_AGF.exists() and not getattr(_ensure, "loaded", False):
            register_agf(DEFAULT_AGF)
            _ensure.loaded = True  # only once every glass is registered


def index(name: str, wl_um, sp=None):
    """Refractive index of medium `name` at wavelength `wl_um` [um] (JAX-traceable).

    `sp`: the surface parameter dict, for name == "model" (uses sp["nd"], sp["vd"]).
    """
    if name in AIR:
        return jnp.ones_like(jnp.asarray(wl_um, float))
    if name == "model":
        return G.index(sp["nd"], sp["vd"], wl_um)
    _ensure(name)
    if name not in _REGISTRY:
        raise KeyError(f"unknown material {name!r}")
    formula, c = _REGISTRY[name]
    L2 = jnp.asarray(wl_um, float) ** 2
    if formula == 2:  # Sellmeier 1
        n2 = 1.0 + c[0] * L2 / (L2 - c[1]) + c[2] * L2 / (L2 - c[3]) + c[4] * L2 / (L2 - c[5])
    elif formula == 1:  # Schott
        n2 = c[0] + c[1] * L2 + c[2] / L2 + c[3] / L2**2 + c[4] / L2**3 + c[5] / L2**4
    else:
        raise NotImplementedError(f"AGF dispersion formula {formula} ({name})")
    return jnp.sqrt(n2)
