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


def disp(name):
    """Dispersion data of a catalog glass as a parameter array: [formula, c0 .. c5]."""
    import numpy as np

    _ensure(name)
    if name not in _REGISTRY:
        raise KeyError(f"unknown material {name!r}")
    formula, c = _REGISTRY[name]
    return np.array([float(formula), *(list(c[:6]) + [0.0] * (6 - len(c[:6])))])


def kind(name):
    """Medium kind: 'air', 'model' or 'catalog' (what the compiled programs depend on)."""
    if name in AIR:
        return "air"
    return "model" if name == "model" else "catalog"


def _index_from_data(d, wl_um):
    """AGF formula 1 (Schott) or 2 (Sellmeier 1) from data, selected without branching
    (so that every catalog glass shares one compiled program)."""
    L2 = jnp.asarray(wl_um, float) ** 2
    f, c = d[0], d[1:7]
    sell = f > 1.5
    den = lambda a: jnp.where(sell, L2 - a, 1.0)  # keep the unused branch finite
    n2_sell = 1.0 + c[0] * L2 / den(c[1]) + c[2] * L2 / den(c[3]) + c[4] * L2 / den(c[5])
    n2_schott = c[0] + c[1] * L2 + c[2] / L2 + c[3] / L2**2 + c[4] / L2**3 + c[5] / L2**4
    return jnp.sqrt(jnp.where(sell, n2_sell, n2_schott))


def index(name: str, wl_um, sp=None):
    """Refractive index of medium `name` at wavelength `wl_um` [um] (JAX-traceable).

    `sp`: the surface parameter dict: sp["nd"], sp["vd"] for name == "model", and the
    dispersion data sp["disp"] of a catalog glass (used when present, so that the glass
    is data, not part of the compiled program).
    """
    if name in AIR:
        return jnp.ones_like(jnp.asarray(wl_um, float))
    if name == "model":
        return G.index(sp["nd"], sp["vd"], wl_um)
    if sp is not None:
        if "disp" not in sp:  # names are not part of the compiled program: data needed
            raise KeyError(f"surface with catalog glass {name!r} has no 'disp' data")
        return _index_from_data(sp["disp"], wl_um)
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
