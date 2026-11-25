"""Bending scan: choose element shapes before the morph, with finite-index theory.

A phase screen carries only an element's net optical path; how a glass element is
bent (biconvex, meniscus, ...) is invisible in phase space, because ideal screens act
like lenses of infinite index and thin-lens shape dependence vanishes as n -> inf.
Left to itself, the morph picks a bending basin by chance early on, when bending
barely matters, and cannot leave it later (one phase design, eight starting
bendings: 0.10 to 0.31 waves after the full morph).

So the bending is chosen deliberately where finite-index refraction exists: each
element becomes a thin glass lens with its phase-space power and real index (and, with
aspheres, its aspheric phase handed to the rear asphere as the morph would), and every
combination of shape factors X = (c1 + c2)/(c1 - c2) on a grid is scored by
third-order aberrations, each weighted by its RMS after balancing with defocus.  Thin-
lens primary color does not depend on bending, so color is not scored.  The best few
distinct combinations then seed full morphs.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from .diagnostics import _seidel_core, _to_w
from .glass import nd_vec
from .system import Layout, Spec

# RMS of each third-order term after optimal balancing with defocus (and tilt):
# W040: 1/(6 sqrt5), W131: 1/(6 sqrt2), W222: 1/(2 sqrt6); field curvature W220 is
# weighted like a defocus (1/(2 sqrt3)), halved since focus can split it across field.
RMS_WEIGHTS = np.array([1 / (6 * np.sqrt(5)), 1 / (6 * np.sqrt(2)), 1 / (2 * np.sqrt(6)),
                        0.5 / (2 * np.sqrt(3))])


def thin_lens_system(p, lay: Layout, X):
    """Phase design -> thin glass lenses with shape factors X (array, one per element)."""
    nd = nd_vec(p, lay)
    phi = -2.0 * p["phase"][:, 0] / lay.r_norm**2
    dc = phi / (nd - 1.0)
    q = dict(p)
    q["c1"] = 0.5 * (1.0 + X) * dc
    q["c2"] = 0.5 * (X - 1.0) * dc
    q["t"] = jnp.zeros(lay.n_elem)
    if lay.aspheres:
        q["asph"] = p["phase"][:, 1:] / (nd[:, None] - 1.0)
    q["phase"] = jnp.zeros_like(p["phase"])
    if lay.two_screens:
        q["phase1"] = jnp.zeros_like(p["phase1"])
    return q


def third_order_merit(p, lay: Layout, spec: Spec, X):
    q = thin_lens_system(p, lay, X)
    _, S, _ = _seidel_core(q, lay, spec, 1.0)
    W = _to_w(S.sum(0), spec.ref_wavelength * 1e-3)  # W040 W131 W222 W220 ...
    terms = jnp.stack([W[0], W[1], W[2], W[3]]) * jnp.asarray(RMS_WEIGHTS)
    return jnp.sum(terms**2)


def scan(p, lay: Layout, spec: Spec, grid=np.linspace(-3.5, 3.5, 15)):
    """Third-order merit for every combination of shape factors on `grid`."""
    combos = np.array(list(itertools.product(grid, repeat=lay.n_elem)))
    f = jax.jit(jax.vmap(lambda X: third_order_merit(p, lay, spec, X)))
    merit = np.asarray(f(jnp.asarray(combos)))
    merit = np.where(np.isfinite(merit), merit, np.inf)
    return combos, merit


def candidates(combos, merit, k=3, min_dist=1.5):
    """The k best combinations that are at least `min_dist` apart (max-norm in X)."""
    out = []
    for i in np.argsort(merit):
        if not np.isfinite(merit[i]):
            break
        if all(np.max(np.abs(combos[i] - combos[j])) >= min_dist for j in out):
            out.append(i)
        if len(out) == k:
            break
    return [(combos[i], float(merit[i])) for i in out]
