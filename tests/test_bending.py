"""Bending scan: thin-lens conversion and third-order scoring."""

import novaoptics  # noqa: F401  (CPU + x64)

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics import bending as B
from novaoptics.system import Layout, Spec, new_params

SPEC = Spec()


def phase_design():
    lay = Layout(n_elem=3, stop_index=1)
    p = new_params(lay, gaps=[6.0, 0.5, 6.0], bfd=40.0)
    phis = np.array([0.035, -0.06, 0.032])
    p["phase"] = p["phase"].at[:, 0].set(-phis * lay.r_norm**2 / 2)
    p.update(nd=jnp.array([1.80, 1.70, 1.62]), vd=jnp.array([45.0, 28.0, 60.0]))
    return p, lay


@pytest.mark.parametrize("X", [(0.0, 0.0, 0.0), (2.0, -1.0, 1.5), (-3.0, 3.0, 0.5)])
def test_thin_lenses_keep_the_power_of_the_screens(X):
    p, lay = phase_design()
    q = B.thin_lens_system(p, lay, jnp.asarray(X))
    assert float(A.paraxial_efl(q, lay, SPEC)) == pytest.approx(
        float(A.paraxial_efl(p, lay, SPEC, beta=0.0)), rel=1e-6)  # finite-height ray
    c1, c2 = np.asarray(q["c1"]), np.asarray(q["c2"])
    np.testing.assert_allclose((c1 + c2) / (c1 - c2), X, atol=1e-12)


def test_scan_matches_direct_evaluation_and_finds_a_minimum():
    p, lay = phase_design()
    grid = np.linspace(-3, 3, 7)
    combos, merit = B.scan(p, lay, SPEC, grid=grid)
    assert combos.shape == (7**3, 3) and np.all(np.isfinite(merit))
    i = int(np.argmin(merit))
    direct = float(B.third_order_merit(p, lay, SPEC, jnp.asarray(combos[i])))
    assert merit[i] == pytest.approx(direct, rel=1e-9)
    cands = B.candidates(combos, merit, k=3)
    assert cands[0][1] == pytest.approx(merit[i])
    for a, b in zip(cands, cands[1:]):
        assert np.max(np.abs(a[0] - b[0])) >= 1.5


def test_single_lens_prefers_the_classic_minimum_aberration_bending():
    """Stop at the lens, object at infinity: the best thin singlet (n ~ 1.5) is close
    to the textbook minimum-spherical-aberration shape, X ~ +0.7 (strong side to the
    object), not a meniscus pointing the other way."""
    lay = Layout(n_elem=1, stop_index=0)
    p = new_params(lay, gaps=[0.0], bfd=50.0, nd=1.5168, vd=64.17)
    p["phase"] = p["phase"].at[0, 0].set(-(1 / 50.0) * lay.r_norm**2 / 2)
    spec = Spec(half_field_deg=1.0)  # on-axis dominated: spherical aberration
    combos, merit = B.scan(p, lay, spec, grid=np.linspace(-3, 3, 61))
    assert 0.3 < combos[int(np.argmin(merit))][0] < 1.2
