"""Growing designs: first-order-preserving splits and the scalable glass ranking."""

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics.continuation import _rank_combos
from novaoptics.grow import split_element
from novaoptics.system import Layout, Spec, new_params, vertex_positions


def _screens(powers, gaps, stop_index=1, bfd=40.0):
    lay = Layout(n_elem=len(powers), stop_index=stop_index)
    p = new_params(lay, gaps=gaps, bfd=bfd)
    p["phase"] = jnp.zeros((lay.n_elem, lay.n_phase)).at[:, 0].set(
        -np.asarray(powers) * lay.r_norm**2 / 2)
    return p, lay


@pytest.mark.parametrize("i", [0, 1, 2])
def test_split_preserves_first_order(i):
    spec = Spec()
    p, lay = _screens([0.035, -0.06, 0.03], gaps=[0.0, 7.5, 18.5])
    efl = float(A.paraxial_efl(p, lay, spec, beta=0.0))
    z_img = float(vertex_positions(p, lay)["image"])
    q, lq, (a, b) = split_element(p, lay, i, 2.5)
    assert lq.n_elem == 4 and b == a + 1
    assert float(A.paraxial_efl(q, lq, spec, beta=0.0)) == pytest.approx(efl, rel=2e-3)
    # the image stays where it was; splitting the first screen puts a half 1.25 mm in
    # front of the old origin, so coordinates measured from the new first item grow by that
    origin_shift = 1.25 if i == 0 else 0.0
    assert float(vertex_positions(q, lq)["image"]) - origin_shift == pytest.approx(z_img,
                                                                                   abs=1e-9)
    # halves carry half of the aspheric phase and (nearly) half of the power
    assert float(q["phase"][a, 0]) == pytest.approx(float(p["phase"][i, 0]) / 2, rel=0.05)


def test_split_in_the_stop_plane_straddles_the_stop():
    p, lay = _screens([0.035, -0.06, 0.03], gaps=[0.0, 7.5, 18.5])  # E0 in the stop plane
    q, lq, (a, b) = split_element(p, lay, 0, 2.5)
    assert lq.items() == [0, "stop", 1, 2, 3]
    np.testing.assert_allclose(np.asarray(q["gaps"])[:2], [1.25, 1.25])


def test_split_refuses_glass():
    p, lay = _screens([0.035, -0.06, 0.03], gaps=[0.0, 7.5, 18.5])
    p["t"] = p["t"].at[1].set(3.0)
    with pytest.raises(ValueError):
        split_element(p, lay, 1, 2.5)


def test_beam_search_finds_the_exhaustive_optimum():
    rng = np.random.default_rng(1)
    n, k, R, G = 4, 3, 60, 30
    J = rng.standard_normal((R, n * k))
    r0 = rng.standard_normal(R)
    C = rng.standard_normal((G, k))
    g0 = rng.standard_normal((n, k)) * 0.1
    top = [list(rng.choice(G, 6, replace=False)) for _ in range(n)]
    ex, pe = _rank_combos(r0, J, C, g0, top, k)
    bm, pb = _rank_combos(r0, J, C, g0, top, k, exhaustive=10, beam=200)
    assert bm[0] == ex[0]
    assert pb[0] == pytest.approx(pe[0])
