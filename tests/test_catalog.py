import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics import glass as G
from novaoptics.catalog import build_glass_space, read_agf
from novaoptics.merit import make_cfg, residuals
from novaoptics.system import Layout, Spec, describe, new_params, set_glass

AGF = Path(__file__).parents[1] / "schott2025.AGF"
SPEC = Spec()


@pytest.fixture(scope="module")
def glasses():
    return read_agf(AGF)


@pytest.fixture(scope="module")
def space(glasses):
    return build_glass_space(glasses, SPEC.wavelengths, k=3)


def test_agf_dispersion_reproduces_catalog_nd_vd(glasses):
    nd = np.array([g.index(G.LAMBDA_D) for g in glasses])
    assert np.max(np.abs(nd - [g.nd for g in glasses])) < 1e-5
    vd = (nd - 1) / np.array([g.index(G.LAMBDA_F) - g.index(G.LAMBDA_C) for g in glasses])
    assert np.max(np.abs(vd - [g.vd for g in glasses])) < 0.01


def test_three_pca_coordinates_capture_the_catalog(space):
    approx = space.mu + space.coords @ space.B.T
    assert np.max(np.abs(approx - space.X)) < 1e-5


def test_snapped_glass_is_exact(glasses, space):
    lay = Layout(n_elem=2, stop_index=1, glass=space)
    p = new_params(lay, gaps=[3.0, 3.0])
    j = space.find("N-BK7")
    set_glass(p, lay, [1], [j])
    ref = next(g for g in glasses if g.name == "N-BK7")
    for w in SPEC.wavelengths:
        assert float(G.index_at(p, 1, w, 1.0, lay)) == pytest.approx(float(ref.index(w)),
                                                                    abs=1e-12)
    assert float(G.nd_vec(p, lay)[1]) == pytest.approx(ref.nd, abs=1e-5)
    assert float(G.vd_vec(p, lay)[1]) == pytest.approx(ref.vd, abs=0.01)
    assert "N-BK7" in describe(p, lay, SPEC)


def test_beta_scales_dispersion_about_nd(space):
    lay = Layout(n_elem=1, stop_index=0, glass=space)
    p = new_params(lay, gaps=[0.0], nd=1.62, vd=36.0)
    nd = float(G.index_at(p, 0, G.LAMBDA_D, 1.0, lay))
    nF = float(G.index_at(p, 0, G.LAMBDA_F, 1.0, lay))
    nF_half = float(G.index_at(p, 0, G.LAMBDA_F, 0.5, lay))
    assert nF_half - nd == pytest.approx(0.5 * (nF - nd), rel=1e-12)


def test_merit_is_differentiable_in_glass_coordinates(space):
    lay = Layout(n_elem=1, stop_index=0, glass=space)
    p = new_params(lay, gaps=[0.0], bfd=40.0)
    p.update(c1=jnp.array([1 / 40.0]), c2=jnp.array([-1 / 60.0]), t=jnp.array([4.0]))
    cfg = make_cfg(nu=1e-2)

    def f(g0):
        q = dict(p)
        q["g"] = q["g"].at[0, 1].set(g0)
        r = residuals(q, lay, SPEC, cfg)
        return jnp.sum(r**2)

    g0 = float(p["g"][0, 1]) + 0.01  # off the catalog point (distance not at a kink)
    h = 1e-6
    fd = (f(g0 + h) - f(g0 - h)) / (2 * h)
    assert float(jax.grad(f)(g0)) == pytest.approx(float(fd), rel=1e-4)


def test_pca_and_cauchy_agree_on_nd(space):
    """The same singlet in both glass models has nearly the same EFL (same n_d)."""
    lay_p = Layout(n_elem=1, stop_index=0, glass=space)
    p = new_params(lay_p, gaps=[0.0], bfd=40.0)
    set_glass(p, lay_p, [0], [space.find("N-BK7")])
    p.update(c1=jnp.array([1 / 40.0]), c2=jnp.array([-1 / 60.0]), t=jnp.array([4.0]))
    lay_c = Layout(n_elem=1, stop_index=0)
    q = dict(p, nd=jnp.array([1.5168]), vd=jnp.array([64.17]))
    assert float(A.paraxial_efl(p, lay_p, SPEC)) == pytest.approx(
        float(A.paraxial_efl(q, lay_c, SPEC)), rel=1e-5)


def test_normal_glass_model_keeps_nd_vd_and_beats_cauchy(space):
    """Continuous (n_d, V_d) glasses with the catalog's normal-glass dispersion."""
    lay = Layout(n_elem=1, stop_index=0, region=space)
    errs_normal, errs_cauchy = [], []
    for j in range(0, len(space.names), 7):
        x = space.X[j]
        nd = x[space.grid_index(G.LAMBDA_D)]
        vd = (nd - 1) / (x[space.grid_index(G.LAMBDA_F)] - x[space.grid_index(G.LAMBDA_C)])
        p = {"nd": jnp.array([nd]), "vd": jnp.array([vd])}
        n = np.array([float(G.index_at(p, 0, w, 1.0, lay)) for w in space.grid])
        c = np.array([float(G.index(nd, vd, w, 1.0)) for w in space.grid])
        assert n[space.grid_index(G.LAMBDA_D)] == pytest.approx(nd, abs=1e-12)
        vfit = (nd - 1) / (n[space.grid_index(G.LAMBDA_F)] - n[space.grid_index(G.LAMBDA_C)])
        assert vfit == pytest.approx(vd, rel=1e-9)
        errs_normal.append(np.abs(n - x).max())
        errs_cauchy.append(np.abs(c - x).max())
    assert np.sqrt(np.mean(np.square(errs_normal))) < 0.2 * np.sqrt(np.mean(np.square(errs_cauchy)))
