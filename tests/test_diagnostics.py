"""Paraxial Seidel sums must agree with real-ray wavefront fits in the third-order
limit (slow beam, small field), for every kind of element the tracer supports."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import diagnostics as D
from novaoptics.system import Layout, Spec, new_params

SLOW = Spec(fnum=40.0, half_field_deg=2.0, efl=50.0)
LAY1 = Layout(n_elem=1, stop_index=0, aspheres=True)
TERMS = ("W040", "W131", "W222", "W220")


def singlet(c1=0.0, c2=0.0, t=0.0, gap=10.0, asph4=0.0, phase2=0.0, phase4=0.0):
    p = new_params(LAY1, gaps=[gap], bfd=45.0, nd=1.5168, vd=64.17)
    p.update(c1=jnp.array([c1]), c2=jnp.array([c2]), t=jnp.array([t]))
    p["asph"] = p["asph"].at[0, 0].set(asph4)
    p["phase"] = p["phase"].at[0, 0].set(phase2).at[0, 1].set(phase4)
    return p, LAY1


def cooke():
    lay = Layout(n_elem=3, stop_index=1)
    p = new_params(lay, gaps=[2.3207, 0.2368, 3.6190], bfd=38.764)
    p.update(c1=jnp.array([0.048007, -0.019118, 0.017153]),
             c2=jnp.array([0.019256, 0.047564, -0.030932]),
             t=jnp.array([8.0001, 1.9997, 2.6670]),
             nd=jnp.array([1.883, 1.8467, 1.883]), vd=jnp.array([40.76, 23.78, 40.76]))
    return p, lay


PH = -LAY1.r_norm**2 / 100.0  # phase power of 1/50 mm^-1
CASES = {
    "spherical singlet, remote stop": singlet(c1=1 / 30, c2=-1 / 80, t=4.0),
    "aspheric singlet": singlet(c1=1 / 30, c2=-1 / 80, t=4.0, asph4=0.05),
    "phase screen, flat substrate": singlet(phase2=PH, phase4=0.02),
    "phase screen, curved substrate": singlet(c2=-1 / 60, phase2=PH, phase4=0.02),
    "phase screen at the stop": singlet(c2=-1 / 60, phase2=PH, gap=0.0),
    "three-element triplet": cooke(),
}


@pytest.mark.parametrize("name", list(CASES))
def test_seidel_matches_real_rays(name):
    p, lay = CASES[name]
    p = dict(p)
    p["bfd"] = jnp.asarray(D.seidel(p, lay, SLOW)["bfd_paraxial"])  # image at paraxial focus
    _, Wt = D.seidel_waves(D.seidel(p, lay, SLOW), SLOW)
    fit = D.aberration_fit(p, lay, SLOW, n_field=7)
    c = {n: fit["coef"][:, k] for k, n in enumerate(fit["names"])}
    scale = max(np.max(np.abs(Wt[:4])), 1e-4)
    for k, term in enumerate(TERMS):
        assert abs(Wt[k] - c[term][1]) < 0.02 * scale, (term, Wt[k], c[term][1])
    # exact paraxial color (all orders in dn) vs F - C of the fitted defocus / tilt
    sd = D.seidel(p, lay, SLOW)
    assert c["W020"][0] - c["W020"][2] == pytest.approx(sd["W020_exact"], rel=0.01, abs=1e-5)
    assert c["W111"][0] - c["W111"][2] == pytest.approx(sd["W111_exact"], rel=0.01, abs=1e-5)


@pytest.mark.parametrize("name", list(CASES))
def test_primary_color_is_the_linear_term(name):
    """C_I, C_II are first order in dn: they match the exact color for weak dispersion."""
    p, lay = CASES[name]
    beta = 0.02
    sd = D.seidel(p, lay, SLOW, beta=beta)
    _, Wt = D.seidel_waves(sd, SLOW)
    assert Wt[5] == pytest.approx(sd["W020_exact"], rel=0.01, abs=1e-7)
    assert Wt[6] == pytest.approx(sd["W111_exact"], rel=0.01, abs=1e-7)


def test_phase_screen_has_no_petzval():
    """Sweatt model: Petzval contribution phi/N -> 0."""
    p, lay = singlet(phase2=PH)
    sd = D.seidel(p, lay, SLOW)
    phi = 1 / 50.0
    assert abs(sd["total"][3]) < 2 * sd["lagrange"] ** 2 * phi / D.SWEATT_N


def test_thin_lens_petzval_is_phi_over_n():
    n = 1.5168
    p, lay = singlet(c1=1 / 40, c2=-1 / 40)
    sd = D.seidel(p, lay, SLOW)
    phi = (n - 1) * (2 / 40)
    assert sd["total"][3] == pytest.approx(sd["lagrange"] ** 2 * phi / n, rel=1e-9)


def test_paraxial_efl_matches_real_ray_efl():
    from novaoptics import analysis as A

    p, lay = cooke()
    assert D.seidel(p, lay, SLOW)["efl"] == pytest.approx(
        float(A.paraxial_efl(p, lay, SLOW)), rel=1e-6)
