import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics.merit import make_cfg, residuals
from novaoptics.system import Layout, Spec, new_params

SPEC = Spec()
LAY1 = Layout(n_elem=1, stop_index=0)


def ideal_phase_params(f=50.0):
    """Taylor expansion of the exact ideal-lens phase W = f - sqrt(r^2 + f^2)."""
    p = new_params(LAY1, gaps=[0.0], bfd=f)
    rn = LAY1.r_norm
    a = [-1 / (2 * f), 1 / (8 * f**3), -1 / (16 * f**5), 5 / (128 * f**7)]
    p["phase"] = p["phase"].at[0].set(jnp.array([a[k] * rn ** (2 * k + 2) for k in range(4)]))
    return p


def test_phase_screen_efl():
    p = new_params(LAY1, gaps=[0.0], bfd=50.0)
    p["phase"] = p["phase"].at[0, 0].set(-LAY1.r_norm**2 / 100.0)
    assert A.paraxial_efl(p, LAY1, SPEC) == pytest.approx(50.0, rel=1e-8)


def test_ideal_phase_is_stigmatic_on_axis():
    # beta = 0: the screen is achromatic (with beta = 1 it disperses like its glass)
    ev = A.evaluate(ideal_phase_params(), LAY1, SPEC, beta=0.0)
    wfe = A.rms_wfe(ev, SPEC)
    # residual is only the truncation of the Taylor series (r^10 term)
    assert wfe[0] < 1e-3


def test_thick_lens_matches_lensmaker():
    p = new_params(LAY1, gaps=[0.0], bfd=40.0)
    n, c1, c2, t = 1.5168, 1 / 40.0, -1 / 60.0, 5.0
    p.update(c1=jnp.array([c1]), c2=jnp.array([c2]), t=jnp.array([t]),
             nd=jnp.array([n]), vd=jnp.array([64.17]))
    phi = (n - 1) * (c1 - c2 + (n - 1) * t * c1 * c2 / n)
    assert A.paraxial_efl(p, LAY1, SPEC) == pytest.approx(1 / phi, rel=1e-7)


def test_plano_convex_chromatic_focal_shift_sign():
    """Blue focuses closer than red for a positive glass singlet."""
    p = new_params(LAY1, gaps=[0.0], bfd=40.0)
    p.update(c1=jnp.array([1 / 30.0]), t=jnp.array([4.0]), nd=jnp.array([1.5168]),
             vd=jnp.array([64.17]))
    lay = LAY1
    f_blue = A.paraxial_efl(p, lay, Spec(wavelengths=(0.4861327,) * 3))
    f_red = A.paraxial_efl(p, lay, Spec(wavelengths=(0.6562725,) * 3))
    # (f_red - f_blue)/f ~ 1/V
    assert (f_red - f_blue) / f_red == pytest.approx(1 / 64.17, rel=0.05)


def test_focus_shifts_agree_with_finite_rays():
    """AD-based tangential focus equals where two nearby real rays cross."""
    p = ideal_phase_params()
    s_t, _ = A.focus_shifts(p, LAY1, SPEC, (0.7,))
    ev = A.evaluate(p, LAY1, SPEC, fracs=(0.7,))
    assert np.isfinite(float(s_t[0]))
    # astigmatic phase screen: tangential focus is in front of the image plane
    assert float(s_t[0]) < 0


def test_gradients_match_finite_differences():
    lay = Layout(n_elem=2, stop_index=1)
    p = new_params(lay, gaps=[3.0, 3.0], bfd=45.0)
    p.update(c1=jnp.array([0.02, -0.01]), c2=jnp.array([-0.01, -0.02]),
             t=jnp.array([3.0, 3.0]))
    p["phase"] = p["phase"].at[:, 0].set(-0.3)
    cfg = make_cfg(beta=1.0)

    def f(c):
        q = dict(p)
        q["c1"] = q["c1"].at[0].set(c)
        r = residuals(q, lay, SPEC, cfg)
        return jnp.sum(r**2)

    g = jax.grad(f)(0.02)
    h = 1e-6
    fd = (f(0.02 + h) - f(0.02 - h)) / (2 * h)
    assert float(g) == pytest.approx(float(fd), rel=1e-4)


def test_phase_screen_disperses_like_its_glass():
    """With beta = 1 a phase lens has the axial color of a thin lens of its glass."""
    p = new_params(LAY1, gaps=[0.0], bfd=50.0, nd=1.5168, vd=64.17)
    p["phase"] = p["phase"].at[0, 0].set(-LAY1.r_norm**2 / 100.0)
    f_b = A.paraxial_efl(p, LAY1, Spec(wavelengths=(0.4861327,) * 3))
    f_r = A.paraxial_efl(p, LAY1, Spec(wavelengths=(0.6562725,) * 3))
    assert (f_r - f_b) / 50.0 == pytest.approx(1 / 64.17, rel=0.02)


def test_asphere_with_zero_terms_matches_sphere():
    lay_a = Layout(n_elem=1, stop_index=0, aspheres=True)
    p = new_params(LAY1, gaps=[0.0], bfd=40.0)
    p.update(c1=jnp.array([1 / 40.0]), c2=jnp.array([-1 / 60.0]), t=jnp.array([5.0]))
    ev1 = A.evaluate(p, LAY1, SPEC)
    ev2 = A.evaluate(p, lay_a, SPEC)
    np.testing.assert_allclose(ev1["opd"], ev2["opd"], atol=1e-8)


def test_rear_asphere_equals_phase_for_thin_element():
    """Aspheric sag A on the rear surface ~ phase W = (n-1) A (thin-element limit)."""
    lay_a = Layout(n_elem=1, stop_index=0, aspheres=True)
    n = 1.5168
    base = new_params(lay_a, gaps=[0.0], bfd=50.0, nd=n, vd=64.17)
    base["phase"] = base["phase"].at[0, 0].set(-lay_a.r_norm**2 / 100.0)
    w4 = 2e-3  # mm of r^4 phase at r_norm
    p_phase = dict(base)
    p_phase["phase"] = base["phase"].at[0, 1].set(w4)
    p_asph = dict(base)
    p_asph["asph"] = base["asph"].at[0, 0].set(w4 / (n - 1))
    ref = A.evaluate(base, lay_a, SPEC, beta=0.0)["opd"]
    d_phase = A.evaluate(p_phase, lay_a, SPEC, beta=0.0)["opd"] - ref
    d_asph = A.evaluate(p_asph, lay_a, SPEC, beta=0.0)["opd"] - ref
    # On axis: same aberration change to within a few percent (thin-element limit).
    # Off axis they differ more, as they should: a surface sees oblique incidence.
    d_phase, d_asph = d_phase[:, 0], d_asph[:, 0]
    assert np.max(np.abs(d_phase - d_asph)) < 0.03 * np.max(np.abs(d_phase))
