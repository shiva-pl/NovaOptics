"""Diffraction analysis against closed forms: Airy pattern, diffraction-limited MTF,
Strehl of pure defocus, and PSF-vs-ray centroid consistency."""

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics.engine import Builder, Spec
from novaoptics.engine import analysis as EA

WL = 0.5875618
F, D = 100.0, 20.0  # f/5 parabola


def _parabola(defocus=0.0):
    b = Builder()
    b.mirror(c=-1 / (2 * F), k=-1.0, stop=True, t=-F - defocus)
    b.image()
    sysm, prm = b.build()
    return sysm, prm, Spec(epd=D, fields=((0.0, 0.0),), wavelengths=(WL,))


def test_airy_first_zero_and_strehl():
    sysm, prm, spec = _parabola()
    p = EA.psf(prm, sysm, spec, (0.0, 0.0), n=64, pad=8)
    assert p["strehl"] == pytest.approx(1.0, abs=1e-9)
    N = p["I"].shape[0]
    cut = p["I"][N // 2, N // 2:]
    x = p["x_um"][N // 2:]
    i = np.argmax(np.diff(cut) > 0)  # first minimum
    r0 = 1.22 * WL * F / D
    assert x[i] == pytest.approx(r0, abs=p["dx_um"])


def test_diffraction_limited_mtf():
    sysm, prm, spec = _parabola()
    m = EA.mtf(prm, sysm, spec, (0.0, 0.0), n=96, pad=4)
    nu_c = D / (WL * 1e-3 * F)
    nu = m["fx"][m["fx"] < 0.9 * nu_c]
    phi = np.arccos(nu / nu_c)
    exact = 2 / np.pi * (phi - np.cos(phi) * np.sin(phi))
    assert np.max(np.abs(m["mtf_x"][: nu.size] - exact)) < 0.02
    assert np.max(np.abs(m["mtf_y"][: nu.size] - exact)) < 0.02


def test_defocus_strehl():
    """Quarter-wave PV defocus: S = sinc^2(1/4) = 0.8106 exactly (uniform pupil)."""
    dz = 0.25 * WL * 1e-3 * 8 * (F / D) ** 2
    sysm, prm, spec = _parabola(defocus=dz)
    o = EA.opd_map(prm, sysm, spec, (0.0, 0.0), n=65)
    assert o["pv"] == pytest.approx(0.25, rel=0.02)
    p = EA.psf(prm, sysm, spec, (0.0, 0.0), n=128, pad=4)
    assert p["strehl"] == pytest.approx(0.8106, abs=0.01)


def test_psf_centroid_matches_ray_centroid():
    """Off-axis field of a parabola (coma): the Fraunhofer PSF centroid equals the ray
    centroid (to the accuracy of the linear direction-cosine mapping)."""
    sysm, prm, spec = _parabola()
    field = (0.3, -0.5)
    sp = EA.spot_diagram(prm, sysm, spec, field, rings=30)
    p = EA.psf(prm, sysm, spec, field, n=128, pad=4)
    X, Y = np.meshgrid(p["x_um"], p["y_um"])
    c = np.array([(p["I"] * X).sum(), (p["I"] * Y).sum()])
    assert np.all(np.sign(c) == np.sign(sp["centroid"]))
    np.testing.assert_allclose(c, sp["centroid"], rtol=0.06)


def test_spot_rms_of_perfect_system_is_zero():
    sysm, prm, spec = _parabola()
    s = EA.spot_diagram(prm, sysm, spec, (0.0, 0.0))
    assert s["rms_um"] < 1e-6
    assert s["fno_working"] == pytest.approx(F / D, rel=0.01)
    assert s["airy_um"] == pytest.approx(1.22 * WL * F / D, rel=0.01)
