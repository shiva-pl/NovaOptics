"""Surface engine: agreement with the element tracer, mirrors, off-axis apertures,
coordinate breaks, gratings, Q-type surfaces and gradients."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics.engine import Builder, Spec, bundle, focal_length, trace_ray
from novaoptics import glass as G
from novaoptics.engine import qpoly
from novaoptics.engine import trace as T

RESULTS = Path(__file__).parents[1] / "examples" / "output" / "results.json"
WL = 0.5875618


def _pupil(n=7):
    g = np.linspace(-1, 1, n)
    x, y = np.meshgrid(g, g)
    m = x**2 + y**2 <= 1
    return jnp.asarray(x[m]), jnp.asarray(y[m])


def _rms_opd_waves(prm, sysm, spec, field, wl=WL):
    px, py = _pupil()
    b = bundle(prm, sysm, spec, field, wl, px, py)
    w = b["opd"] / (wl * 1e-3)
    return jnp.sqrt(jnp.mean((w - jnp.mean(w)) ** 2)), b


@pytest.mark.skipif(not RESULTS.exists(), reason="example results not present")
def test_matches_element_tracer():
    from novaoptics.engine import convert
    from novaoptics.gui import runner as R
    from novaoptics.trace import aim_chief, launch, trace_ray as old

    for row in json.loads(RESULTS.read_text()):
        d = R.from_json(row)
        sysm, prm, es = convert.from_layout(d.params, d.layout, d.spec.epd, [0, 20])
        for th in (0.0, 20.0):
            y0 = aim_chief(d.params, d.layout, jnp.deg2rad(th), d.spec.ref_wavelength)
            xy = T.aim_chief(prm, sysm, (0.0, th), d.spec.ref_wavelength)
            for px, py in [(0, 0), (0, 1), (0.6, -0.5)]:
                o, dd = launch(jnp.deg2rad(th), y0, px, py, d.spec.epd / 2, d.layout)
                a = old(d.params, d.layout, G.LAMBDA_F, 1.0, o, dd)["img"]
                o, dd = T.launch(xy, (0.0, th), px, py, d.spec.epd, sysm)
                b = T.trace_ray(prm, sysm, G.LAMBDA_F, o, dd)["pts"][-1]
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)
        assert float(focal_length(prm, sysm, es)) == pytest.approx(50.0, abs=1e-3)


def _parabola(f=100.0, offset=0.0, epd=20.0):
    b = Builder()
    b.mirror(c=-1 / (2 * f), k=-1.0, stop=True, t=-f, sy=offset)
    b.image()
    sysm, prm = b.build()
    return sysm, prm, Spec(epd=epd, fields=((0.0, 0.0),), wavelengths=(WL,))


@pytest.mark.parametrize("offset", [0.0, 30.0])
def test_parabola_on_axis_is_perfect(offset):
    """Also off-axis sections (decentered stop on the parent's axis)."""
    sysm, prm, spec = _parabola(offset=offset)
    rms, b = _rms_opd_waves(prm, sysm, spec, (0.0, 0.0))
    assert float(rms) < 1e-9
    assert float(jnp.max(jnp.abs(b["img"]))) < 1e-9
    assert bool(jnp.all(b["ok"]))
    # local focal length: mirror-to-focus distance f + y^2 / (4 f)
    L = 100.0 + offset**2 / 400.0
    assert float(focal_length(prm, sysm, spec)) == pytest.approx(L, rel=1e-6)
    assert float(focal_length(prm, sysm, spec, axis=0)) == pytest.approx(L, rel=1e-6)


def test_sphere_mirror_has_spherical_aberration():
    sysm, prm, spec = _parabola()
    prm[0]["k"] = jnp.asarray(0.0)
    # paraxial focus R/2: W040 = h^4 / (4 R^3) per reflection (approximately)
    rms, _ = _rms_opd_waves(prm, sysm, spec, (0.0, 0.0))
    assert float(rms) > 0.01


def test_fold_mirror_with_coordinate_breaks():
    """45 deg fold: the axis turns by 90 deg; a perfect lens behind it stays perfect."""
    b = Builder()
    b.stop(t=10.0)
    b.cb(tx=45.0)
    b.mirror(t=0.0)
    b.cb(tx=45.0, t=-30.0)  # follow the reflected axis; distance is negative after a mirror
    b.add(phase="ideal", pcoef=[-(10.0**2) / (2 * 50.0)], r_norm=10.0, t=-50.0,
          interact="refract")
    b.image()
    sysm, prm = b.build()
    spec = Spec(epd=4.0, fields=((0.0, 0.0),), wavelengths=(WL,))
    px, py = _pupil()
    r = bundle(prm, sysm, spec, (0.0, 0.0), WL, jnp.r_[0.0, px], jnp.r_[0.0, py])
    pts = r["pts"][0]  # chief ray
    np.testing.assert_allclose(np.asarray(pts[2]), [0, 0, 10], atol=1e-12)  # mirror
    # after the fold the beam travels along +y (right-handed tilt of +45 deg about x)
    np.testing.assert_allclose(np.asarray(pts[4]), [0, 30, 10], atol=1e-9)
    np.testing.assert_allclose(np.asarray(r["dir"][0]), [0, 1, 0], atol=1e-12)
    assert float(focal_length(prm, sysm, spec)) == pytest.approx(50.0, rel=1e-3)
    assert float(jnp.max(jnp.abs(r["img"]))) < 2e-3  # thin ideal lens: small residuals


@pytest.mark.parametrize("order", [1, -1, 2])
def test_grating_equation(order):
    freq = 600.0  # lines / mm
    b = Builder()
    b.stop(t=5.0)
    b.add(phase="grating", order=order, freq=freq, t=10.0)
    b.image()
    sysm, prm = b.build()
    o = jnp.array([0.0, 0.0, -20.0])
    d = jnp.array([0.0, 0.0, 1.0])
    r = trace_ray(prm, sysm, WL, o, d)
    assert float(r["dir"][1]) == pytest.approx(order * WL * 1e-3 * freq, rel=1e-12)
    assert bool(r["ok"])
    # beyond cutoff: evanescent order is flagged
    prm[1]["freq"] = jnp.asarray(2000.0)
    if abs(order) >= 1:
        assert not bool(trace_ray(prm, sysm, WL, o, d)["ok"])


@pytest.mark.parametrize("shape,coef", [("qcon", [2e-3, -1e-3, 5e-4]),
                                        ("qbfs", [5e-3, 2e-3, -1e-3, 4e-4]),
                                        ("even", [1e-3, -2e-4])])
def test_aspheric_intersection_lies_on_surface(shape, coef):
    b = Builder()
    b.stop(t=0.0)
    i = b.add(shape=shape, coef=coef, c=1 / 40, k=-0.6, r_norm=8.0, material="N-BK7", t=6.0)
    b.add(c=-1 / 60, t=40.0)
    b.image()
    sysm, prm = b.build()
    o = jnp.array([0.0, 7.0, -20.0])
    d = jnp.array([0.0, -0.08, 1.0])
    d = d / jnp.linalg.norm(d)
    r = trace_ray(prm, sysm, WL, o, d)
    q = r["local"][i]
    z = T.sag(sysm.surfaces[i], prm[i], q[0], q[1])
    assert abs(float(q[2] - z)) < 1e-12


def test_qbfs_departure_vanishes_at_center_and_edge():
    from novaoptics.engine.system import Surface, surface_params

    s = Surface(shape="qbfs", n_terms=4, r_norm=10.0)
    p = surface_params(s, c=1 / 50, coef=[1e-2, 3e-3, -2e-3, 1e-3])
    sph = surface_params(Surface(), c=1 / 50)
    for r in (0.0, 10.0):
        assert float(T.sag(s, p, 0.0, r) - T.sag(Surface(), sph, 0.0, r)) == pytest.approx(
            0.0, abs=1e-15)
    assert float(T.sag(s, p, 0.0, 5.0) - T.sag(Surface(), sph, 0.0, 5.0)) != 0.0


def test_qbfs_published_values():
    x = np.linspace(0, 1, 5)
    Q = np.asarray(qpoly.qbfs(jnp.asarray(x), 5))
    np.testing.assert_allclose(Q[1], (13 - 16 * x) / np.sqrt(19), atol=1e-12)
    np.testing.assert_allclose(Q[4], (7737 - 16 * x * (4653 - 2 * x * (7381 - 8 * x * (
        1168 - 509 * x)))) / (3 * np.sqrt(131831)), atol=1e-12)


def test_gradient_through_aiming_matches_finite_differences():
    """Off-axis field, stop behind a lens: d(RMS OPD)/d(curvature) and d/d(stop sy)."""
    b = Builder()
    b.add(c=1 / 30, material="N-BK7", t=5.0)
    b.add(c=-1 / 80, t=4.0)
    b.stop(t=44.0)
    b.image()
    sysm, prm = b.build()
    spec = Spec(epd=8.0, fields=((3.0, 8.0),), wavelengths=(WL,))

    def f(c, sy):
        q = [dict(p) for p in prm]
        q[0]["c"] = c
        q[2]["sy"] = sy
        return _rms_opd_waves(q, sysm, spec, (3.0, 8.0))[0]

    g = jax.grad(f, argnums=(0, 1))(jnp.asarray(1 / 30), jnp.asarray(0.5))
    h = 1e-6
    fd_c = (f(1 / 30 + h, 0.5) - f(1 / 30 - h, 0.5)) / (2 * h)
    fd_s = (f(1 / 30, 0.5 + h) - f(1 / 30, 0.5 - h)) / (2 * h)
    assert float(g[0]) == pytest.approx(float(fd_c), rel=1e-5)
    assert float(g[1]) == pytest.approx(float(fd_s), rel=1e-5)
