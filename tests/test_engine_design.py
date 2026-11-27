"""Engine design model: exactly continuous insertions, multi-configuration
(per-wavelength focus, zoom), the optimizer on engine systems."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics.engine import Builder, Spec
from novaoptics.engine import design as D
from novaoptics.engine import merit as EM

VIS = (0.4861327, 0.5875618, 0.6562725)


def _doublet_system():
    b = Builder()
    b.add(c=1 / 40, t=5.0, material="N-BK7")
    b.add(c=-1 / 60, t=6.0)
    b.stop(t=2.0)
    b.add(c=-1 / 45, t=2.5, material="N-SF5")
    b.add(c=1 / 200, t=40.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=8.0, fields=((0.0, 0.0), (0.0, 5.0)), wavelengths=VIS)
    prob = EM.Problem(system, targets=EM.Targets(efl=50.0))
    return prob, EM.make_params(prm, prob), spec


def test_inserted_screen_and_cement_layer_change_nothing():
    prob, P, spec = _doublet_system()
    c0 = D.cost(prob, P, spec, s=0.0)
    w0 = D.mean_wfe(prob, P, spec)
    for i in (-1, 1, 4):
        q, Q = D.insert_screen(prob, P, i, r_norm=5.0)
        assert len(q.system.surfaces) == len(prob.system.surfaces) + 2
        assert D.mean_wfe(q, Q, spec) == pytest.approx(w0, rel=1e-12, abs=1e-12)
        assert D.cost(q, Q, spec, s=0.0) == pytest.approx(c0, rel=1e-9)
    q, Q = D.insert_cement_layer(prob, P, 1)
    assert q.system.surfaces[1].material == "model"
    assert D.mean_wfe(q, Q, spec) == pytest.approx(w0, rel=1e-12, abs=1e-12)
    assert D.cost(q, Q, spec, s=0.0) == pytest.approx(c0, rel=1e-9)
    with pytest.raises(ValueError):
        D.insert_cement_layer(prob, P, 2)  # the stop is not a lens rear surface


def test_per_wavelength_focus_separates_image_planes():
    """A singlet has axial color; each wavelength in its own plane is sharp."""
    b = Builder()
    b.stop(t=0.0)
    b.add(c=1 / 51.68, t=4.0, material="N-BK7")
    b.add(c=0.0, t=48.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=5.0, fields=((0.0, 0.0),), wavelengths=VIS)
    prob = EM.Problem(system)
    P = EM.make_params(prm, prob)
    free = {"s": [{}, {}, {"t": True}, {}], "mc": [{}]}
    P1, _ = D.optimize_design(prob, P, spec, free, iters=60)
    one_plane = D.mean_wfe(prob, P1, spec)
    q, Q = D.per_wavelength_focus(prob, P1, spec)
    assert [c.wavelengths for c in q.configs] == [(w,) for w in VIS]
    Q, _ = D.optimize_design(q, Q, spec, D.variables(q, curvatures=False, thicknesses=False),
                             iters=60)
    per_wl = np.concatenate(EM.rms_wfe(Q, q, spec))
    assert per_wl.max() < 0.25 * one_plane
    d = [float(Q["mc"][c]["2:t"]) for c in range(3)]
    assert d[0] < d[1] < d[2]  # blue focuses first
    assert d[2] - d[0] > 0.5


def test_zoom_configurations_reach_their_focal_lengths():
    """Two thin groups; their spacing and the back focus are per-configuration
    variables, the focal-length targets differ."""
    b = Builder()
    b.stop(t=0.0)
    b.add(c=1 / 25.8, t=3.0, material="N-BK7")
    b.add(c=-1 / 25.8, t=10.0)
    b.add(c=-1 / 25.8, t=2.0, material="N-BK7")
    b.add(c=1 / 25.8, t=30.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=4.0, fields=((0.0, 0.0), (0.0, 3.0)), wavelengths=(0.5875618,))
    configs = (EM.Config("wide", overrides=((2, "t"), (4, "t")), efl=40.0),
               EM.Config("tele", overrides=((2, "t"), (4, "t")), efl=60.0))
    prob = EM.Problem(system, configs=configs)
    P = EM.make_params(prm, prob)
    free = D.variables(prob, curvatures=False, thicknesses=False)
    assert free["mc"] == [{"2:t": True, "4:t": True}] * 2
    P, info = D.optimize_design(prob, P, spec, free, iters=80, gmin=0.5)
    from novaoptics.engine.trace import focal_length

    for c, target in enumerate((40.0, 60.0)):
        surfs = EM.config_surfaces(P, prob, c)
        assert float(focal_length(surfs, system, spec)) == pytest.approx(target, rel=2e-3)
    assert float(P["mc"][0]["2:t"]) != pytest.approx(float(P["mc"][1]["2:t"]), rel=1e-2)


def test_stop_must_stay_in_air():
    prob, P, spec = _doublet_system()
    c0 = D.cost(prob, P, spec)
    bad = dict(P, s=[dict(q) for q in P["s"]])
    bad["s"][1] = dict(bad["s"][1], t=jnp.asarray(8.5))  # stop plane behind the next lens'
    bad["s"][2] = dict(bad["s"][2], t=jnp.asarray(-0.5))  # front vertex
    assert D.cost(prob, bad, spec) > c0 + 1.0


def test_variables_skip_values_every_configuration_overrides():
    prob, P, spec = _doublet_system()
    q = dataclasses.replace(prob, configs=(EM.Config(overrides=((4, "t"),)),
                                           EM.Config(overrides=((4, "t"),))))
    v = D.variables(q)
    assert "t" not in v["s"][4] and v["mc"][0] == {"4:t": True}
