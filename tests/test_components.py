"""Native components: grouping, glass-dispersing phase, dematerialize."""

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import glass as G
from novaoptics.engine import Builder, Spec
from novaoptics.engine import components as K
from novaoptics.engine import design as D
from novaoptics.engine import merit as EM
from novaoptics.engine import trace as T


def test_grouping_of_every_kind():
    b = Builder()
    b.add(c=1 / 30, t=4.0, material="N-BK7", label="L1 front")  # cemented doublet
    b.add(c=-1 / 25, t=2.0, material="N-SF5")
    b.add(c=-1 / 90, t=3.0)
    b.stop(t=2.0)
    b.add(phase="grating", freq=300.0, t=5.0, label="G")  # grating screen in air
    b.add(material="model", t=0.0, nd=1.6, vd=50.0)  # slab + screen (materializable)
    b.add(phase="glass", pcoef=[-0.5, 0, 0, 0], r_norm=5.0, t=10.0)
    b.cb(tx=10.0)
    b.mirror(c=-1 / 100, t=-20.0)
    b.add(interact="none", t=-5.0, label="dummy")
    b.image()
    system, prm = b.build()
    kinds = [c.kind for c in K.group(system, prm)]
    assert kinds == ["lens", "stop", "screen", "screen", "break", "mirror", "dummy", "image"]
    comps = K.group(system, prm)
    assert (comps[0].start, comps[0].end) == (0, 2)  # cemented: three surfaces
    assert comps[0].label == "L1"
    assert "cemented" in K.describe(system, prm)
    assert K.power(comps[5], system, prm) == pytest.approx(2 / 100)
    # a slab that has grown thickness is a lens, not a screen
    prm[5] = dict(prm[5], t=jnp.asarray(2.0))
    assert [c.kind for c in K.group(system, prm)][3] == "lens"


def test_glass_phase_matches_element_model():
    """Residual phase that disperses like the element's glass (the morph's)."""
    from novaoptics.engine import convert
    from novaoptics.system import Layout, Spec as ESpec, new_params
    from novaoptics.trace import aim_chief, launch, trace_ray

    lay = Layout(n_elem=2, stop_index=1)
    p = new_params(lay, gaps=[3.0, 4.0], bfd=40.0, nd=1.7, vd=30.0)
    p.update(c1=jnp.array([0.02, -0.01]), c2=jnp.array([-0.005, 0.01]), t=jnp.array([4.0, 3.0]),
             phase=jnp.array([[-0.6, 0.01, 0.0, 0.0], [0.3, -0.004, 0.0, 0.0]]))
    spec = ESpec(fnum=6.0)
    sysm, prm, es = convert.from_layout(p, lay, spec.epd, [0, 10])
    for wl in G.VISIBLE:
        y0 = aim_chief(p, lay, jnp.deg2rad(10.0), spec.ref_wavelength)
        xy = T.aim_chief(prm, sysm, (0.0, 10.0), spec.ref_wavelength)
        a = trace_ray(p, lay, wl, 1.0, *launch(jnp.deg2rad(10.0), y0, 0.3, 0.8, spec.epd / 2,
                                               lay))["img"]
        b = T.trace_ray(prm, sysm, wl, *T.launch(xy, (0.0, 10.0), 0.3, 0.8, spec.epd, sysm))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b["pts"][-1]), atol=1e-10)


def test_dematerialize_keeps_first_order():
    b = Builder()
    b.add(c=1 / 40, t=3.0, material="N-BK7")
    b.add(c=-1 / 40, t=4.0)
    b.stop(t=3.0)
    b.add(c=1 / 25, t=5.0, material="N-LAK22")  # a thick singlet
    b.add(c=-1 / 60, t=30.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=6.0, fields=((0.0, 0.0), (0.0, 5.0)))
    prob = EM.Problem(system)
    P = EM.make_params(prm, prob)
    f0 = float(T.focal_length(P["s"], system, spec))
    q, Q = D.dematerialize(prob, P, spec, 2, iters=0)  # the first-order swap alone
    kinds = [c.kind for c in D.components(q, Q)]
    assert kinds == ["lens", "stop", "screen", "image"]
    assert q.system.surfaces[3].material == "N-LAK22"  # the screen disperses like the lens
    assert q.system.surfaces[4].phase == "glass"
    assert float(T.focal_length(Q["s"], q.system, spec)) == pytest.approx(f0, rel=1e-6)
