"""Stop-shift relations against the paraxial Seidel trace and the tracer."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics import diagnostics as D
from novaoptics import stopshift as SS
from novaoptics.system import Layout, Spec, new_params, vertex_positions

SPEC = Spec()


def triplet():
    lay = Layout(n_elem=3, stop_index=1)
    p = new_params(lay, gaps=[2.3207, 0.2368, 3.6190], bfd=38.764)
    p.update(c1=jnp.array([0.048007, -0.019118, 0.017153]),
             c2=jnp.array([0.019256, 0.047564, -0.030932]),
             t=jnp.array([8.0001, 1.9997, 2.6670]),
             nd=jnp.array([1.883, 1.8467, 1.883]), vd=jnp.array([40.76, 23.78, 40.76]))
    return p, lay


def test_relayout_keeps_the_optical_system():
    p, lay = triplet()
    z_old = float(vertex_positions(p, lay)["stop"])
    q, lq = SS.relayout(p, lay, z_old)  # no shift
    assert float(A.paraxial_efl(q, lq, SPEC)) == pytest.approx(float(A.paraxial_efl(p, lay, SPEC)))
    # move the stop in front of the first element: same lens, same image plane
    q, lq = SS.relayout(p, lay, -3.0)
    assert lq.stop_index == 0
    vp, vq = vertex_positions(p, lay), vertex_positions(q, lq)
    off = float(vq["elem_front"][0]) - float(vp["elem_front"][0])
    for i in range(3):
        assert float(vq["elem_front"][i]) - off == pytest.approx(float(vp["elem_front"][i]))
    assert float(vq["image"]) - off == pytest.approx(float(vp["image"]))


@pytest.mark.parametrize("z_new", [-4.0, 9.0, 14.5])
def test_shift_formulas_match_a_retrace(z_new):
    """Closed-form stop-shift sums equal a full Seidel trace with the stop moved."""
    p, lay = triplet()
    sc = SS.scan(p, lay, SPEC, virtual_petzval=0.0)
    i = int(np.argmin(np.abs(sc["z"] - z_new)))
    q, lq = SS.relayout(p, lay, float(sc["z"][i]))
    S_new = D.seidel(q, lq, SPEC)["total"]
    pred = SS.shifted_sums(D.seidel(p, lay, SPEC)["total"], sc["E"][i])
    np.testing.assert_allclose(pred, S_new, rtol=1e-6, atol=1e-9)


def test_lateral_color_independent_of_stop_when_axial_color_is_zero():
    S = np.array([1.0, 0.5, 0.2, 0.3, 0.1, 0.0, 0.04])  # C_I = 0
    E = np.linspace(-2, 2, 9)
    assert np.allclose(SS.shifted_sums(S, E)[:, 6], 0.04)
    E = np.linspace(-3, 3, 13)
    S[5] = 0.02  # with axial color, lateral color is linear in E and crosses zero
    cii = SS.shifted_sums(S, E)[:, 6]
    assert np.allclose(np.diff(cii), np.diff(cii)[0]) and cii.min() < 0 < cii.max()
