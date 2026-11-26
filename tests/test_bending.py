"""Element re-bending (used by bending hops)."""

import novaoptics  # noqa: F401  (CPU + x64)

import jax.numpy as jnp
import numpy as np
import pytest

from novaoptics import analysis as A
from novaoptics.bending import rebend
from novaoptics.system import Layout, Spec, new_params

SPEC = Spec()


@pytest.mark.parametrize("X", [-3.0, 0.0, 1.7, 4.5])
def test_rebend_sets_shape_and_keeps_thin_lens_power(X):
    lay = Layout(n_elem=1, stop_index=0)
    p = new_params(lay, gaps=[0.0], bfd=45.0, nd=1.5168, vd=64.17)
    p.update(c1=jnp.array([1 / 40.0]), c2=jnp.array([-1 / 60.0]), t=jnp.array([0.0]))
    q = rebend(p, 0, X)
    c1, c2 = float(q["c1"][0]), float(q["c2"][0])
    assert (c1 + c2) / (c1 - c2) == pytest.approx(X, abs=1e-12)
    assert c1 - c2 == pytest.approx(1 / 40.0 + 1 / 60.0, rel=1e-12)
    # thin lens: same power, so same EFL
    assert float(A.paraxial_efl(q, lay, SPEC)) == pytest.approx(
        float(A.paraxial_efl(p, lay, SPEC)), rel=1e-6)
    assert float(p["c1"][0]) == pytest.approx(1 / 40.0)  # input untouched
