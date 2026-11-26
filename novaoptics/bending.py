"""Element re-bending, for basin hopping over bendings (`stage_bending_hops`).

A phase screen carries only an element's net optical path; how the glass element is
bent is found during the morph, which can lock an element into a poor bending basin.
Both basins can be fully corrected in third order (they differ in fifth-order
residuals), so third-order bending theory cannot choose between them; re-bending an
element and re-polishing with the real-ray merit can.

(Tried and removed: two separated phase screens per element, whose power split turned
out to be nearly bending-blind because ideal screens act like infinite-index lenses,
and a third-order bending scan before the morph, which did not predict the outcome.)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def rebend(p, i, X):
    """Element i re-bent to shape factor X = (c1 + c2)/(c1 - c2), keeping c1 - c2."""
    c1, c2 = np.array(p["c1"], dtype=float), np.array(p["c2"], dtype=float)  # copies
    dc = c1[i] - c2[i]
    c1[i], c2[i] = 0.5 * (1.0 + X) * dc, 0.5 * (X - 1.0) * dc
    return dict(p, c1=jnp.asarray(c1), c2=jnp.asarray(c2))
