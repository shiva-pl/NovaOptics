"""Growing a design element by element: the phase-first way to more lenses.

A phase screen can be split into two screens that each carry half of its phase,
a short distance apart.  Two thin screens close together act like one screen with the
summed phase, so the merit is nearly continuous across the split; re-optimizing then
uses the new freedom (an extra glass for color, power spread over two surfaces, more
room to balance aberrations).  The same operation on a glass system would be a jump:
two thick lenses are not one.

`stage_grow` splits greedily: at every step each element is tried as the one to
split (concurrently), the system is re-optimized in phase space, and the best result
is kept, until the target element count is reached.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import continuation as C
from .merit import make_cfg
from .optimize import optimize
from .parallel import pmap
from .system import Layout, Spec

PER_ELEMENT = ("c1", "c2", "t", "nd", "vd", "phase", "asph", "g", "gr")


def split_element(p, lay: Layout, i, delta):
    """Split phase screen i into two screens `delta` mm apart, placed symmetrically about
    its position (the stop may end up between them).

    Each half gets the power phi' with 2 phi' - delta phi'^2 = phi, so the pair has the
    original power, and its principal planes stay within O(delta^2 phi) of the original
    screen: the split preserves first-order properties, and the merit barely changes.
    Aspheric phase orders are halved.  Only for phase-space designs (thin screens): a
    glass element of finite thickness cannot be split this way.

    Returns (params, layout, (index of the first half, index of the second half)).
    """
    if float(p["t"][i]) > 1e-9 or abs(float(p["c1"][i])) + abs(float(p["c2"][i])) > 1e-12:
        raise ValueError("split_element splits phase screens (thin, flat), not glass")
    items = lay.items()
    n = lay.n_elem
    space = [float(g) for g in np.asarray(p["gaps"])] + [float(p["bfd"])]
    z, zi = [], 0.0  # front z of every item
    for k, it in enumerate(items):
        z.append(zi)
        zi += (float(p["t"][it]) if it != "stop" else 0.0) + space[k]
    z_img = zi
    z0 = z[items.index(i)]
    others = [z[k] for k, it in enumerate(items) if it not in ("stop", i)]
    near = min([abs(zz - z0) for zz in others], default=np.inf)
    d = min(delta, near)  # do not jump over a neighbouring element

    phi = -2.0 * float(p["phase"][i, 0]) / lay.r_norm**2
    root = 1.0 - d * phi
    phi_h = (1.0 - np.sqrt(root)) / d if root > 0 and d > 0 else 0.5 * phi
    arr = {k: np.array(v, dtype=float) for k, v in p.items() if k in PER_ELEMENT}
    half = {k: v[i].copy() for k, v in arr.items()}
    for k in ("phase", "asph"):
        if k in half:
            half[k] = 0.5 * half[k]
    half["phase"][0] = -phi_h * lay.r_norm**2 / 2.0

    # (z, kind, source element) for every item after the split, sorted along the axis
    # sort along the axis; items at the same z (e.g. a screen in the stop plane) keep
    # their original order
    k_i = items.index(i)
    entries = [(z[k], k, it) for k, it in enumerate(items) if it != i]
    entries += [(z0 - d / 2, k_i - 0.5, "a"), (z0 + d / 2, k_i + 0.5, "b")]
    entries.sort(key=lambda e: (round(e[0], 9), e[1]))
    new_items, rows = [], []
    for zz, _, src in entries:
        if src == "stop":
            new_items.append("stop")
            continue
        rows.append(src)
        new_items.append(len(rows) - 1)
    q = {}
    for k, v in arr.items():
        q[k] = np.stack([half[k] if r in ("a", "b") else v[r] for r in rows])
    zs = [e[0] for e in entries]
    ts = [0.0 if it == "stop" else float(q["t"][it]) for it in new_items]
    gaps = [zs[k + 1] - (zs[k] + ts[k]) for k in range(len(zs) - 1)]
    q["gaps"] = np.array(gaps)
    q["bfd"] = np.array(z_img - (zs[-1] + ts[-1]))
    lay2 = Layout(n_elem=n + 1, stop_index=new_items.index("stop"), n_phase=lay.n_phase,
                  r_norm=lay.r_norm, z_launch=lay.z_launch, aspheres=lay.aspheres,
                  glass=lay.glass, region=lay.region)
    return ({k: jnp.asarray(v) for k, v in q.items()}, lay2,
            (rows.index("a"), rows.index("b")))


def stage_grow(p, lay: Layout, spec: Spec, rec: C.Recorder, n_target, mech=C.Mech(),
               virtual_petzval=1.0, iters=80, mu_sphere=0.0):
    """Split screens (phase space, beta = 1) until the design has `n_target` elements.

    mu_sphere > 0 adds the morph's penalty on aspheric phase to the split criterion,
    favouring splits that make the screens easier to realize with spherical glass.
    """
    gmin = 2.0 * mech.gmin  # phase-space spacing rule, as in DISTRIBUTE / COLOR
    cfg = make_cfg(beta=1.0, gmin=gmin, virtual_petzval=virtual_petzval, mu=mu_sphere)
    free = {"phase": True, "gaps": True, "bfd": True, **{v: True for v in C._gvars(lay)}}
    while lay.n_elem < n_target:
        def try_split(i):
            q, lq, _ = split_element(p, lay, i, delta=1.25 * gmin)
            q, info = optimize(q, lq, spec, cfg, free, iters=iters)
            c = info.get("cost", np.inf)
            return (c if np.isfinite(c) else np.inf), i, q, lq

        tried = pmap(try_split, range(lay.n_elem))  # independent candidates
        if rec.verbose:
            print("   grow: " + ", ".join(f"split E{i} -> {c:.4g}" for c, i, _, _ in tried))
        cost, i, p, lay = min(tried, key=lambda r: r[0])
        rec.add("grow", lay.n_elem, p, lay, 1.0,
                f"split E{i}: {lay.n_elem} screens  powers[1/m] "
                f"{C._powers(p['phase'], lay.r_norm)}  glasses {C._glasses(p, lay)}",
                virtual_petzval)
    return p, lay
