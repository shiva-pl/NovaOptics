"""Conversions between the element model of the phase-first pipeline (`system.Layout`)
and the surface engine."""

from __future__ import annotations

import numpy as np

from .. import glass as G
from ..system import Layout, glass_label
from .system import Builder, Spec


def from_layout(params, layout: Layout, epd, fields_deg, wavelengths=G.VISIBLE):
    """(System, params, Spec) equivalent to an element-model design.

    Exact catalog glasses keep their names; continuous glasses become "model" glasses
    (n_d, V_d, two-term Cauchy, as in the element model without a catalog).  Residual
    phase screens become "glass" phase, which disperses like the element's glass, as in
    the element model.
    """
    p = {k: np.asarray(v) for k, v in G.resolve(params, layout).items()}
    b = Builder(z_launch=layout.z_launch)
    for k, item in enumerate(layout.items()):
        gap = float(p["gaps"][k]) if k < layout.n_gaps else float(p["bfd"])
        if item == "stop":
            b.stop(t=gap, label="stop")
            continue
        i = item
        name = glass_label(p, layout, i) if layout.glass is not None else "model"
        extra = {} if name != "model" else {"nd": float(p["nd"][i]), "vd": float(p["vd"][i])}
        b.add(c=float(p["c1"][i]), t=float(p["t"][i]), material=name, label=f"E{i} front",
              **extra)
        rear = {"c": float(p["c2"][i]), "t": gap, "material": "air", "label": f"E{i} rear"}
        if layout.aspheres:
            rear.update(shape="even", coef=p["asph"][i], r_norm=layout.r_norm)
        if np.any(p["phase"][i] != 0):  # residual phase disperses like the element's glass
            rear.update(phase="glass", pcoef=p["phase"][i], r_norm=layout.r_norm)
        b.add(**rear)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=epd, fields=tuple((0.0, float(f)) for f in fields_deg),
                wavelengths=tuple(wavelengths))
    return system, prm, spec
