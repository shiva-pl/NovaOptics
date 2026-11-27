"""Sequential surface model.

A `System` is a static tuple of `Surface` descriptions (what each surface *is*) and a
parameter list of per-surface dicts (its numbers, differentiable).  The conventions
follow common lens design programs:

* Surface 0 is the first optical surface; the object is at infinity.  The last surface
  is the image surface.
* Each surface sits at the origin of the running coordinate frame; `t` is the signed
  distance along the frame's z axis to the next surface (negative after a mirror, as
  usual).  `material` is the medium *after* the surface ("air", a catalog glass name,
  or "model" for an (n_d, V_d) design glass).  A mirror keeps the medium.
* A coordinate break (`shape="cb"`) does not interact with rays.  It decenters (dx, dy)
  and tilts (tx, ty, tz, degrees, right-handed about the local axes) the running frame
  for all following surfaces.  With
  order 0 it decenters then tilts (x, y, z); with order 1 it tilts (z, y, x) then
  decenters, which undoes an order-0 break with the opposite values.
* Shapes: "conic" (c, k), "even" (conic + a2 r^2 + sum coef_j u^(2j+4)), "qbfs" (Forbes type 0),
  "qcon" (Forbes type 1), with u = r / r_norm.  See qpoly.py.
* Phase (OPD, mm, added on the surface; the interface law turns its gradient into a
  change of the tangential ray momentum):
    "ideal"        W = sum pcoef_k rho^k, rho = r^2 / r_norm^2, achromatic
    "diffractive"  W = order * lambda * sum pcoef_k rho^k   (pcoef_k in waves)
    "grating"      W = order * lambda * freq * y            (freq in lines/mm, lines // x)
* The stop surface carries an aiming target (sx, sy) in its local coordinates, so
  off-axis apertures of coaxial parents (decentered stops) are aimed correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

SHAPES = ("conic", "even", "qbfs", "qcon", "cb")
PHASES = ("", "ideal", "diffractive", "grating")


@dataclass(frozen=True)
class Surface:
    shape: str = "conic"
    n_terms: int = 0  # aspheric / Q coefficients
    r_norm: float = 10.0  # normalization radius for aspheres and phase [mm]
    interact: str = "refract"  # refract | reflect | none
    material: str = "air"  # medium after the surface
    phase: str = ""
    n_phase: int = 0
    order: int = 1  # diffraction order for diffractive / grating phase
    stop: bool = False
    cb_order: int = 0
    label: str = ""

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}")
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase!r}")


@dataclass(frozen=True)
class Spec:
    """Object at infinity: entrance beam diameter, field angles (x, y) [deg], spectrum."""

    epd: float = 12.5
    fields: tuple = ((0.0, 0.0), (0.0, 14.0), (0.0, 20.0))
    wavelengths: tuple = (0.4861327, 0.5875618, 0.6562725)  # um
    wl_weights: tuple = (1.0, 1.0, 1.0)
    field_weights: tuple = None
    ref: int = None  # index of the reference wavelength (default: middle)

    @property
    def iref(self):
        return len(self.wavelengths) // 2 if self.ref is None else self.ref

    @property
    def ref_wavelength(self):
        return self.wavelengths[self.iref]

    @property
    def fweights(self):
        return self.field_weights or (1.0,) * len(self.fields)


@dataclass(frozen=True)
class System:
    surfaces: tuple
    z_launch: float = -20.0  # rays start on this plane (global z) in front of surface 0

    def __post_init__(self):
        stops = [i for i, s in enumerate(self.surfaces) if s.stop]
        if len(stops) != 1:
            raise ValueError("exactly one surface must be the stop")

    @property
    def stop(self):
        return next(i for i, s in enumerate(self.surfaces) if s.stop)

    @property
    def n(self):
        return len(self.surfaces)


KEYS = ("c", "k", "t", "dx", "dy", "tx", "ty", "tz", "freq", "nd", "vd", "sx", "sy", "a2")


def surface_params(s: Surface, **values):
    """Parameter dict for surface `s` with defaults (all floats are JAX scalars)."""
    p = {k: 0.0 for k in KEYS}
    p.update(nd=1.5168, vd=64.17)
    p["coef"] = np.zeros(s.n_terms)
    p["pcoef"] = np.zeros(s.n_phase)
    for k, v in values.items():
        if k not in p:
            raise KeyError(f"unknown surface parameter {k!r}")
        p[k] = v
    return {k: jnp.asarray(v, float) for k, v in p.items()}


def _rot(axis, deg):
    a = jnp.deg2rad(deg)
    c, s = jnp.cos(a), jnp.sin(a)
    o, z = jnp.ones_like(a), jnp.zeros_like(a)
    if axis == 0:
        return jnp.array([[o, z, z], [z, c, -s], [z, s, c]])
    if axis == 1:
        return jnp.array([[c, z, s], [z, o, z], [-s, z, c]])
    return jnp.array([[c, -s, z], [s, c, z], [z, z, o]])


def frames(params, system: System):
    """Global (origin, rotation) of every surface; columns of the rotation are the
    surface's local axes."""
    O = jnp.zeros(3)
    R = jnp.eye(3)
    out = []
    for s, p in zip(system.surfaces, params):
        if s.shape == "cb":
            dec = jnp.array([p["dx"], p["dy"], 0.0])
            if s.cb_order == 0:
                O = O + R @ dec
                R = R @ _rot(0, p["tx"]) @ _rot(1, p["ty"]) @ _rot(2, p["tz"])
            else:
                R = R @ _rot(2, p["tz"]) @ _rot(1, p["ty"]) @ _rot(0, p["tx"])
                O = O + R @ dec
        out.append((O, R))
        O = O + R @ jnp.array([0.0, 0.0, 1.0]) * p["t"]
    return out


@dataclass
class Builder:
    """Convenience: append surfaces with their parameters in order.

        b = Builder()
        b.add(c=1/50, t=5, material="N-BK7")
        b.add(c=-1/50, t=45, stop=True)
        b.image()
        system, params = b.build()
    """

    surfaces: list = field(default_factory=list)
    params: list = field(default_factory=list)
    z_launch: float = -20.0

    def add(self, shape="conic", n_terms=None, r_norm=10.0, interact="refract",
            material=None, phase="", n_phase=None, order=1, stop=False, label="",
            cb_order=0, **values):
        if material is None:  # refracting surfaces default to air, others keep the medium
            prev = self.surfaces[-1].material if self.surfaces else "air"
            material = "air" if interact == "refract" else prev
        n_terms = len(values.get("coef", ())) if n_terms is None else n_terms
        n_phase = len(values.get("pcoef", ())) if n_phase is None else n_phase
        s = Surface(shape, n_terms, r_norm, interact, material, phase, n_phase, order, stop,
                    cb_order, label)
        self.surfaces.append(s)
        self.params.append(surface_params(s, **values))
        return len(self.surfaces) - 1

    def mirror(self, **kw):
        return self.add(interact="reflect", **kw)

    def cb(self, dx=0.0, dy=0.0, tx=0.0, ty=0.0, tz=0.0, t=0.0, order=0, label=""):
        return self.add(shape="cb", interact="none", cb_order=order, dx=dx, dy=dy, tx=tx,
                        ty=ty, tz=tz, t=t, label=label)

    def stop(self, t=0.0, **kw):
        return self.add(interact="none", stop=True, t=t, **kw)

    def image(self, **kw):
        return self.add(interact="none", label=kw.pop("label", "image"), **kw)

    def build(self):
        return System(tuple(self.surfaces), self.z_launch), list(self.params)
