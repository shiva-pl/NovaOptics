"""System representation.

The central object is the *element*: a slab of glass bounded by two spherical
surfaces, carrying an additional thin phase (OPD) screen W(r) on its rear surface.

    element = { c1, c2 (curvatures), t (center thickness), n_d, V_d, W(r) }

This single parameterization spans the whole morphing path:

* a pure phase screen      : c1 = c2 = 0, t = 0, W != 0      (ideal, Petzval-free)
* a real singlet           : c1, c2, t != 0, W = 0           (manufacturable)
* anything in between      : a lens that still "borrows" some ideal phase

W(r) is an even polynomial in normalized radius, in units of optical path (mm):

    W(r) = sum_k a_k (r / r_norm)^(2k),   k = 1..n_phase

The phase is dispersion-free (an *ideal* phase), so it behaves like a perfectly
achromatic, Petzval-free thin element.  That is precisely why it makes a good
starting point and why the morph to glass is informative: glass power brings
Petzval curvature and dispersion back in, gradually, under the optimizer's eye.

`Layout` holds the static structure (number of elements, stop position); the
numeric parameters live in a plain dict pytree so JAX can differentiate through
all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from . import glass as G


@dataclass(frozen=True)
class Layout:
    n_elem: int
    stop_index: int  # stop sits in front of element `stop_index` (== n_elem: after last)
    n_phase: int = 4
    r_norm: float = 10.0
    z_launch: float = -20.0  # launch plane for rays, in front of the first item
    aspheres: bool = False  # even aspheric terms (orders 4..2*n_phase) on rear surfaces
    # Two-screen elements: an ideal phase screen on the front surface too ("phase1").
    # With real thickness, how the power is split between the screens is the bending.
    two_screens: bool = False
    glass: object = None  # catalog.GlassSpace -> PCA glass coordinates; None -> (n_d, V_d)
    region: object = None  # catalog.GlassSpace whose (n_d, V_d) hull bounds Cauchy glasses

    def items(self):
        """Sequence of items: element indices and the string 'stop'."""
        seq = list(range(self.n_elem))
        seq.insert(self.stop_index, "stop")
        return seq

    @property
    def n_gaps(self):
        return self.n_elem  # (n_elem + 1 items) - 1


@dataclass(frozen=True)
class Spec:
    """First-order requirements for an infinite-conjugate imaging lens."""

    efl: float = 50.0
    fnum: float = 4.0
    half_field_deg: float = 20.0
    wavelengths: tuple = G.VISIBLE
    wl_weights: tuple = (1.0, 1.0, 1.0)
    # Five fields: with only three (0, 0.7, 1) designs overfit, with good correction
    # at the sampled angles and bumps in between.
    field_fracs: tuple = (0.0, 0.4, 0.65, 0.85, 1.0)
    field_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    max_distortion: float = 0.02  # |y_real - f tan(theta)| / (f tan(theta))
    max_track: float = 80.0  # first vertex to image plane

    @property
    def epd(self):
        return self.efl / self.fnum

    @property
    def ref_wavelength(self):
        return self.wavelengths[len(self.wavelengths) // 2]


def new_params(layout: Layout, gaps=None, bfd=50.0, nd=1.62, vd=50.0):
    n = layout.n_elem
    out = {
        "c1": jnp.zeros(n),
        "c2": jnp.zeros(n),
        "t": jnp.zeros(n),
        "nd": jnp.full(n, nd),
        "vd": jnp.full(n, vd),
        "phase": jnp.zeros((n, layout.n_phase)),
        "asph": jnp.zeros((n, layout.n_phase - 1)),
        **({"phase1": jnp.zeros((n, layout.n_phase))} if layout.two_screens else {}),
        **({"g": jnp.zeros((n, layout.glass.k)),
            "gr": jnp.zeros((n, len(layout.glass.grid)))} if layout.glass is not None else {}),
        "gaps": jnp.asarray(gaps if gaps is not None else np.full(n, 5.0), dtype=float),
        "bfd": jnp.asarray(bfd, dtype=float),
    }
    if layout.glass is not None:
        out = dict(out)
        set_glass(out, layout, list(range(n)), [nearest_in_space(layout.glass, nd, vd)] * n)
    return out


def nearest_in_space(gs, nd, vd):
    """Catalog glass of a glass space closest to (n_d, V_d)."""
    d = ((gs.nd - nd) / 0.1) ** 2 + ((gs.vd - vd) / 10.0) ** 2
    return int(np.argmin(d))


def set_glass(params, layout, elems, glass_ids, exact=True):
    """Put catalog glass `glass_ids[k]` into element `elems[k]` (PCA mode).

    exact=True also sets the off-span residual, so the index curve is the catalog's.
    """
    gs = layout.glass
    g = np.array(params["g"], dtype=float)
    gr = np.array(params["gr"], dtype=float)
    for e, j in zip(elems, glass_ids):
        g[e] = gs.coords[j] / gs.scale
        gr[e] = gs.resid[j] if exact else 0.0
    params["g"] = jnp.asarray(g)
    params["gr"] = jnp.asarray(gr)
    return params


def to_numpy(params):
    return {k: np.asarray(v) for k, v in params.items()}


def vertex_positions(params, layout: Layout):
    """z of every item's front vertex, plus each element's rear vertex and the image."""
    z = 0.0
    out = {"elem_front": [None] * layout.n_elem, "elem_back": [None] * layout.n_elem}
    for k, item in enumerate(layout.items()):
        if item == "stop":
            out["stop"] = z
        else:
            out["elem_front"][item] = z
            z = z + params["t"][item]
            out["elem_back"][item] = z
        if k < layout.n_gaps:
            z = z + params["gaps"][k]
    out["image"] = z + params["bfd"]
    return out


def glass_label(p, layout, i):
    """Catalog name if element i holds an exact catalog glass, else a model label."""
    gs = layout.glass
    if gs is None:
        return G.glass_name(p["nd"][i], p["vd"][i])
    c = gs.coords / gs.scale
    j = int(np.argmin(((c - p["g"][i]) ** 2).sum(-1)))
    if np.allclose(c[j], p["g"][i], atol=1e-9) and np.allclose(gs.resid[j], p["gr"][i],
                                                               atol=1e-12):
        return gs.names[j]
    return f"pca({p['nd'][i]:.4f}/{p['vd'][i]:.1f})"


def describe(params, layout: Layout, spec: Spec | None = None) -> str:
    """Human-readable prescription."""
    p = to_numpy(G.resolve(params, layout))
    lines = [
        f"{'item':<8}{'R1 [mm]':>13}{'R2 [mm]':>13}{'t [mm]':>9}  {'glass':<24}{'gap->':>9}"
        f"{'  max |W| [waves]':>18}"
    ]
    wl = (spec.ref_wavelength if spec else G.LAMBDA_D) * 1e-3
    for k, item in enumerate(layout.items()):
        gap = p["gaps"][k] if k < layout.n_gaps else p["bfd"]
        if item == "stop":
            lines.append(f"{'STOP':<8}{'':>13}{'':>13}{'':>9}  {'':<24}{gap:>9.3f}")
            continue
        i = item
        r1 = "inf" if abs(p["c1"][i]) < 1e-9 else f"{1 / p['c1'][i]:.3f}"
        r2 = "inf" if abs(p["c2"][i]) < 1e-9 else f"{1 / p['c2'][i]:.3f}"
        gname = glass_label(p, layout, i)
        if layout.aspheres and np.any(np.abs(p["asph"][i]) > 1e-9):
            r2 = r2 + "*"
        rr = np.linspace(0, 1, 50) ** 2
        w = np.polyval(np.r_[p["phase"][i][::-1], 0.0], rr)
        lines.append(
            f"{'E' + str(i):<8} {r1:>12} {r2:>12} {p['t'][i]:>8.3f}  {gname:<24}{gap:>9.3f}"
            f"{np.max(np.abs(w)) / wl:>18.2f}"
        )
    if layout.aspheres:
        lines.append("* rear surface is an even asphere")
    return "\n".join(lines)
