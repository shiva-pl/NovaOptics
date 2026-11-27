"""Native components: the objects a design is made of.

A design is a sequence of components, each spanning one or more engine surfaces:

    lens     a singlet or a cemented group: refracting surfaces with glass between them
             (any surface may carry phase, e.g. the residual phase of the morph or a
             diffractive surface)
    screen   a phase screen in air: ideal (achromatic), glass-dispersing, diffractive
             (OPD proportional to the wavelength) or a linear grating (wavelength
             dependent, lines/mm and order).  A screen can sit on the zero-thickness
             model-glass slab that the phase-first morph turns into a lens.
    mirror   a reflecting surface (optionally with a grating)
    stop     the aperture stop
    break    a coordinate break (decenter / tilt)
    dummy    a surface without optical effect
    image    the image surface

Surface-by-surface data (a .zmx file, the surface table of the lens data editor) is
grouped into components with `group`; the surface engine only traces them.  The
phase-first operations act on components: a screen is materialized into a lens, a
lens dematerialized into its equivalent screen, a singlet cemented into a group, a new
element grown from a screen inserted into an air space (engine/design.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR = ("air", "", "vacuum")


@dataclass(frozen=True)
class Component:
    kind: str
    start: int  # first surface
    end: int  # last surface (inclusive)
    label: str = ""

    @property
    def surfaces(self):
        return range(self.start, self.end + 1)


def group(system, params=None):
    """Components of a system, in order (every surface belongs to exactly one).

    With `params`, a two-surface glass group whose front is flat with zero thickness and
    whose rear carries ideal / glass phase is a phase screen on the morph's slab;
    without, that is decided from the structure alone (model-glass slab)."""
    S = system.surfaces
    n = len(S)
    out, i, k_lens, k_screen = [], 0, 0, 0
    while i < n:
        s = S[i]
        if i == n - 1:
            out.append(Component("image", i, i, s.label or "image"))
        elif s.shape == "cb":
            out.append(Component("break", i, i, s.label))
        elif s.interact == "reflect":
            out.append(Component("mirror", i, i, s.label))
        elif s.interact == "none":
            out.append(Component("stop" if s.stop else "dummy", i, i, s.label))
        elif s.material not in AIR:
            # a lens: every refracting surface up to the one that returns to air
            j = i
            while j + 1 < n - 1 and S[j].material not in AIR and S[j + 1].interact == "refract" \
                    and S[j + 1].shape != "cb":
                j += 1
            k_lens += 1
            slab = j == i + 1 and S[j].phase in ("ideal", "glass") and not S[i].phase \
                and S[i].shape == "conic"
            if params is not None:
                slab = slab and abs(float(params[i]["t"])) < 1e-9 and \
                    abs(float(params[i]["c"])) < 1e-12
            else:
                slab = slab and S[i].material == "model"
            kind = "screen" if slab else "lens"
            if kind == "screen":
                k_screen += 1
            label = S[i].label if S[i].label and not S[i].label.endswith("front") else \
                (S[i].label.rsplit(" ", 1)[0] if S[i].label else "")
            out.append(Component(kind, i, j, label or (f"L{k_lens}" if kind == "lens"
                                                       else f"P{k_screen}")))
            i = j
        else:
            k_screen += 1
            out.append(Component("screen", i, i, s.label or f"P{k_screen}"))
        i += 1
    return tuple(out)


def is_thin_slab(comp, params):
    p = params[comp.start]
    return comp.kind == "screen" and comp.end == comp.start + 1 and abs(float(p["t"])) < 1e-9 \
        and abs(float(p["c"])) < 1e-12


def paraxial_matrix(comp, system, params, wl=0.5875618):
    """Reduced paraxial matrix [[A, B], [C, D]] of a lens from its front to its rear
    vertex, acting on (y, n u)."""
    from . import materials as M

    S = system.surfaces
    Mx = np.eye(2)
    n = 1.0
    for k in comp.surfaces:
        n2 = 1.0 if S[k].material in AIR else float(M.index(S[k].material, wl, params[k]))
        phi = (n2 - n) * float(params[k]["c"])
        Mx = np.array([[1.0, 0.0], [-phi, 1.0]]) @ Mx
        if k < comp.end:
            Mx = np.array([[1.0, float(params[k]["t"]) / n2], [0.0, 1.0]]) @ Mx
        n = n2
    return Mx


def power(comp, system, params, wl=0.5875618):
    """Paraxial power [1/mm] of a lens (thick-lens formula, chained) or a screen."""
    from . import materials as M

    S = system.surfaces
    if comp.kind == "screen":
        p = params[comp.end]
        s = S[comp.end]
        if s.phase == "grating":
            return 0.0
        a1 = float(np.asarray(p["pcoef"])[0]) if s.n_phase else 0.0
        phi = -2.0 * a1 / s.r_norm**2
        if s.phase == "diffractive":
            phi *= s.order * wl * 1e-3
        return phi
    if comp.kind == "mirror":
        return -2.0 * float(params[comp.start]["c"])
    if comp.kind != "lens":
        return 0.0
    # paraxial y-nu trace of a ray parallel to the axis
    y, u, n = 1.0, 0.0, 1.0
    for k in comp.surfaces:
        s = S[k]
        n2 = 1.0 if s.material in AIR else float(M.index(s.material, wl, params[k]))
        c = float(params[k]["c"])
        u = (n * u - y * (n2 - n) * c) / n2
        n = n2
        if k < comp.end:
            y = y + float(params[k]["t"]) * u
    return -u / 1.0


def rows(system, params, comps=None):
    """One dict per component: kind, label, surfaces, power [1/m] (None if not
    applicable), details, space after [mm] (None for the image)."""
    comps = comps or group(system, params)
    S = system.surfaces
    out = []
    for c in comps:
        P = [params[i] for i in c.surfaces]
        det = ""
        if c.kind == "lens":
            radii = " / ".join("inf" if abs(float(p["c"])) < 1e-12 else f"{1 / float(p['c']):.2f}"
                               for p in P)
            th = " + ".join(f"{float(p['t']):.2f}" for p in P[:-1])
            gl = " | ".join(_glass(S[i], params[i]) for i in list(c.surfaces)[:-1])
            ph = [S[i].phase for i in c.surfaces if S[i].phase]
            det = f"R {radii}  t {th}  {gl}" + (f"  phase: {', '.join(ph)}" if ph else "")
            det += "  cemented" if c.end - c.start > 1 else ""
        elif c.kind == "screen":
            s = S[c.end]
            if s.phase == "grating":
                det = f"grating {float(params[c.end]['freq']):.1f} lines/mm, order {s.order}"
            else:
                det = f"{s.phase or 'no'} phase"
                if c.end > c.start:
                    det += f" on a zero-thickness {_glass(S[c.start], params[c.start])} slab " \
                           "(materializable)"
        elif c.kind == "mirror":
            p = P[0]
            det = ("R inf" if abs(float(p["c"])) < 1e-12 else f"R {1 / float(p['c']):.2f}") + \
                f"  k {float(p['k']):.4g}" + (f"  {S[c.start].phase}" if S[c.start].phase else "")
        elif c.kind == "break":
            p = P[0]
            det = " ".join(f"{key}={float(p[key]):g}" for key in ("dx", "dy", "tx", "ty", "tz")
                           if float(p[key]) != 0.0)
        pw = power(c, system, params) * 1e3 if c.kind in ("lens", "screen", "mirror") else None
        out.append({"kind": c.kind, "label": c.label, "surfaces": f"{c.start}-{c.end}",
                    "power": pw, "details": det,
                    "space": None if c.kind == "image" else float(params[c.end]["t"])})
    return out


def describe(system, params, comps=None):
    """One line per component."""
    lines = [f"{'#':<3}{'component':<10}{'label':<14}{'surfaces':<10}{'power [1/m]':>12}  details"]
    for k, r in enumerate(rows(system, params, comps)):
        pw = "" if r["power"] is None else f"{r['power']:+.2f}"
        sp = "" if r["space"] is None else f"  -> {r['space']:.3f} mm"
        lines.append(f"{k:<3}{r['kind']:<10}{r['label'][:13]:<14}{r['surfaces']:<10}{pw:>12}  "
                     f"{r['details']}{sp}")
    return "\n".join(lines)


def _glass(s, p):
    if s.material == "model":
        return f"{float(p['nd']):.4f}/{float(p['vd']):.1f}"
    return s.material
