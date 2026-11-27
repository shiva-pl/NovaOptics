"""Differentiable real-ray tracing through a sequential `System`.

One law covers refraction, reflection and phase (diffractive / ideal / grating)
surfaces, with N the unit surface normal and W the surface phase (OPD):

    v  = n d + grad W                     (grad W lies in the surface's local xy plane)
    vt = v - (v . N) N                    (tangential momentum)
    refraction:  n' d' = vt + sign(d.N) sqrt(n'^2 - |vt|^2) N
    reflection:  n  d' = vt - sign(d.N) sqrt(n^2  - |vt|^2) N

and the optical path picks up W.  Intersections: closed form for the conic base
surface (the root that stays finite as c -> 0), then Newton iterations on the full
sag for aspheres and Q-type surfaces; normals from autodiff of the sag.  Rays that
miss, or meet total internal reflection / evanescent orders, are clamped and flagged
(`ok`), never NaN.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import materials as M
from . import qpoly
from .system import Spec, System, frames

_EPS = 1e-14
LAMBDA_D = 0.5875618


def _sqrt(x):
    return jnp.sqrt(jnp.maximum(x, _EPS))


# --------------------------------------------------------------------------- shapes

def sag(s, p, x, y):
    """Local sag z(x, y) of surface `s` with parameters `p`."""
    r2 = x * x + y * y
    c = p["c"]
    if s.shape == "qbfs":
        root = _sqrt(1.0 - c * c * r2)
        z = c * r2 / (1.0 + root)
        if s.n_terms:
            u2 = r2 / s.r_norm**2
            Q = qpoly.qbfs(u2, s.n_terms)
            z = z + u2 * (1.0 - u2) / root * jnp.dot(p["coef"], Q)
        return z
    k = p["k"]
    z = c * r2 / (1.0 + _sqrt(1.0 - (1.0 + k) * c * c * r2))
    if s.shape == "even":
        z = z + p["a2"] * r2  # unnormalized r^2 term (Zemax EVENASPH alpha_1)
    if s.shape == "even" and s.n_terms:
        rho = r2 / s.r_norm**2
        pw = [rho * rho]
        for _ in range(s.n_terms - 1):
            pw.append(pw[-1] * rho)
        z = z + jnp.dot(p["coef"], jnp.stack(pw))
    elif s.shape == "qcon" and s.n_terms:
        u2 = r2 / s.r_norm**2
        z = z + u2 * u2 * jnp.dot(p["coef"], qpoly.qcon(u2, s.n_terms))
    return z


def _intersect(s, p, pos, d, newton=6):
    """Distance along d from pos (local coordinates) to the surface, and an ok flag."""
    c = p["c"]
    kk = 0.0 if s.shape == "qbfs" else p["k"]
    # transfer to the vertex plane first: from there the root that stays finite as
    # c -> 0 is the intercept on the vertex side of the surface
    t0 = -pos[2] / jnp.where(jnp.abs(d[2]) > _EPS, d[2], _EPS)
    start, pos = pos, pos + t0 * d
    a = c * (d[0] ** 2 + d[1] ** 2 + (1.0 + kk) * d[2] ** 2)
    B = c * (pos[0] * d[0] + pos[1] * d[1] + (1.0 + kk) * pos[2] * d[2]) - d[2]
    C = c * (pos[0] ** 2 + pos[1] ** 2 + (1.0 + kk) * pos[2] ** 2) - 2.0 * pos[2]
    disc = B * B - a * C
    sgn = jnp.where(B >= 0, 1.0, -1.0)
    den = B + sgn * _sqrt(disc)
    t = t0 - C / jnp.where(jnp.abs(den) > _EPS, den, _EPS)
    pos = start
    ok = disc > 0
    if s.shape == "even" or s.shape in ("qcon", "qbfs") and s.n_terms:
        def step(_, t):
            f = lambda tt: (pos[2] + tt * d[2]) - sag(s, p, pos[0] + tt * d[0],
                                                       pos[1] + tt * d[1])
            g, dg = jax.jvp(f, (t,), (jnp.ones_like(t),))
            return t - g / jnp.where(jnp.abs(dg) > _EPS, dg, _EPS)

        t = jax.lax.fori_loop(0, newton, step, t)
    return t, ok


def _normal(s, p, x, y):
    if s.shape == "conic" or s.n_terms == 0 and s.shape not in ("cb", "even"):
        # analytic conic normal (also valid for c = 0)
        c, k = p["c"], (0.0 if s.shape == "qbfs" else p["k"])
        r2 = x * x + y * y
        z = c * r2 / (1.0 + _sqrt(1.0 - (1.0 + k) * c * c * r2))
        N = jnp.array([-c * x, -c * y, 1.0 - (1.0 + k) * c * z])
    else:
        gx, gy = jax.grad(lambda xy: sag(s, p, xy[0], xy[1]))(jnp.array([x, y]))
        N = jnp.array([-gx, -gy, 1.0])
    return N / jnp.linalg.norm(N)


def phase(s, p, x, y, wl_um):
    """Surface phase W(x, y) [mm of optical path] at wavelength wl_um."""
    if s.phase == "grating":
        return s.order * wl_um * 1e-3 * p["freq"] * y
    rho = (x * x + y * y) / s.r_norm**2
    pw = [rho]
    for _ in range(s.n_phase - 1):
        pw.append(pw[-1] * rho)
    w = jnp.dot(p["pcoef"], jnp.stack(pw)) if s.n_phase else 0.0 * rho
    if s.phase == "diffractive":
        w = s.order * wl_um * 1e-3 * w
    return w


# --------------------------------------------------------------------------- tracing

def media(params, system: System, wl_um):
    """Index after each surface at wl_um (index before surface 0 is air)."""
    return [M.index(s.material, wl_um, p) for s, p in zip(system.surfaces, params)]


def trace_ray(params, system: System, wl_um, origin, direction, fr=None):
    """Trace one ray from `origin` (global, in air) to the image surface.

    Returns the global intercept at every surface `pts` [S, 3], the final direction,
    the optical path to every surface `opl` [S], the local intercepts `local` [S, 3],
    and `ok` (no miss / TIR / evanescent order anywhere).
    """
    fr = frames(params, system) if fr is None else fr
    pos, d = origin, direction
    opl = jnp.dot(origin, direction)  # plane-wave phase at the origin
    n1 = jnp.asarray(1.0)
    ok = jnp.asarray(True)
    pts, opls, locs = [], [], []
    medium = ("air", None)  # material and parameters of the medium the ray is in
    for s, p, (O, R) in zip(system.surfaces, params, fr):
        if s.shape == "cb":
            pts.append(pos)
            opls.append(opl)
            locs.append(R.T @ (pos - O))
            continue
        pl = R.T @ (pos - O)
        dl = R.T @ d
        t, hit = _intersect(s, p, pl, dl)
        q = pl + t * dl
        opl = opl + n1 * t
        ok = ok & hit
        n2 = M.index(s.material, wl_um, p)
        if s.interact != "none":
            N = _normal(s, p, q[0], q[1])
            v = n1 * dl
            if s.phase:
                W, (gx, gy) = jax.value_and_grad(
                    lambda xy: phase(s, p, xy[0], xy[1], wl_um))(q[:2])
                if s.phase == "glass" and medium[0] not in M.AIR:
                    # disperses like the glass it sits on: (n(lambda) - 1) / (n_d - 1)
                    kap = (n1 - 1.0) / (M.index(medium[0], LAMBDA_D, medium[1]) - 1.0)
                    W, gx, gy = kap * W, kap * gx, kap * gy
                opl = opl + W
                v = v + jnp.array([gx, gy, 0.0])
            vn = jnp.dot(v, N)
            vt = v - vn * N
            sg = jnp.where(jnp.dot(dl, N) >= 0, 1.0, -1.0)
            if s.interact == "reflect":
                arg = n1 * n1 - jnp.dot(vt, vt)
                dl = (vt - sg * _sqrt(arg) * N) / n1
                n2 = n1
            else:
                arg = n2 * n2 - jnp.dot(vt, vt)
                dl = (vt + sg * _sqrt(arg) * N) / n2
            ok = ok & (arg > 0)
        pos = O + R @ q
        d = R @ dl
        n1 = n2
        if s.interact == "refract":
            medium = (s.material, p)
        pts.append(pos)
        opls.append(opl)
        locs.append(q)
    return {"pts": jnp.stack(pts), "dir": d, "opl": jnp.stack(opls), "local": jnp.stack(locs),
            "ok": ok, "n_img": n1}


def field_direction(field_deg):
    tx, ty = jnp.tan(jnp.deg2rad(field_deg[0])), jnp.tan(jnp.deg2rad(field_deg[1]))
    d = jnp.array([tx, ty, 1.0])
    return d / jnp.linalg.norm(d)


def pupil_basis(d):
    """Orthonormal (e1, e2) perpendicular to d, e2 in the plane of d and y."""
    e2 = jnp.array([0.0, 1.0, 0.0]) - d[1] * d
    e2 = e2 / jnp.linalg.norm(e2)
    e1 = jnp.cross(e2, d)
    return e1, e2


def _stop_xy(xy0, params, system, field_deg, wl_um):
    d = field_direction(field_deg)
    o = jnp.array([xy0[0], xy0[1], system.z_launch])
    r = trace_ray(params, system, wl_um, o, d)
    i = system.stop
    return r["local"][i, :2] - jnp.array([params[i]["sx"], params[i]["sy"]])


@partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def _aim(params, field_deg, wl_um, xy_init, system, iters):
    def body(_, xy):
        f = _stop_xy(xy, params, system, field_deg, wl_um)
        J = jax.jacfwd(_stop_xy)(xy, params, system, field_deg, wl_um)
        J = J + 1e-12 * jnp.eye(2)
        return xy - jnp.linalg.solve(J, f)

    return jax.lax.fori_loop(0, iters, body, xy_init)


@_aim.defjvp
def _aim_jvp(system, iters, primals, tangents):
    """Implicit differentiation of the aiming condition f(xy, theta) = 0."""
    params, fd, wl, xy_init = primals
    dparams, dfd, dwl, _ = tangents
    xy = _aim(params, fd, wl, xy_init, system, iters)
    J = jax.jacfwd(_stop_xy)(xy, params, system, fd, wl) + 1e-12 * jnp.eye(2)
    _, ft = jax.jvp(lambda p_, f_, w_: _stop_xy(xy, p_, system, f_, w_), (params, fd, wl),
                    (dparams, dfd, dwl))
    return xy, -jnp.linalg.solve(J, ft)


def aim_chief(params, system: System, field_deg, wl_um, iters=8):
    """Launch-plane (x, y) of the ray through the stop's aiming point (sx, sy)."""
    fd = jnp.asarray(field_deg, float)
    fr = frames(params, system)
    O, _ = fr[system.stop]
    d = field_direction(fd)
    # straight-line start through the stop position (ignores the optics in front)
    xy0 = O[:2] - d[:2] / d[2] * (O[2] - system.z_launch)
    return _aim(params, fd, jnp.asarray(wl_um, float), xy0, system, iters)


def launch(xy_chief, field_deg, px, py, epd, system: System):
    """Ray of a collimated beam: the beam cross-section is a disk of diameter `epd`
    centered on the chief ray, sampled at normalized pupil coordinates (px, py)."""
    d = field_direction(jnp.asarray(field_deg, float))
    e1, e2 = pupil_basis(d)
    o = jnp.array([xy_chief[0], xy_chief[1], system.z_launch]) + 0.5 * epd * (px * e1 + py * e2)
    return o, d


def bundle(params, system: System, spec: Spec, field_deg, wl_um, px, py, xy_chief=None,
           ref_wl=None):
    """Trace a pupil sample (px, py arrays) at one field and wavelength.

    Returns image-surface local (x, y) [P, 2], wavefront OPD [P] in mm relative to the
    chief ray on the reference sphere (centered on the reference-wavelength chief image
    point, through the exit pupil), validity flags, and the chief-ray data.  OPD > 0
    means the wavefront leads (the usual sign).
    """
    ref_wl = spec.ref_wavelength if ref_wl is None else ref_wl
    if xy_chief is None:
        xy_chief = aim_chief(params, system, field_deg, ref_wl)
    fr = frames(params, system)

    def one(a, b, wl):
        o, d = launch(xy_chief, field_deg, a, b, spec.epd, system)
        return trace_ray(params, system, wl, o, d, fr)

    chief_ref = one(0.0, 0.0, ref_wl)
    chief = one(0.0, 0.0, wl_um)
    r = jax.vmap(lambda a, b: one(a, b, wl_um))(px, py)
    P = chief_ref["pts"][-1]
    last = _last_interacting(system)
    Rs = _exit_pupil_distance(params, system, spec, field_deg, ref_wl, last, P)

    def to_sphere(pos, d, opl, n):
        w = pos - P
        wd = jnp.dot(w, d)
        disc = jnp.maximum(wd * wd - (jnp.dot(w, w) - Rs * Rs), 0.0)
        return opl + n * (-wd - jnp.sqrt(disc))

    opl = jax.vmap(to_sphere, in_axes=(0, 0, 0, None))(r["pts"][:, last], r["dir"],
                                                        r["opl"][:, last], chief["n_img"])
    opl_c = to_sphere(chief["pts"][last], chief["dir"], chief["opl"][last], chief["n_img"])
    return {"img": r["local"][:, -1, :2], "opd": -(opl - opl_c), "ok": r["ok"],
            "local": r["local"],
            "dir": r["dir"], "pts": r["pts"], "chief_img": chief["local"][-1, :2],
            "chief_ref_img": chief_ref["local"][-1, :2], "xy_chief": xy_chief, "R_ref": Rs}


def _exit_pupil_distance(params, system, spec, field_deg, wl, last, P, r_max=1e5):
    """Distance from the image point P back to the exit pupil along the chief ray.

    The exit pupil center is where neighbouring chief rays (fields differing slightly
    in y) cross in image space: with the chief ray's image-space line x0 + t d0 and its
    derivative dx + t dd with respect to the field angle, it is at t* minimizing
    |dx + t dd|.  This depends on the rays only, so surfaces that do nothing (a dummy
    plane, a zero-power screen) do not change the reference sphere.  Telecentric image
    space (no crossing) is capped at r_max."""
    field_deg = jnp.asarray(field_deg, float)

    def chief_line(fy):
        f = field_deg.at[1].set(fy)
        xy = aim_chief(params, system, f, wl)
        o, d = launch(xy, f, 0.0, 0.0, spec.epd, system)
        r = trace_ray(params, system, wl, o, d)
        return r["pts"][last], r["dir"]

    (x0, d0), (dx, dd) = jax.jvp(chief_line, (field_deg[1],), (jnp.asarray(1.0),))
    den = jnp.dot(dd, dd)
    t = -jnp.dot(dx, dd) / jnp.where(den > 1e-30, den, 1e-30)
    E = x0 + t * d0
    R = jnp.linalg.norm(E - P)
    return jnp.where(den > 1e-30, jnp.minimum(R, r_max), r_max)


def _last_interacting(system: System):
    for i in range(system.n - 2, -1, -1):
        if system.surfaces[i].interact != "none":
            return i
    return 0


def focal_length(params, system: System, spec: Spec, wl_um=None, axis=1):
    """Effective focal length f = -dh / d(alpha): rays parallel to the zero-field chief
    ray, displaced by h in the pupil along `axis` (0: x, 1: y), and their image-space
    angle alpha to the chief ray, measured in the plane spanned by the chief ray and the
    displacement.  For a coaxial system this is the usual EFL; for off-axis sections it
    is the local focal length (e.g. the mirror-to-focus distance of an off-axis parabola).
    """
    wl = spec.ref_wavelength if wl_um is None else wl_um
    f0 = jnp.zeros(2)
    xy = aim_chief(params, system, f0, wl)

    def out_dir(h):
        pp = jnp.zeros(2).at[axis].set(h)
        o, d = launch(xy, f0, pp[0], pp[1], 2.0, system)  # epd 2: h in mm
        return trace_ray(params, system, wl, o, d)["dir"]

    d0, dd = jax.jvp(out_dir, (0.0,), (1.0,))
    # direction in which the image-space rays fan out (perpendicular to the chief ray),
    # oriented like the pupil displacement (unfolded through mirrors)
    e = dd - jnp.dot(dd, d0) * d0
    return 1.0 / jnp.linalg.norm(e) * jnp.where(_converging(params, system, xy, wl, axis),
                                                  1.0, -1.0)


def _converging(params, system, xy, wl, axis):
    """True when rays displaced in the pupil approach the chief ray after the system."""
    pp = jnp.zeros(2).at[axis].set(1e-3)
    o, d = launch(xy, jnp.zeros(2), pp[0], pp[1], 2.0, system)
    o0, d0 = launch(xy, jnp.zeros(2), 0.0, 0.0, 2.0, system)
    a = trace_ray(params, system, wl, o, d)
    b = trace_ray(params, system, wl, o0, d0)
    sep = a["pts"][-2] - b["pts"][-2]  # separation at the last surface before the image
    rel = a["dir"] - b["dir"]
    return jnp.dot(sep, rel) < 0
