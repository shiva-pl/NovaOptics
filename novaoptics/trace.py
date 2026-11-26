"""Differentiable real-ray tracer.

Every interface is handled by one generalized law that covers refraction and an
ideal phase screen at the same time (the "grating equation" form of Snell's law):

    tangential( n' d' ) = tangential( n d ) + grad_s W

and the optical path picks up W at the screen.  With W = 0 this is ordinary
vector Snell; with flat surfaces and n = n' it is a pure phase screen.  This is
what lets an element continuously morph from "ideal phase" to "real glass".

Everything is written for a single ray and vectorized with `jax.vmap`; surface
intersections for spheres use Welford's form, which is stable as c -> 0.
Rays that would fail (missed sphere, TIR) are clamped instead of producing NaN,
so gradients stay finite; the merit function sees the resulting large errors.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import glass as G
from .system import Layout

_EPS = 1e-12


def _safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, _EPS))


def phase_and_grad(coef, x, y, r_norm):
    """W(x, y) and its transverse gradient for W = sum a_k (r^2/r_norm^2)^k."""
    rho = (x * x + y * y) / r_norm**2
    # powers rho^0..rho^K by repeated multiplication (rho**0 has a NaN gradient at 0)
    pw = [jnp.ones_like(rho)]
    for _ in range(coef.shape[0]):
        pw.append(pw[-1] * rho)
    pw = jnp.stack(pw)
    k = jnp.arange(1, coef.shape[0] + 1)
    W = jnp.sum(coef * pw[1:])
    dW_drho = jnp.sum(coef * k * pw[:-1])
    gx = 2.0 * x / r_norm**2 * dW_drho
    gy = 2.0 * y / r_norm**2 * dW_drho
    return W, gx, gy


def asphere_poly(asph, r2, r_norm):
    """Even aspheric departure sum_k A_k (r^2/r_norm^2)^k, k = 2..K, and d/d(r^2)."""
    rho = r2 / r_norm**2
    pw = [jnp.ones_like(rho)]
    for _ in range(asph.shape[0] + 1):
        pw.append(pw[-1] * rho)
    k = jnp.arange(2, asph.shape[0] + 2)
    val = jnp.sum(asph * jnp.stack(pw[2:]))
    dval = jnp.sum(asph * k * jnp.stack(pw[1:-1])) / r_norm**2
    return val, dval


def surface_sag(c, asph, r2, r_norm):
    """Sag of a conic-free even asphere at squared radius r2, and d(sag)/d(r^2)."""
    root = _safe_sqrt(1.0 - c * c * r2)
    sag = c * r2 / (1.0 + root)
    dsag = c / (2.0 * root)
    if asph is not None:
        v, dv = asphere_poly(asph, r2, r_norm)
        sag, dsag = sag + v, dsag + dv
    return sag, dsag


def interface(pos, d, opl, zv, c, n1, n2, phase=None, r_norm=1.0, asph=None, newton=4):
    """Propagate to a surface at vertex `zv` and cross it.

    The surface is a sphere of curvature `c`, optionally with even aspheric terms
    `asph` (found by Newton iteration from the sphere intercept).  `phase` adds an
    ideal phase screen on the surface.

    Returns new position, direction (unit), accumulated optical path, radial height.
    """
    t0 = (zv - pos[2]) / d[2]
    p0 = pos + t0 * d
    x, y = p0[0], p0[1]
    F = c * (x * x + y * y)
    Gq = d[2] - c * (x * d[0] + y * d[1])
    s = F / (Gq + _safe_sqrt(Gq * Gq - c * F))
    if asph is not None:
        def newton_step(_, s):
            q = p0 + s * d
            r2 = q[0] ** 2 + q[1] ** 2
            sag, dsag = surface_sag(c, asph, r2, r_norm)
            f = q[2] - zv - sag
            fp = d[2] - 2.0 * dsag * (q[0] * d[0] + q[1] * d[1])
            return s - f / fp

        s = jax.lax.fori_loop(0, newton, newton_step, s)  # a loop, not unrolled
    p1 = p0 + s * d
    opl = opl + n1 * (t0 + s)

    if asph is None:
        zl = p1[2] - zv
        N = jnp.array([-c * p1[0], -c * p1[1], 1.0 - c * zl])
    else:
        _, dsag = surface_sag(c, asph, p1[0] ** 2 + p1[1] ** 2, r_norm)
        N = jnp.array([-2.0 * dsag * p1[0], -2.0 * dsag * p1[1], 1.0])
    N = N / jnp.linalg.norm(N)

    v = n1 * d
    if phase is not None:
        W, gx, gy = phase_and_grad(phase, p1[0], p1[1], r_norm)
        opl = opl + W
        v = v + jnp.array([gx, gy, 0.0])
    vt = v - jnp.dot(v, N) * N
    vn = _safe_sqrt(n2 * n2 - jnp.dot(vt, vt))
    d_new = (vt + vn * N) / n2
    h = jnp.sqrt(p1[0] ** 2 + p1[1] ** 2 + _EPS)
    return p1, d_new, opl, h


def trace_ray(params, layout: Layout, wavelength, beta, origin, direction, c_img=0.0):
    """Trace one ray from the launch plane to the image surface.

    The image surface is a sphere of curvature `c_img` (0 = flat).

    Returns a dict with image-plane position, final direction, optical path to the
    last surface (`opl_last`) and its position (`pos_last`), stop intercept and
    the radial height at every refracting surface (2 per element, in order).
    """
    pos, d = origin, direction
    # The launch plane is not a wavefront for tilted beams: start every ray with the
    # optical path of the incident plane wave at its origin.
    opl = jnp.dot(origin, direction)
    z = 0.0
    heights = []
    pts = [origin]
    stop_xy = jnp.zeros(2)
    for k, item in enumerate(layout.items()):
        if item == "stop":
            t = (z - pos[2]) / d[2]
            pos = pos + t * d
            opl = opl + t
            stop_xy = pos[:2]
            pts += [pos]
        else:
            i = item
            n = G.index_at(params, i, wavelength, beta, layout)
            nd_i = G.index_at(params, i, G.LAMBDA_D, 0.0, layout)
            # Phase screens disperse like a thin lens of the element's glass:
            # W(lambda) = W * (n(lambda) - 1) / (n_d - 1).  With beta = 0 they are achromatic.
            kappa = (n - 1.0) / (nd_i - 1.0)
            pos, d, opl, h1 = interface(pos, d, opl, z, params["c1"][i], 1.0, n)
            pts += [pos]
            z = z + params["t"][i]
            asph = params["asph"][i] if layout.aspheres else None
            pos, d, opl, h2 = interface(
                pos, d, opl, z, params["c2"][i], n, 1.0, params["phase"][i] * kappa, layout.r_norm,
                asph=asph,
            )
            heights += [h1, h2]
            pts += [pos]  # rear-surface intercept
        if k < layout.n_gaps:
            z = z + params["gaps"][k]
    z_img = z + params["bfd"]
    t0 = (z_img - pos[2]) / d[2]
    p0 = pos + t0 * d
    F = c_img * (p0[0] ** 2 + p0[1] ** 2)
    Gq = d[2] - c_img * (p0[0] * d[0] + p0[1] * d[1])
    img = p0 + F / (Gq + _safe_sqrt(Gq * Gq - c_img * F)) * d
    return {
        "img": img,
        "dir": d,
        "pos_last": pos,
        "opl_last": opl,
        "stop_xy": stop_xy,
        "heights": jnp.stack(heights) if heights else jnp.zeros(0),
        "path": jnp.stack(pts + [img]),  # every intercept, for drawing
    }


def launch(field_angle, y0, px, py, radius, layout: Layout):
    """Ray in a collimated beam at field angle `field_angle` (rad, in the y-z plane).

    (px, py) are normalized pupil coordinates; the beam cross-section perpendicular
    to the propagation direction is a disk of `radius` centered on the chief ray,
    whose launch-plane height is `y0`.
    """
    d = jnp.array([0.0, jnp.sin(field_angle), jnp.cos(field_angle)])
    o = jnp.array([px * radius, y0 + py * radius / jnp.cos(field_angle), layout.z_launch])
    return o, d


def _stop_y(y0, params, layout: Layout, field_angle, wavelength, beta):
    o, d = launch(field_angle, y0, 0.0, 0.0, 0.0, layout)
    return trace_ray(params, layout, wavelength, beta, o, d)["stop_xy"][1]


@partial(jax.custom_jvp, nondiff_argnums=(5, 6))
def _aim(params, field_angle, wavelength, beta, y_init, layout, iters):
    def body(_, y0):
        val, slope = jax.jvp(lambda y: _stop_y(y, params, layout, field_angle, wavelength, beta),
                             (y0,), (jnp.ones_like(y0),))
        return y0 - val / jnp.where(jnp.abs(slope) > 1e-9, slope, 1e-9)

    return jax.lax.fori_loop(0, iters, body, y_init)


@_aim.defjvp
def _aim_jvp(layout, iters, primals, tangents):
    """Implicit differentiation: y solves f(y, theta) = 0, so dy = -(df/dtheta) / (df/dy).

    Differentiating through the Newton iterations instead would make the compiled
    Jacobian of the merit several times larger (and slower to compile)."""
    params, fa, wl, beta, y_init = primals
    dparams, dfa, dwl, dbeta, _ = tangents
    y = _aim(params, fa, wl, beta, y_init, layout, iters)
    _, f_y = jax.jvp(lambda y_: _stop_y(y_, params, layout, fa, wl, beta), (y,),
                     (jnp.ones_like(y),))
    _, f_t = jax.jvp(lambda p_, a_, w_, b_: _stop_y(y, p_, layout, a_, w_, b_),
                     (params, fa, wl, beta), (dparams, dfa, dwl, dbeta))
    return y, -f_t / jnp.where(jnp.abs(f_y) > 1e-9, f_y, 1e-9)


def aim_chief(params, layout: Layout, field_angle, wavelength, beta=1.0, iters=6):
    """Find the launch height whose ray passes through the center of the stop.

    Newton iterations in a loop (not unrolled), differentiated implicitly."""
    # Initial guess: straight line to the stop vertex.
    z_stop = 0.0
    for k, item in enumerate(layout.items()):
        if item == "stop":
            break
        z_stop = z_stop + params["t"][item]
        z_stop = z_stop + params["gaps"][k]
    y_init = -jnp.tan(field_angle) * (z_stop - layout.z_launch)
    f = jnp.asarray
    return _aim(params, f(field_angle, float), f(wavelength, float), f(beta, float),
                jnp.asarray(y_init, float), layout, iters)
