"""Design diagnostics: Seidel / primary-chromatic surface contributions, real-ray
aberration fits, aberration-content measures and morph path gaps.

Seidel sums
-----------
Classic paraxial y-nu trace of a marginal ray (object at infinity, height EPD/2)
and a chief ray (through the stop center, angle tan(theta_max)), with Welford's
surface contributions

    S_I   = -A^2   y dU          S_IV = -H^2 c d(1/n)
    S_II  = -A Ab  y dU          S_V  = (Ab/A)(S_III + S_IV)
    S_III = -Ab^2  y dU          C_I  =  A  y d(dn/n),   C_II = Ab y d(dn/n)

(A = n(u + y c), Ab = n(ub + yb c), dU = u'/n' - u/n, H = Lagrange invariant), plus
the fourth-order aspheric term dS_I = 8 a4 (n' - n) y^4 and its stop-shifted
companions.  Wavefront coefficients follow as W040 = S_I/8, W131 = S_II/2,
W222 = S_III/2, W220 = (S_III + S_IV)/4 (sagittal), W311 = S_V/2, W020 = C_I/2,
W111 = C_II.  `seidel_waves` also reports the Petzval part S_IV/4 separately.

Ideal phase screens use the Sweatt model: a screen of power phi and r^4 phase b4 on
a substrate of curvature c is exactly the N -> infinity limit of a zero-thickness
lens of index N with curvatures c + phi/(N-1) and c, plus an r^4 asphere chosen
so its optical thickness (along the local normal, where rays travel as N -> inf)
reproduces W(x, y) exactly; see `paraxial_surfaces`.  Its Petzval contribution
phi/N vanishes, as it should.

The trace is written in jax.numpy: `element_waves` gives differentiable per-element
coefficients for the aberration-content penalty in the merit function.

The paraxial contributions are cross-checked against `aberration_fit`, a least
squares fit of the real-ray wavefront (Hopkins expansion through fifth order), in
the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import analysis as A
from . import glass as G
from .system import Layout, Spec

SWEATT_N = 1.0e6
SEIDEL_NAMES = ("SI", "SII", "SIII", "SIV", "SV", "CI", "CII")
W_NAMES = ("W040", "W131", "W222", "W220", "W311", "W020", "W111", "W220P")


@dataclass
class ParaxialSurface:
    label: str
    z: object  # float or traced scalar
    c: object
    n: dict  # wavelength -> index before
    n2: dict  # wavelength -> index after
    a4: object = 0.0  # fourth-order aspheric coefficient [1/mm^3]
    is_stop: bool = False
    element: int = -1  # owning element, -1 for the stop


def paraxial_surfaces(params, layout: Layout, spec: Spec, beta=1.0):
    """Flatten the system into paraxial surfaces (phase screens as Sweatt lenses).

    Written with jax.numpy so it can sit inside the merit function; the structure
    (which surfaces exist) depends only on the layout.  Every element gets a Sweatt
    pair; with zero phase its two surfaces cancel exactly.
    """
    p = params
    wls = list(spec.wavelengths)
    air = {w: 1.0 for w in wls}
    out = []
    z = 0.0
    N = SWEATT_N
    for k, item in enumerate(layout.items()):
        if item == "stop":
            out.append(ParaxialSurface("STOP", z, 0.0, air, air, is_stop=True))
        else:
            i = item
            glass = {w: G.index_at(p, i, w, beta, layout) for w in wls}
            nd_i = G.index_at(p, i, G.LAMBDA_D, 0.0, layout)
            out.append(ParaxialSurface(f"E{i}.1", z, p["c1"][i], air, glass, element=i))
            z = z + p["t"][i]
            a4 = p["asph"][i][0] / layout.r_norm**4 if layout.aspheres else 0.0
            out.append(ParaxialSurface(f"E{i}.2", z, p["c2"][i], glass, air, a4=a4, element=i))
            phi_ph = -2.0 * p["phase"][i][0] / layout.r_norm**2
            b4 = p["phase"][i][1] / layout.r_norm**4 if p["phase"].shape[1] > 1 else 0.0
            # phase disperses like the element's glass: N(lambda)-1 = (N-1) kappa
            kap = {w: (glass[w] - 1.0) / (nd_i - 1.0) for w in wls}
            sweatt = {w: 1.0 + (N - 1.0) * kap[w] for w in wls}
            ca, cb = p["c2"][i] + phi_ph / (N - 1.0), p["c2"][i]
            # Fourth-order term of the Sweatt lens.  For N -> inf a ray inside travels
            # along the local surface normal, which on a curved substrate is tilted by
            # ~c r, so W = -(N-1) dz (1 - c^2 r^2 / 2) with dz = sag_a - sag_b.
            # The two spheres already give dz an r^4 term (ca^3 - cb^3)/8; matching
            # W = -phi r^2/2 + b4 r^4 leaves a4 = -b4/(N-1) - cb^2 (ca - cb)/8.
            a4w = -b4 / (N - 1.0) - cb**2 * (ca - cb) / 8.0
            out.append(ParaxialSurface(f"E{i}.W", z, ca, air, sweatt, a4=a4w, element=i))
            out.append(ParaxialSurface(f"E{i}.W'", z, cb, sweatt, air, element=i))
        if k < layout.n_gaps:
            z = z + p["gaps"][k]
    return out


def _trace(surfs, wl, y0, u0):
    """Paraxial trace; returns heights y and angles u (before) / u' (after) per surface."""
    ys, us, ups = [], [], []
    y, u = y0, u0
    z_prev = surfs[0].z
    for s in surfs:
        y = y + (s.z - z_prev) * u
        z_prev = s.z
        n, n2 = s.n[wl], s.n2[wl]
        up = (n * u - y * s.c * (n2 - n)) / n2
        ys.append(y)
        us.append(u)
        ups.append(up)
        u = up
    return jnp.stack(ys), jnp.stack(us), jnp.stack(ups), y, u


def _seidel_core(params, layout: Layout, spec: Spec, beta=1.0, field_frac=1.0):
    """Traceable core: per-paraxial-surface contributions [n_surf, 7] and the traces."""
    surfs = paraxial_surfaces(params, layout, spec, beta)
    wls = list(spec.wavelengths)
    wd = spec.ref_wavelength
    wF, wC = wls[0], wls[-1]
    ymarg = spec.epd / 2
    ubar = np.tan(np.deg2rad(spec.half_field_deg) * field_frac)

    ym, um, upm, y_end, u_end = _trace(surfs, wd, ymarg, 0.0)
    # chief ray: combination of two base rays that passes through the stop center
    ks = next(j for j, s in enumerate(surfs) if s.is_stop)
    ya, *_ = _trace(surfs, wd, 1.0, 0.0)
    yb, *_ = _trace(surfs, wd, 0.0, 1.0)
    y0c = -ubar * yb[ks] / ya[ks]
    yc, uc, upc, *_ = _trace(surfs, wd, y0c, ubar)

    H = ubar * ymarg  # Lagrange invariant n (ub y - u yb) at the first surface (u = 0)
    rows = []
    for j, s in enumerate(surfs):
        if s.is_stop:
            rows.append(jnp.zeros(7))
            continue
        n, n2 = s.n[wd], s.n2[wd]
        Am = n * (um[j] + ym[j] * s.c)
        Ac = n * (uc[j] + yc[j] * s.c)
        dU = upm[j] / n2 - um[j] / n
        SI = -Am**2 * ym[j] * dU
        SII = -Am * Ac * ym[j] * dU
        SIII = -Ac**2 * ym[j] * dU
        SIV = -H**2 * s.c * (1.0 / n2 - 1.0 / n)
        # S_V = (Ab/A)(S_III + S_IV); A can vanish (collimated beam on a flat surface),
        # where S_III + S_IV vanishes too.  Safe division keeps values finite.
        Am_safe = jnp.where(jnp.abs(Am) > 1e-14, Am, 1e-14)
        SV = Ac / Am_safe * (SIII + SIV)
        # aspheric terms, written without dividing by y
        k4 = 8.0 * s.a4 * (n2 - n)
        SI = SI + k4 * ym[j] ** 4
        SII = SII + k4 * ym[j] ** 3 * yc[j]
        SIII = SIII + k4 * ym[j] ** 2 * yc[j] ** 2
        SV = SV + k4 * ym[j] * yc[j] ** 3
        dd = (s.n2[wF] - s.n2[wC]) / n2 - (s.n[wF] - s.n[wC]) / n
        CI = Am * ym[j] * dd
        CII = Ac * ym[j] * dd
        rows.append(jnp.stack([SI, SII, SIII, SIV, SV, CI, CII]))
    return surfs, jnp.stack(rows), dict(ym=ym, y_end=y_end, u_end=u_end, y0c=y0c, ubar=ubar,
                                        H=H, ymarg=ymarg)


def element_waves(params, layout: Layout, spec: Spec, beta=1.0):
    """Per-element wavefront coefficients [n_elem, len(W_NAMES)] in d-line waves.

    An element's contribution is the sum over its two glass surfaces and its phase
    screen.  Differentiable: used by the aberration-content penalty in the merit.
    """
    surfs, S, _ = _seidel_core(params, layout, spec, beta)
    lam = spec.ref_wavelength * 1e-3
    rows = []
    for i in range(layout.n_elem):
        idx = [j for j, s in enumerate(surfs) if s.element == i]
        rows.append(jnp.sum(S[jnp.asarray(idx)], axis=0))
    return _to_w(jnp.stack(rows), lam)


def seidel(params, layout: Layout, spec: Spec, beta=1.0, field_frac=1.0):
    """Per-surface Seidel and primary chromatic contributions (units: mm of OPD).

    Returns a dict with `labels`, `S` [n_surf, 7] in the order of SEIDEL_NAMES (each
    phase screen's Sweatt pair merged into one `E<i>.W` row), the totals, paraxial
    EFL/BFD, the Lagrange invariant and the exact paraxial color.
    """
    surfs, S, tr = _seidel_core(params, layout, spec, beta, field_frac)
    S = np.asarray(S)
    labels, rows = [], []
    for s, row in zip(surfs, S):
        if s.label.endswith("W'"):
            rows[-1] = rows[-1] + row
            continue
        labels.append(s.label)
        rows.append(row)
    S = np.array(rows)
    wls = list(spec.wavelengths)
    wd, wF, wC = spec.ref_wavelength, wls[0], wls[-1]
    u_end, y_end = float(tr["u_end"]), float(tr["y_end"])
    efl = -tr["ymarg"] / u_end
    bfd = -y_end / u_end

    # Exact paraxial color (all orders in dn): trace marginal and chief rays at F and C
    # and measure at the d-line paraxial image.  For well-corrected systems the total
    # first-order C_I, C_II is a small difference of large terms and the dn^2 terms
    # left out of it are not negligible.
    z_last = float(surfs[-1].z)
    z_img = z_last + bfd
    lam = wd * 1e-3
    out = {}
    for w in (wF, wC):
        _, _, _, yme, ume = _trace(surfs, w, tr["ymarg"], 0.0)
        _, _, _, yce, uce = _trace(surfs, w, tr["y0c"], tr["ubar"])
        out[w] = (float(-yme / ume), float(yce + (z_img - z_last) * uce))
    dz = out[wF][0] - out[wC][0]
    dy = out[wF][1] - out[wC][1]
    return {"labels": labels, "S": S, "total": S.sum(0), "efl": efl,
            "bfd_paraxial": bfd, "lagrange": float(tr["H"]), "axial_color_mm": dz,
            "lateral_color_mm": dy, "W020_exact": -dz * u_end**2 / 2 / lam,
            "W111_exact": dy * u_end / lam}


def _to_w(S, lam):
    SI, SII, SIII, SIV, SV, CI, CII = jnp.moveaxis(jnp.asarray(S), -1, 0)
    W = jnp.stack([SI / 8, SII / 2, SIII / 2, (SIII + SIV) / 4, SV / 2, CI / 2, CII, SIV / 4],
                  axis=-1)
    return W / lam


def seidel_waves(sd, spec: Spec):
    """Wavefront coefficients in d-line waves, in the order of W_NAMES.

    W220 is the sagittal field-curvature coefficient (S_III + S_IV)/4; W220P = S_IV/4
    is its Petzval part.  Per-surface rows and system totals are returned.
    """
    lam = spec.ref_wavelength * 1e-3
    return np.asarray(_to_w(sd["S"], lam)), np.asarray(_to_w(sd["total"], lam))


def element_contributions(sd):
    """Sum surface contributions per element (glass surfaces + its phase screen)."""
    groups = {}
    for lab, row in zip(sd["labels"], sd["S"]):
        if lab == "STOP":
            continue
        key = lab.split(".")[0]
        groups[key] = groups.get(key, 0) + row
    return groups


def aberration_content(sd, spec: Spec):
    """How much aberration the design cancels, per Seidel term.

    `abs_sum` is sum_j |W_j| over surfaces (waves); `cancellation` is abs_sum / |total|.
    Large abs_sum means strong surface contributions that cancel each other: such
    designs tend to have large high-order residuals and tight tolerances.
    """
    Wj, Wt = seidel_waves(sd, spec)
    abs_sum = np.abs(Wj).sum(0)
    return {"abs_sum": abs_sum, "total": Wt,
            "cancellation": abs_sum / np.maximum(np.abs(Wt), 1e-6)}


# ------------------------------------------------------------------ real-ray fit

FIT_TERMS = (
    # (name, k, l, m): H^k rho^l cos^m(phi)
    ("W020", 0, 2, 0), ("W111", 1, 1, 1),
    ("W040", 0, 4, 0), ("W131", 1, 3, 1), ("W222", 2, 2, 2), ("W220", 2, 2, 0),
    ("W311", 3, 1, 1),
    ("W060", 0, 6, 0), ("W151", 1, 5, 1), ("W242", 2, 4, 2), ("W240", 2, 4, 0),
    ("W333", 3, 3, 3), ("W331", 3, 3, 1), ("W422", 4, 2, 2), ("W420", 4, 2, 0),
    ("W511", 5, 1, 1),
)


def aberration_fit(params, layout: Layout, spec: Spec, beta=1.0, n_field=9, n_rings=8,
                   n_arms=16, terms=FIT_TERMS):
    """Least-squares Hopkins-expansion fit of the real-ray wavefront, per wavelength.

    Returns coefficients [n_wl, n_terms] in d-line waves (full field, full aperture),
    the term names, and the RMS of the fit residual (orders above fifth) per wavelength.
    Piston is fitted per field.  W311 (distortion) is not observable here: the
    wavefront is referenced to the chief-ray image point, which absorbs it.
    """
    fracs = tuple(np.linspace(0, 1, n_field))
    ev = A.evaluate(params, layout, spec, beta, fracs=fracs, n_rings=n_rings, n_arms=n_arms)
    px, py, wt = A.pupil_grid(n_rings, n_arms)
    m = wt > 0
    rho = np.sqrt(px[m] ** 2 + py[m] ** 2)
    cphi = py[m] / rho
    Hs = np.array(fracs)
    rows, piston = [], []
    for f, Hf in enumerate(Hs):
        for j in range(m.sum()):
            rows.append([Hf**k * rho[j] ** l * cphi[j] ** mm for _, k, l, mm in terms])
            e = np.zeros(n_field)
            e[f] = 1.0
            piston.append(e)
    M = np.c_[np.array(rows), np.array(piston)]
    # report everything in waves of the reference wavelength
    wl = np.asarray(spec.wavelengths)
    opd = np.asarray(ev["opd"])[:, :, m] * (wl / spec.ref_wavelength)[:, None, None]
    coefs, resid = [], []
    for w in range(opd.shape[0]):
        b = opd[w].reshape(-1)
        c, *_ = np.linalg.lstsq(M, b, rcond=None)
        coefs.append(c[: len(terms)])
        resid.append(np.sqrt(np.mean((M @ c - b) ** 2)))
    return {"names": [t[0] for t in terms], "coef": np.array(coefs), "resid_rms": np.array(resid)}


# ------------------------------------------------------------------ reporting

def seidel_table(params, layout: Layout, spec: Spec, beta=1.0) -> str:
    sd = seidel(params, layout, spec, beta)
    Wj, Wt = seidel_waves(sd, spec)
    head = f"{'surface':<9}" + "".join(f"{n:>9}" for n in W_NAMES)
    lines = [head + "   [waves, d line; color terms F-C]"]
    for lab, row in zip(sd["labels"], Wj):
        if lab == "STOP" or not np.any(np.abs(row) > 1e-9):
            continue
        lines.append(f"{lab:<9}" + "".join(f"{v:9.3f}" for v in row))
    lines.append(f"{'TOTAL':<9}" + "".join(f"{v:9.3f}" for v in Wt))
    ac = aberration_content(sd, spec)
    lines.append(f"{'sum|.|':<9}" + "".join(f"{v:9.3f}" for v in ac["abs_sum"]))
    lines.append(f"exact paraxial color: axial F-C {1e3 * sd['axial_color_mm']:+.1f} um "
                 f"(W020 {sd['W020_exact']:+.3f}), lateral F-C {1e3 * sd['lateral_color_mm']:+.2f} um "
                 f"(W111 {sd['W111_exact']:+.3f})")
    return "\n".join(lines)


def fit_table(params, layout: Layout, spec: Spec, beta=1.0) -> str:
    fit = aberration_fit(params, layout, spec, beta)
    names = fit["names"]
    lines = [f"{'':<4}" + "".join(f"{n:>8}" for n in names) + f"{'resid':>8}"]
    for w, lab in zip(range(fit["coef"].shape[0]), "FdC"):
        lines.append(f"{lab:<4}" + "".join(f"{v:8.3f}" for v in fit["coef"][w])
                     + f"{fit['resid_rms'][w]:8.3f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ morph path gap

def path_gap(snap, spec: Spec, iters=300):
    """How far a recorded morph state is from optimal for its own homotopy problem.

    Re-optimizes the snapshot's design variables z at the same s and cfg:

    * `converged`: same parameterization, many more iterations.  The gap between the
      recorded and converged merit is lack of convergence along the path.
    * `free_curvature`: also frees independent curvature corrections dc1, dc2 on top
      of the thin-lens rule (if the morph does not already carry them).  The extra
      drop measures how restrictive the morph parameterization is.

    Merit values are the optimizer's cost (sum of squared residuals) and the mean
    polychromatic RMS wavefront error on the snapshot's design image surface.
    """
    from .continuation import morph_to_params
    from .merit import make_cfg, residuals, virtual_petzval_curvature
    from .optimize import optimize

    z0 = snap.extra["z"]
    cfg = make_cfg(**snap.extra["cfg"])
    lay = snap.layout

    def measure(z):
        p = morph_to_params(z, cfg, lay)
        r = residuals(p, lay, spec, cfg)
        c_img = snap.extra["cfg"]["virtual_petzval"] * float(virtual_petzval_curvature(p, lay))
        return float(r @ r), A.summary(p, lay, spec, 1.0, c_img=c_img)["mean_wfe"]

    out = {"s": snap.extra["cfg"]["s"], "recorded": measure(z0)}
    z1, _ = optimize(dict(z0), lay, spec, cfg, {k: True for k in z0}, iters=iters,
                     transform=morph_to_params)
    out["converged"] = measure(z1)
    if "dc1" in z1:
        # the morph already carries free curvature corrections: nothing to add
        out["free_curvature"] = out["converged"]
    else:
        z2 = dict(z1, dc1=jnp.zeros(lay.n_elem), dc2=jnp.zeros(lay.n_elem))
        z2, _ = optimize(z2, lay, spec, cfg, {k: True for k in z2}, iters=iters,
                         transform=morph_to_params)
        out["free_curvature"] = measure(z2)
    return out
