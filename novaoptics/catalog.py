"""Real glass catalogs (Zemax AGF) and a PCA glass space.

AGF records used here::

    NM <name> <formula> <MIL> <nd> <vd> <exclude> <status> [<melt freq>]
    CD <c0> <c1> ... <c9>        dispersion coefficients (wavelength in um)
    LD <lambda_min> <lambda_max> validity range of the dispersion formula [um]

Dispersion formulas: 1 = Schott (n^2 = a0 + a1 L^2 + a2 L^-2 + a3 L^-4 + a4 L^-6
+ a5 L^-8), 2 = Sellmeier 1 (n^2 - 1 = sum K_i L^2 / (L^2 - L_i)).  Status codes
follow Zemax: 0 standard, 1 preferred, 2 obsolete, 3 special, 4 melt.

PCA glass space
---------------
Each glass is the vector of its indices x_g = n_g(lambda_k) on a wavelength grid
covering the design band.  With mean mu and leading principal directions B,

    x(a) = mu + B a                (a: a few continuous glass coordinates)

The first coordinate is essentially the index level, the second the dispersion
and the third the partial dispersion (secondary spectrum), which an (n_d, V_d)
Cauchy model cannot represent at all.  A real glass g is exactly
x_g = mu + B a_g + r_g with a small residual r_g outside the span of B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATUS = {0: "standard", 1: "preferred", 2: "obsolete", 3: "special", 4: "melt"}


@dataclass
class Glass:
    name: str
    formula: int
    nd: float
    vd: float
    status: int
    coef: np.ndarray
    lmin: float = 0.0
    lmax: float = 100.0

    def index(self, wl_um):
        L2 = np.asarray(wl_um, dtype=float) ** 2
        c = self.coef
        if self.formula == 2:  # Sellmeier 1
            n2 = 1.0 + c[0] * L2 / (L2 - c[1]) + c[2] * L2 / (L2 - c[3]) + c[4] * L2 / (L2 - c[5])
        elif self.formula == 1:  # Schott
            n2 = c[0] + c[1] * L2 + c[2] / L2 + c[3] / L2**2 + c[4] / L2**3 + c[5] / L2**4
        else:
            raise NotImplementedError(f"AGF dispersion formula {self.formula} ({self.name})")
        return np.sqrt(n2)


def read_agf(path) -> list[Glass]:
    """Parse a Zemax AGF file (CRLF or LF; UTF-8/Latin-1 tolerant)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    glasses, cur = [], None
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "NM":
            if cur is not None:
                glasses.append(cur)
            status = int(float(tok[7])) if len(tok) > 7 else 0
            cur = Glass(name=tok[1], formula=int(float(tok[2])), nd=float(tok[4]),
                        vd=float(tok[5]), status=status, coef=np.zeros(10))
        elif cur is not None and tok[0] == "CD":
            vals = [float(v) for v in tok[1:]]
            cur.coef[: len(vals)] = vals
        elif cur is not None and tok[0] == "LD":
            cur.lmin, cur.lmax = float(tok[1]), float(tok[2])
    if cur is not None:
        glasses.append(cur)
    return glasses


@dataclass
class GlassSpace:
    """Catalog glasses sampled on a wavelength grid, with their PCA coordinates.

    Hashable by identity, so it can live in the (hashable) `Layout`.
    """

    names: list
    nd: np.ndarray
    vd: np.ndarray
    status: np.ndarray
    grid: np.ndarray  # wavelengths [um]
    X: np.ndarray  # [n_glass, n_grid] indices
    mu: np.ndarray
    B: np.ndarray  # [n_grid, k]
    coords: np.ndarray  # [n_glass, k] PCA coordinates of the catalog glasses
    resid: np.ndarray  # [n_glass, n_grid] part of each glass outside span(B)
    explained: np.ndarray  # variance fraction per component (all components)
    scale: np.ndarray = field(default=None)  # per-coordinate std over the catalog

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def k(self):
        return self.B.shape[1]

    def grid_index(self, wl):
        """Position of a design wavelength on the grid (wavelengths must be on it)."""
        j = int(np.argmin(np.abs(self.grid - wl)))
        if abs(self.grid[j] - wl) > 1e-9:
            raise ValueError(f"wavelength {wl} um is not on the glass-space grid")
        return j

    def project(self, x):
        return (np.asarray(x) - self.mu) @ self.B

    def find(self, name):
        return self.names.index(name)


def build_glass_space(glasses, wavelengths, k=3, n_grid=16, statuses=(0, 1)) -> GlassSpace:
    """PCA of catalog indices over the band spanned by `wavelengths` (plus those exact
    wavelengths, so the design wavelengths are evaluated without interpolation)."""
    wl = np.asarray(wavelengths, dtype=float)
    lo, hi = wl.min(), wl.max()
    grid = np.unique(np.r_[np.linspace(lo, hi, n_grid), wl])
    keep = [g for g in glasses if g.status in statuses and g.lmin <= lo + 1e-9
            and g.lmax >= hi - 1e-9]
    X = np.array([g.index(grid) for g in keep])
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    B = Vt[:k].T
    coords = (X - mu) @ B
    resid = X - mu - coords @ B.T
    explained = S**2 / np.sum(S**2)
    return GlassSpace(names=[g.name for g in keep], nd=np.array([g.nd for g in keep]),
                      vd=np.array([g.vd for g in keep]), status=np.array([g.status for g in keep]),
                      grid=grid, X=X, mu=mu, B=B, coords=coords, resid=resid,
                      explained=explained, scale=coords.std(0))
