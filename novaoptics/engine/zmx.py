"""Zemax OpticStudio .zmx files (sequential mode): import into the surface engine and
export from it.

Supported:

* encodings: UTF-16 (with BOM, as OpticStudio writes them), UTF-8, Latin-1
* UNIT (lens units; everything is converted to mm), ENPD or FNUM (image-space F/#),
  WAVM / WAVL + WWGT (wavelengths, weights), PWAV, angle fields (FTYP 0: XFLN, YFLN,
  FWGN)
* surfaces: STANDARD, EVENASPH (PARM 1..8: r^2 .. r^16), COORDBRK (PARM 1..6:
  decenter x, y, tilt x, y, z, order), DGRATING (PARM 1 lines/um, PARM 2 order),
  PARAXIAL (PARM 1 focal length, as a quadratic ideal phase); CURV, DISZ, CONI, GLAS
  (catalog name, MIRROR, or a model glass with n_d, V_d), STOP, DIAM, COMM
* object at infinity

Anything else (other surface types, object at a finite distance, other field types,
multi-configuration data) raises `ZmxError` naming it, instead of being silently
dropped.  Tilt signs follow the engine's right-handed convention; check fold mirrors
of imported systems.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

import numpy as np

from .system import Builder, Spec

UNITS = {"MM": 1.0, "CM": 10.0, "IN": 25.4, "METER": 1000.0, "M": 1000.0}


class ZmxError(ValueError):
    pass


def read_text(path) -> str:
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8")
    if len(raw) > 1 and raw[1:2] == b"\x00":  # UTF-16 LE without BOM
        return raw.decode("utf-16-le")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _tok(line):
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _f(x):
    return np.inf if str(x).upper().startswith("INF") else float(x)


@dataclass
class _Surf:
    index: int
    type: str = "STANDARD"
    curv: float = 0.0
    disz: float = 0.0
    conic: float = 0.0
    glass: str = ""
    nd: float = None
    vd: float = None
    parm: dict = field(default_factory=dict)
    diam: float = None
    stop: bool = False
    comment: str = ""


def parse(text: str):
    """Raw records: global settings and the surface list."""
    g = {"unit": "MM", "waves": [], "pwav": None, "xfln": [], "yfln": [], "fwgn": [],
         "ftyp": None, "enpd": None, "fnum": None, "name": "", "mce": False, "wavl": None,
         "wwgt": None}
    surfs, cur = [], None
    for line in text.splitlines():
        t = _tok(line.strip())
        if not t:
            continue
        key, args = t[0].upper(), t[1:]
        indented = line[:1] in (" ", "\t")
        if key == "SURF":
            cur = _Surf(int(args[0]))
            surfs.append(cur)
            continue
        if cur is not None and indented:
            if key == "TYPE":
                cur.type = args[0].upper()
            elif key == "CURV":
                cur.curv = _f(args[0])
            elif key == "DISZ":
                cur.disz = _f(args[0])
            elif key == "CONI":
                cur.conic = _f(args[0])
            elif key == "GLAS":
                cur.glass = args[0]
                if len(args) >= 5:
                    try:
                        cur.nd, cur.vd = float(args[3]), float(args[4])
                    except ValueError:
                        pass
            elif key == "PARM":
                cur.parm[int(args[0])] = _f(args[1])
            elif key == "DIAM":
                cur.diam = _f(args[0])
            elif key == "STOP":
                cur.stop = True
            elif key == "COMM":
                cur.comment = " ".join(args)
            continue
        cur = None if not indented else cur
        if key == "UNIT":
            g["unit"] = args[0].upper()
        elif key == "NAME":
            g["name"] = " ".join(args)
        elif key == "ENPD":
            g["enpd"] = float(args[0])
        elif key == "FNUM":
            g["fnum"] = float(args[0])
        elif key in ("OBNA", "FLOA"):
            raise ZmxError(f"aperture type {key} is not supported (use ENPD or FNUM)")
        elif key == "WAVM":
            g["waves"].append((int(args[0]), float(args[1]), float(args[2]) if len(args) > 2
                               else 1.0))
        elif key == "WAVL":
            g["wavl"] = [float(a) for a in args]
        elif key == "WWGT":
            g["wwgt"] = [float(a) for a in args]
        elif key == "PWAV":
            g["pwav"] = int(args[0])
        elif key == "FTYP":
            g["ftyp"] = [int(float(a)) for a in args]
        elif key == "XFLN":
            g["xfln"] = [float(a) for a in args]
        elif key == "YFLN":
            g["yfln"] = [float(a) for a in args]
        elif key == "FWGN":
            g["fwgn"] = [float(a) for a in args]
        elif key in ("MNUM",):
            if int(args[0]) > 1:
                g["mce"] = True
    return g, surfs


def load(path):
    """(System, list of surface parameter dicts, Spec, info) from a .zmx file."""
    return from_text(read_text(path))


def from_text(text):
    g, surfs = parse(text)
    if g["mce"]:
        raise ZmxError("the file has multi-configuration data, which is not imported yet")
    scale = UNITS.get(g["unit"])
    if scale is None:
        raise ZmxError(f"lens unit {g['unit']} is not supported")
    if not surfs or not np.isinf(surfs[0].disz):
        raise ZmxError("only objects at infinity are supported (object DISZ INFINITY)")
    # wavelengths (um)
    if g["waves"]:
        n_w = None
        if g["ftyp"] and len(g["ftyp"]) > 3:
            n_w = g["ftyp"][3]
        waves = sorted(g["waves"])[:n_w] if n_w else sorted(g["waves"])
        wls = tuple(w for _, w, _ in waves)
        wts = tuple(x for _, _, x in waves)
    elif g["wavl"]:
        wls = tuple(w for w in g["wavl"] if w > 0)
        wts = tuple((g["wwgt"] or [1.0] * len(wls))[:len(wls)])
    else:
        wls, wts = (0.5875618,), (1.0,)
    ref = (g["pwav"] - 1) if g["pwav"] else len(wls) // 2
    # fields (angles in degrees)
    ftype = g["ftyp"][0] if g["ftyp"] else 0
    if ftype != 0:
        raise ZmxError(f"field type {ftype} is not supported (angles only)")
    n_f = g["ftyp"][2] if g["ftyp"] and len(g["ftyp"]) > 2 else max(len(g["yfln"]), 1)
    xs = (g["xfln"] + [0.0] * n_f)[:n_f]
    ys = (g["yfln"] + [0.0] * n_f)[:n_f]
    fw = (g["fwgn"] + [1.0] * n_f)[:n_f] if g["fwgn"] else [1.0] * n_f
    fields = tuple((float(x), float(y)) for x, y in zip(xs, ys))

    b = Builder()
    medium = "air"
    stop_seen = False
    for s in surfs[1:]:
        last = s is surfs[-1]
        t = 0.0 if last or np.isinf(s.disz) else s.disz * scale
        c = s.curv / scale
        common = dict(t=t, label=s.comment or f"S{s.index}")
        if s.stop:
            stop_seen = True
        if s.type == "COORDBRK":
            p = s.parm
            b.cb(dx=p.get(1, 0.0) * scale, dy=p.get(2, 0.0) * scale, tx=p.get(3, 0.0),
                 ty=p.get(4, 0.0), tz=p.get(5, 0.0), t=t, order=int(p.get(6, 0)),
                 label=common["label"])
            continue
        g_up = s.glass.upper()
        if g_up == "MIRROR":
            interact, material = "reflect", medium
        elif last:
            interact, material = "none", medium
        else:
            interact = "refract"
            if not s.glass:
                material = "air"
            elif s.glass.startswith("___BLANK") or g_up == "MODEL":
                material = "model"
            else:
                material = s.glass
        values = dict(common, c=c, k=s.conic)
        kw = {}
        if material == "model":
            values.update(nd=s.nd or 1.5, vd=s.vd or 50.0)
        if s.type in ("STANDARD", ""):
            pass
        elif s.type == "EVENASPH":
            a = [s.parm.get(k, 0.0) for k in range(1, 9)]
            values["a2"] = a[0] * scale ** (1 - 2)
            coef = [a[k - 1] * scale ** (1 - 2 * k) for k in range(2, 9)]
            while coef and coef[-1] == 0.0:
                coef.pop()
            kw.update(shape="even", r_norm=1.0)
            if coef:
                values["coef"] = coef
        elif s.type == "DGRATING":
            kw.update(phase="grating", order=int(s.parm.get(2, 1)))
            values["freq"] = s.parm.get(1, 0.0) * 1000.0 / scale * 1.0  # lines/um -> /mm
        elif s.type == "PARAXIAL":
            f = s.parm.get(1, 0.0) * scale
            rn = (s.diam * scale) if s.diam else 10.0
            kw.update(phase="ideal", r_norm=rn)
            values["pcoef"] = [-(rn**2) / (2 * f) if f else 0.0]
        else:
            raise ZmxError(f"surface {s.index}: type {s.type} is not supported")
        if last:
            b.image(**{k: v for k, v in values.items() if k in ("c", "k", "t", "label")})
        elif s.stop and interact == "refract" and material == "air" and c == 0.0 and \
                s.type in ("STANDARD", "") and medium == "air":
            b.stop(t=t, label=values["label"])
        else:
            b.add(interact=interact, material=material, stop=s.stop, **kw, **values)
        medium = material
    if not stop_seen:
        raise ZmxError("no stop surface")
    system, params = b.build()
    spec = Spec(epd=(g["enpd"] or 1.0) * scale, fields=fields, wavelengths=wls,
                wl_weights=wts, field_weights=tuple(fw), ref=ref)
    info = {"name": g["name"], "fnum": g["fnum"], "unit": g["unit"]}
    if g["enpd"] is None and g["fnum"]:
        from .trace import focal_length

        f = abs(float(focal_length(params, system, spec)))
        spec = Spec(epd=f / g["fnum"], fields=fields, wavelengths=wls, wl_weights=wts,
                    field_weights=tuple(fw), ref=ref)
    return system, params, spec, info


# --------------------------------------------------------------------------- export

def to_text(system, params, spec: Spec, name="NovaOptics design"):
    """Sequential .zmx text (lens units mm) for a single-configuration engine system."""
    L = ["VERS 190513 80 123457 L123457", "MODE SEQ", f"NAME {name}", "PFIL 0 0 0",
         "UNIT MM X W X CM MR CPMM", f"ENPD {spec.epd:.10g}", "GCAT SCHOTT", "RAIM 0 0 1 1 0 0 0 0 0",
         f"FTYP 0 0 {len(spec.fields)} {len(spec.wavelengths)} 0 0 0",
         "XFLN " + " ".join(f"{f[0]:.10g}" for f in spec.fields),
         "YFLN " + " ".join(f"{f[1]:.10g}" for f in spec.fields),
         "FWGN " + " ".join(f"{w:.10g}" for w in spec.fweights)]
    for k, (w, wt) in enumerate(zip(spec.wavelengths, spec.wl_weights), start=1):
        L.append(f"WAVM {k} {w:.10g} {wt:.10g}")
    L.append(f"PWAV {spec.iref + 1}")
    L += ["SURF 0", "  TYPE STANDARD", "  CURV 0.0", "  DISZ INFINITY"]
    for i, (s, p) in enumerate(zip(system.surfaces, params), start=1):
        L.append(f"SURF {i}")
        if s.label:
            L.append(f'  COMM "{s.label}"')
        if s.stop:
            L.append("  STOP")
        t = float(p["t"])
        if s.shape == "cb":
            L.append("  TYPE COORDBRK")
            for k, key in enumerate(("dx", "dy", "tx", "ty", "tz"), start=1):
                L.append(f"  PARM {k} {float(p[key]):.12g}")
            L.append(f"  PARM 6 {s.cb_order}")
            L.append(f"  DISZ {t:.12g}")
            continue
        if s.phase == "grating":
            L.append("  TYPE DGRATING")
            L.append(f"  PARM 1 {float(p['freq']) / 1000.0:.12g}")
            L.append(f"  PARM 2 {s.order}")
        elif s.phase:
            raise ZmxError(f"surface {i}: {s.phase} phase has no .zmx equivalent here")
        elif s.shape == "even":
            L.append("  TYPE EVENASPH")
            L.append(f"  PARM 1 {float(p['a2']):.12g}")
            coef = np.asarray(p["coef"], float)
            for k, a in enumerate(coef, start=2):
                L.append(f"  PARM {k} {a / s.r_norm ** (2 * k):.12g}")
        elif s.shape in ("qbfs", "qcon"):
            raise ZmxError(f"surface {i}: Q-type export is not implemented")
        else:
            L.append("  TYPE STANDARD")
        L.append(f"  CURV {float(p['c']):.15g}")
        L.append(f"  CONI {float(p['k']):.12g}")
        if s.interact == "reflect":
            L.append("  GLAS MIRROR 0 0 0 0 0 0 0 0 0 0")
        elif s.interact == "refract" and s.material == "model":
            L.append(f"  GLAS ___BLANK 1 0 {float(p['nd']):.8g} {float(p['vd']):.8g} 0 0 0 0 0 0")
        elif s.interact == "refract" and s.material not in ("air", "", "vacuum"):
            L.append(f"  GLAS {s.material} 0 0 1.5 50 0 0 0 0 0 0")
        L.append(f"  DISZ {t:.12g}" if i < system.n else "  DISZ 0")
    return "\n".join(L) + "\n"


def save(path, system, params, spec, name="NovaOptics design"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_text(system, params, spec, name))
