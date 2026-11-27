"""Qt-free job logic behind the GUI: run options, the pipeline runs, evaluation of the
finished designs, and saving / loading designs as JSON.

Everything here can be used (and is tested) without PySide6.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .. import analysis as A
from .. import continuation as C
from .. import progress
from ..optimize import STATS
from ..parallel import pmap
from ..system import Layout, Spec, describe, layout_dict, layout_from_dict, to_numpy

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AGF = REPO / "schott2025.AGF"


@dataclass
class RunOptions:
    workflow: str = "phase-first"  # or "conventional"
    efl: float = 50.0
    fnum: float = 4.0
    half_field_deg: float = 20.0
    max_distortion: float = 0.02
    max_track: float = 80.0
    aspheres: bool = True
    spherize: bool = True  # phase-first + aspheres: also fade the aspheres out
    catalog: str | None = str(DEFAULT_AGF) if DEFAULT_AGF.exists() else None
    architectures: tuple = tuple(C.PATTERNS)
    n_elem: int = 3  # phase-first: grow the 3-screen architectures to this many elements
    top_k: int = 4
    stop_shift: bool = True
    bending_hops: bool = False
    morph_iters: int = 30
    glass_pull: float = 0.1
    tmin: float = 2.0
    tmax: float = 8.0
    emin: float = 1.0
    gmin: float = 1.0
    n_starts: int = 12
    seed: int = 0

    def spec(self) -> Spec:
        return Spec(efl=self.efl, fnum=self.fnum, half_field_deg=self.half_field_deg,
                    max_distortion=self.max_distortion, max_track=self.max_track)

    def mech(self) -> C.Mech:
        return C.Mech(tmin=self.tmin, emin=self.emin, gmin=self.gmin, tmax=self.tmax)


@dataclass(eq=False)
class Design:
    """A finished design and everything the GUI shows about it."""

    name: str
    params: dict
    layout: Layout
    spec: Spec
    seconds: float = 0.0
    note: str = ""
    catalog: str | None = None
    snaps: list = field(default_factory=list)  # the pipeline path, for scrubbing
    perf: dict = field(default_factory=dict)  # filled by `evaluate`

    @property
    def prescription(self) -> str:
        return describe(self.params, self.layout, self.spec)


# --------------------------------------------------------------------------- catalogs

_CATALOGS: dict = {}


def load_catalog(path, spec: Spec):
    """GlassSpace of the preferred glasses of an AGF catalog (cached); None -> None."""
    if not path:
        return None
    from ..catalog import build_glass_space, read_agf

    key = (str(Path(path).resolve()), tuple(spec.wavelengths))
    if key not in _CATALOGS:
        _CATALOGS[key] = build_glass_space(read_agf(path), spec.wavelengths)
    return _CATALOGS[key]


# --------------------------------------------------------------------------- evaluation

def evaluate(d: Design, n_field=21, n_curve=11) -> Design:
    """Dense-grid image quality, distortion and AD field curves (fills `d.perf`)."""
    spec, p, lay = d.spec, d.params, d.layout
    rep = A.dense_report(p, lay, spec, n_field=n_field)
    fracs = tuple(np.linspace(0, 1, n_field))
    ev = A.evaluate(p, lay, spec, fracs=fracs)
    dist = np.asarray(A.distortion(ev, rep["efl"]))
    cf = tuple(np.linspace(0, 1, n_curve))
    st, ss = A.focus_shifts(p, lay, spec, cf)
    d.perf = {
        "field_deg": rep["fracs"] * spec.half_field_deg,
        "rms_wfe": rep["rms_wfe"], "rms_spot_um": rep["rms_spot_um"],
        "distortion": dist, "curve_deg": np.asarray(cf) * spec.half_field_deg,
        "tangential": np.asarray(st), "sagittal": np.asarray(ss),
        "mean_wfe": rep["mean_wfe"], "max_wfe": rep["max_wfe"],
        "mean_spot_um": float(rep["rms_spot_um"].mean()), "efl": rep["efl"],
        "max_distortion": rep["max_distortion"],
    }
    return d


# --------------------------------------------------------------------------- runs

class _Lines(io.TextIOBase):
    """stdout replacement that forwards complete lines."""

    def __init__(self, fn):
        self.fn, self.buf = fn, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.fn(line)
        return len(s)


def run(opts: RunOptions, on_snap=None, on_design=None, on_log=None):
    """Run one job; returns the finished designs (also passed one by one to on_design).

    on_snap(label, snapshot) is called from worker threads for every pipeline snapshot.
    Raises progress.Cancelled if progress.request_cancel() is called meanwhile.
    """
    log = on_log or (lambda s: None)
    listener = (lambda rec, sn: on_snap(rec.label or "path", sn)) if on_snap else None
    progress.reset_cancel()
    STATS.reset()
    if listener:
        progress.add_listener(listener)
    out = []

    def done(d):
        log(f"evaluating {d.name} on the dense field grid ...")
        evaluate(d)
        out.append(d)
        log(f"### {d.name}: mean {d.perf['mean_wfe']:.3f}, max {d.perf['max_wfe']:.3f} waves "
            f"({d.seconds:.0f} s)\n{d.prescription}\n")
        if on_design:
            on_design(d)

    try:
        with contextlib.redirect_stdout(_Lines(log)):
            spec, mech = opts.spec(), opts.mech()
            glass = load_catalog(opts.catalog, spec)
            if opts.workflow == "phase-first":
                _phase_first(opts, spec, mech, glass, done, log)
            else:
                _conventional(opts, spec, mech, glass, done)
            log(STATS.report())
    finally:
        if listener:
            progress.remove_listener(listener)
    return out


def _phase_first(opts, spec, mech, glass, done, log):
    patterns = {k: v for k, v in C.PATTERNS.items() if k in opts.architectures}
    if not patterns:
        raise ValueError("select at least one architecture")
    t = time.time()
    kind = "aspheric" if opts.aspheres else "spherical"
    p, lay, rec, info = C.design_from_phase(
        spec, mech=mech, patterns=patterns, top_k=max(1, min(opts.top_k, len(patterns))),
        aspheres=opts.aspheres, glass=glass, morph_iters=opts.morph_iters,
        glass_pull=opts.glass_pull, stop_shift=opts.stop_shift,
        bending_hops=opts.bending_hops, verbose=True, n_elem=opts.n_elem)
    ranking = ", ".join(f"{n} {s:.3f}" for n, s in info["architectures"])
    done(Design(f"phase-first, {lay.n_elem} elements, {kind}", p, lay, spec, time.time() - t,
                f"phase-space ranking: {ranking}; chosen {info['chosen']}", opts.catalog,
                list(rec.snaps)))
    if not (opts.aspheres and opts.spherize):
        return
    t = time.time()

    def spherize(cand):
        rec_s = C.Recorder(spec, verbose=False, label=f"spherize {cand['name']}")
        ps, lays = C.stage_spherize(cand["params"], cand["layout"], spec, rec_s, mech=mech)
        return rec_s.snaps[-1].mean_wfe, ps, lays, cand, rec_s

    best = None
    for score, ps, lays, cand, rec_s in pmap(spherize, info["candidates"]):
        log(f"spherized architecture {cand['name']}: {score:.3f} waves")
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, ps, lays, cand, rec_s)
    if best is None:
        log("no architecture survived the asphere fade")
        return
    score, ps, lays, cand, rec_s = best
    snaps = list(cand["rec"].snaps) + list(rec_s.snaps)
    done(Design(f"phase-first, {lays.n_elem} elements, spherical (asphere fade)", ps, lays,
                spec, time.time() - t,
                f"asphere fade of each materialized architecture; best was {cand['name']}",
                opts.catalog, snaps))


def _conventional(opts, spec, mech, glass, done):
    from ..baseline import conventional

    t = time.time()
    p, lay, info = conventional(spec, n_starts=opts.n_starts, aspheres=opts.aspheres,
                                free_glass=True, glass=glass, stop_shift=opts.stop_shift,
                                mech=mech, seed=opts.seed, verbose=True)
    kind = "aspheric" if opts.aspheres else "spherical"
    done(Design(f"conventional, {kind}", p, lay, spec, time.time() - t,
                f"{info['n_good']}/{info['n_starts']} Cooke-type starts converged usefully",
                opts.catalog, list(info["rec"].snaps)))


# --------------------------------------------------------------------------- files

def to_json(d: Design) -> dict:
    lay = d.layout
    perf = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.perf.items()}
    return {
        "design": d.name, "note": d.note, "seconds": d.seconds, "catalog": d.catalog,
        "spec": {k: getattr(d.spec, k) for k in ("efl", "fnum", "half_field_deg",
                                                 "max_distortion", "max_track")},
        "layout": layout_dict(lay),
        "params": {k: v.tolist() for k, v in to_numpy(d.params).items()},
        "prescription": d.prescription, "perf": perf,
    }


def save(designs, path):
    rows = [engine_to_json(d) if isinstance(d, EngineDesign) else to_json(d) for d in designs]
    Path(path).write_text(json.dumps(rows, indent=1))


def from_json(row: dict, default_catalog=None) -> Design:
    """A design from `to_json` output, or from a row of examples/output/results.json."""
    params = {k: jnp.asarray(v) for k, v in row["params"].items()}
    spec = Spec(**row.get("spec", {}))
    catalog = row.get("catalog", default_catalog)
    L = row.get("layout") or {  # results.json written before layouts were saved
        "n_elem": len(row["params"]["t"]), "stop_index": 1,
        "aspheres": "aspheric" in row["design"], "exact_glass": "g" in params}
    glass = None
    if L.get("exact_glass"):
        catalog = catalog or (str(DEFAULT_AGF) if DEFAULT_AGF.exists() else None)
        if catalog is None:
            raise ValueError(f"{row['design']}: needs a glass catalog")
        glass = load_catalog(catalog, spec)
    lay = layout_from_dict(L, glass)
    perf = {k: (np.asarray(v) if isinstance(v, list) else v)
            for k, v in row.get("perf", {}).items()}
    return Design(row["design"], params, lay, spec, row.get("seconds", 0.0), row.get("note", ""),
                  catalog, perf=perf)


def load(path, default_catalog=None):
    rows = json.loads(Path(path).read_text())
    rows = rows if isinstance(rows, list) else [rows]
    return [engine_from_json(r) if r.get("kind") == "engine" else from_json(r, default_catalog)
            for r in rows]



# --------------------------------------------------------------------------- engine designs

@dataclass(eq=False)
class EngineDesign:
    """A design held as a general surface system (imported, edited, cemented, grown or
    multi-configuration).  Same interface as `Design` for the GUI."""

    name: str
    problem: object  # engine.merit.Problem
    params: dict  # {"s": surface dicts, "mc": configuration values}
    spec: object  # engine.system.Spec
    seconds: float = 0.0
    note: str = ""
    catalog: str | None = None
    snaps: list = field(default_factory=list)
    perf: dict = field(default_factory=dict)
    free: dict = None  # variables marked in the lens data editor (None: the default set)

    @property
    def system(self):
        return self.problem.system

    @property
    def prescription(self) -> str:
        return engine_prescription(self.problem, self.params)


def engine_prescription(problem, params) -> str:
    from ..engine import merit as EM

    S = problem.system.surfaces
    lines = [f"{'surf':<5}{'type':<8}{'R [mm]':>12}{'t [mm]':>10}  {'material':<14}{'conic':>9}"
             f"  label"]
    for i, (s, p) in enumerate(zip(S, params["s"])):
        c = float(p["c"])
        r = "inf" if abs(c) < 1e-12 else f"{1 / c:.4f}"
        kind = "CB" if s.shape == "cb" else ("mirror" if s.interact == "reflect" else s.shape)
        mat = s.material if s.interact == "refract" else ("" if s.shape == "cb" else "-")
        if s.material == "model":
            mat = f"{float(p['nd']):.4f}/{float(p['vd']):.1f}"
        extra = []
        if s.stop:
            extra.append("STOP")
        if s.phase:
            extra.append(f"{s.phase} phase")
        if s.shape == "cb":
            extra.append(" ".join(f"{k}={float(p[k]):g}" for k in ("dx", "dy", "tx", "ty", "tz")
                                  if float(p[k]) != 0.0))
        lines.append(f"{i:<5}{kind:<8}{r:>12}{float(p['t']):>10.4f}  {mat:<14}"
                     f"{float(p['k']):>9.4g}  {s.label} {' '.join(extra)}".rstrip())
    if len(problem.configs) > 1 or problem.configs[0].overrides:
        lines += ["", "configurations:"]
        for c, conf in enumerate(problem.configs):
            vals = ", ".join(f"{k} = {float(v):.4f}" for k, v in params["mc"][c].items())
            wl = "" if conf.wavelengths is None else \
                " wavelengths " + "/".join(f"{w * 1e3:.0f}" for w in conf.wavelengths) + " nm"
            efl = "" if conf.efl is None else (f" EFL {conf.efl:g}" if conf.efl else
                                               " (no EFL target)")
            lines.append(f"  {c + 1} {conf.name}: {vals}{wl}{efl}")
    del EM
    return "\n".join(lines)


def evaluate_engine(d: EngineDesign, n_field=21) -> EngineDesign:
    """Dense-field RMS wavefront error, RMS spot and distortion of every configuration."""
    import dataclasses as dc

    from ..engine import design as ED
    from ..engine import merit as EM
    from ..engine.trace import focal_length

    prob, P, spec = d.problem, d.params, d.spec
    rep = ED.dense_report(prob, P, spec, n_field=n_field)
    fields = None
    per = []
    for c, conf in enumerate(prob.configs):
        sp = EM.config_spec(spec, conf)
        fmax = max(sp.fields, key=lambda f: np.hypot(*f))
        r = np.hypot(*fmax)
        u = np.asarray(fmax) / r if r > 0 else np.array([0.0, 1.0])
        fields = tuple(tuple(float(v) for v in u * a) for a in np.linspace(0, r, n_field))
        sp = dc.replace(sp, fields=fields, field_weights=None)
        surfs = EM.config_surfaces(P, prob, c)
        f = EM.compiled(f"evaluate {c}", prob, sp,
                        lambda p, c=c, sp=sp: EM.evaluate_config(
                            EM.config_surfaces(p, prob, c), prob.system, sp, prob.targets))
        ev = {k: np.asarray(v) for k, v in f(P).items()}
        img = ev["img"]  # [F, W, P, 2]
        wt = ev["wt"]
        lw = np.asarray(sp.wl_weights) / np.sum(sp.wl_weights)
        wgt = lw[None, :, None] * wt[None, None, :]
        wgt = wgt / wgt.sum(axis=(1, 2), keepdims=True)
        cen = np.sum(img * wgt[..., None], axis=(1, 2), keepdims=True)
        spot = np.sqrt(np.sum(np.sum((img - cen) ** 2, -1) * wgt, axis=(1, 2))) * 1e3
        try:
            efl = float(focal_length(surfs, prob.system, sp))
        except Exception:  # noqa: BLE001 - e.g. afocal
            efl = float("nan")
        chief = np.linalg.norm(np.asarray(ev["chief"])[:, sp.iref], axis=-1)
        th = np.deg2rad(np.linspace(0, r, n_field))
        ideal = efl * np.tan(th)
        dist = np.where(ideal > 1e-9, (chief - ideal) / np.where(ideal > 1e-9, ideal, 1), 0.0)
        per.append({"name": conf.name or f"config {c + 1}", "field_deg": np.linspace(0, r, n_field),
                    "rms_wfe": rep["per_config"][c], "rms_spot_um": spot, "distortion": dist,
                    "efl": efl})
    nanc = np.full(1, np.nan)
    p0 = per[0]
    d.perf = {"field_deg": p0["field_deg"], "rms_wfe": p0["rms_wfe"],
              "rms_spot_um": p0["rms_spot_um"], "distortion": p0["distortion"],
              "curve_deg": nanc, "tangential": nanc, "sagittal": nanc,
              "mean_wfe": rep["mean_wfe"], "max_wfe": rep["max_wfe"],
              "mean_spot_um": float(np.mean([q["rms_spot_um"].mean() for q in per])),
              "efl": p0["efl"],
              "max_distortion": float(max(np.max(np.abs(q["distortion"])) for q in per)),
              "configs": per}
    return d


def engine_to_json(d: EngineDesign) -> dict:
    import dataclasses as dc

    prob = d.problem
    t = prob.targets
    return {
        "kind": "engine", "design": d.name, "note": d.note, "seconds": d.seconds,
        "catalog": d.catalog,
        "surfaces": [dc.asdict(s) for s in prob.system.surfaces],
        "z_launch": prob.system.z_launch,
        "configs": [dc.asdict(c) for c in prob.configs],
        "targets": {"efl": t.efl, "max_distortion": t.max_distortion,
                    "max_track": t.max_track, "n_rings": t.n_rings, "n_arms": t.n_arms,
                    "region": t.region is not None},
        "spec": dc.asdict(d.spec),
        "params": {"s": [{k: np.asarray(v).tolist() for k, v in p.items()} for p in d.params["s"]],
                   "mc": [{k: float(v) for k, v in m.items()} for m in d.params["mc"]]},
        "prescription": d.prescription,
    }


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def engine_from_json(row: dict) -> EngineDesign:
    from ..engine import merit as EM
    from ..engine.system import Spec as ESpec
    from ..engine.system import Surface, System

    S = tuple(Surface(**s) for s in row["surfaces"])
    system = System(S, row.get("z_launch", -20.0))
    configs = tuple(EM.Config(**{k: _tuples(v) for k, v in c.items()}) for c in row["configs"])
    t = dict(row["targets"])
    region = None
    if t.pop("region", False):
        region = load_catalog(DEFAULT_AGF, ESpec())  # glass map of the default catalog
    spec = ESpec(**{k: _tuples(v) for k, v in row["spec"].items()})
    if region is not None:
        region = load_catalog(DEFAULT_AGF, spec)
    prob = EM.Problem(system, configs, EM.Targets(region=region, **t))
    params = {"s": [{k: jnp.asarray(v, float) for k, v in p.items()} for p in row["params"]["s"]],
              "mc": [{k: jnp.asarray(v, float) for k, v in m.items()} for m in row["params"]["mc"]]}
    return EngineDesign(row["design"], prob, params, spec, row.get("seconds", 0.0),
                        row.get("note", ""), row.get("catalog"))


def engine_from_zmx(path) -> EngineDesign:
    from ..engine import merit as EM
    from ..engine import zmx

    system, params, spec, info = zmx.load(path)
    prob = EM.Problem(system, targets=EM.Targets())
    name = info.get("name") or Path(path).stem
    return EngineDesign(name, prob, EM.make_params(params, prob), spec,
                        note=f"imported from {Path(path).name}")


def engine_from_design(d: Design) -> EngineDesign:
    """An element-model design as an engine design (to edit, grow, cement, configure)."""
    from ..engine import design as ED

    region = load_catalog(d.catalog, d.spec) if d.catalog else None
    prob, P, spec = ED.from_element(d.params, d.layout, d.spec, region=region)
    return EngineDesign(d.name + " (surfaces)", prob, P, spec, note=d.note, catalog=d.catalog)
