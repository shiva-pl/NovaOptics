"""Three-element imaging lens, f = 50 mm, F/4, +/-20 deg, visible (F, d, C).

Runs the phase-first pipeline and the conventional baseline on the same merit,
constraints and optimizer, then judges every result on a dense 21-point field
grid that no merit function sampled.

    python examples/design_triplet.py            # full run, ~15-20 min on 4 CPU cores
    python examples/design_triplet.py --quick    # fewer starts / steps, for smoke tests

Outputs go to examples/output/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from novaoptics import analysis as A  # noqa: E402
from novaoptics import continuation as C  # noqa: E402
from novaoptics import plot as P  # noqa: E402
from novaoptics.baseline import conventional  # noqa: E402
from novaoptics.optimize import clear_compiled  # noqa: E402
from novaoptics.system import Spec, describe, to_numpy  # noqa: E402

OUT = Path(__file__).parent / "output"
AGF = Path(__file__).parents[1] / "schott2025.AGF"


def load_catalog(spec):
    """Real glasses: preferred SCHOTT catalog (AR waveguide wafers excluded)."""
    from novaoptics.catalog import build_glass_space, read_agf

    return build_glass_space(read_agf(AGF), spec.wavelengths)


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    spec = Spec()
    glass = load_catalog(spec)
    n_starts = 4 if quick else 12
    rows, designs = [], {}

    def record(name, p, lay, seconds, note):
        rep = A.dense_report(p, lay, spec)
        designs[name] = (p, lay)
        rows.append({"design": name, "mean_wfe": rep["mean_wfe"], "max_wfe": rep["max_wfe"],
                     "mean_spot_um": float(rep["rms_spot_um"].mean()), "efl": rep["efl"],
                     "max_distortion": rep["max_distortion"], "seconds": seconds, "note": note,
                     "prescription": describe(p, lay, spec),
                     "params": {k: v.tolist() for k, v in to_numpy(p).items()}})
        print(f"\n### {name}  (dense grid: mean {rep['mean_wfe']:.3f}, max {rep['max_wfe']:.3f} "
              f"waves, {seconds:.0f} s)\n{describe(p, lay, spec)}\n", flush=True)

    # 1. phase-first, aspheric
    t = time.time()
    # All four architectures are materialized: the aspheric ranking in phase space is
    # the wrong filter for spherical designs (+-+ spherizes well, the -+- family not).
    p, lay, rec, info = C.design_from_phase(spec, aspheres=True,
                                            top_k=1 if quick else len(C.PATTERNS),
                                            glass=glass, snap="gradual", stop_shift=True)
    record("phase-first, aspheric", p, lay, time.time() - t,
           f"architecture ranking in phase space: {info['architectures']}; chosen {info['chosen']}")
    rec_phase = rec

    clear_compiled()
    # 2. phase-first -> all-spherical: fade the aspheres out of every materialized
    #    architecture and keep the best (ranking by materialized, not phase-space, merit)
    t = time.time()
    best = None
    for cand in info["candidates"]:
        rec_s = C.Recorder(spec)
        ps, lays = C.stage_spherize(cand["params"], cand["layout"], spec, rec_s,
                                    steps=5 if quick else 10)
        score = rec_s.snaps[-1].mean_wfe
        print(f"spherized architecture {cand['name']}: {score:.3f} waves", flush=True)
        if best is None or score < best[0]:
            best = (score, ps, lays, cand["name"])
    record("phase-first, spherical (asphere fade)", best[1], best[2], time.time() - t,
           f"asphere fade of each materialized architecture; best was {best[3]}")

    clear_compiled()
    # 3./4. conventional multi-start from Cooke-type starting points
    for asph in (True, False):
        t = time.time()
        pc, layc, cinfo = conventional(spec, n_starts=n_starts, aspheres=asph, free_glass=True,
                                       glass=glass,
                                       verbose=False)
        kind = "aspheric" if asph else "spherical"
        record(f"conventional, {kind}", pc, layc, time.time() - t,
               f"{cinfo['n_good']}/{cinfo['n_starts']} Cooke-type starts converged usefully")

    # figures
    sn = rec_phase.snaps

    def pick(stage, step=None):
        cands = [x for x in sn if x.stage == stage and (step is None or x.step == step)]
        return cands[-1]

    mat = [x for x in sn if x.stage == "materialize"]
    strip = [("0 one ideal screen", sn[0]),
             ("1 distributed screens (phase)", pick("distribute", 1)),
             ("2 color corrected in phase space", pick("color")),
             ("3 morph, 25% of power in glass", mat[len(mat) // 4 - 1]),
             ("3 morph, 60% of power in glass", mat[int(len(mat) * 0.6) - 1]),
             ("4 catalog glass, aspheric", sn[-1])]
    P.film_strip(strip, spec, OUT / "film_strip.png")
    P.trajectory([rec_phase], ["phase-first"], OUT / "trajectory.png")
    P.performance([("phase-first", *designs["phase-first, aspheric"]),
                   ("conventional", *designs["conventional, aspheric"])], spec,
                  OUT / "performance_aspheric.png")
    P.performance([("phase-first", *designs["phase-first, spherical (asphere fade)"]),
                   ("conventional", *designs["conventional, spherical"])], spec,
                  OUT / "performance_spherical.png")

    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| design | mean RMS WFE [waves] | max RMS WFE [waves] | mean RMS spot [um] | "
             "max distortion | time [s] |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['design']} | {r['mean_wfe']:.3f} | {r['max_wfe']:.3f} | "
                     f"{r['mean_spot_um']:.1f} | {100 * r['max_distortion']:.2f}% | "
                     f"{r['seconds']:.0f} |")
    lines.append("")
    lines.append("All figures evaluated on 21 field points (0-20 deg) x 8x16 pupil samples, "
                 "F/d/C, after optimization on 5 field points x 6x12 pupil samples.")
    for r in rows:
        lines += ["", f"## {r['design']}", "", r["note"], "", "```", r["prescription"], "```"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: len(rows) + 2]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
