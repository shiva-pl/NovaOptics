"""Multi-element lenses grown from phase: f = 50 mm, +/-20 deg, visible, all-spherical.

For each aperture (F/4, F/2.8) the phase-first pipeline is run with the final element
count N = 3..6: the four 3-screen architectures are each grown to N screens by
splitting (grow.py), morphed into glass, stop-shifted and snapped to preferred SCHOTT
glasses; the best architecture is kept.

    python examples/design_multilens.py          # ~1 h on 4 CPU cores
    python examples/design_multilens.py --quick  # one architecture, N = 3, 5

Writes examples/output/multilens/: results.md, results.json, layouts.png, performance.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from novaoptics import analysis as A  # noqa: E402
from novaoptics import continuation as C  # noqa: E402
from novaoptics import plot as P  # noqa: E402
from novaoptics.gui.runner import load_catalog  # noqa: E402
from novaoptics.optimize import clear_compiled  # noqa: E402
from novaoptics.system import Spec, describe, layout_dict, to_numpy  # noqa: E402

OUT = Path(__file__).parent / "output" / "multilens"
AGF = Path(__file__).parents[1] / "schott2025.AGF"


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, designs = [], {}
    patterns = {"+-+": C.PATTERNS["+-+"]} if quick else C.PATTERNS
    for fnum in (4.0, 2.8):
        spec = Spec(fnum=fnum)
        glass = load_catalog(AGF, spec)
        for n in ((3, 5) if quick else (3, 4, 5, 6)):
            t = time.time()
            p, lay, rec, info = C.design_from_phase(spec, patterns=patterns,
                                                    top_k=len(patterns), glass=glass,
                                                    stop_shift=True, n_elem=n, verbose=False)
            rep = A.dense_report(p, lay, spec)
            name = f"F/{fnum:g}, {n} elements"
            designs[name] = (p, lay, spec)
            rows.append({"design": name, "fnum": fnum, "n_elem": n,
                         "mean_wfe": rep["mean_wfe"], "max_wfe": rep["max_wfe"],
                         "mean_spot_um": float(rep["rms_spot_um"].mean()),
                         "max_distortion": rep["max_distortion"], "efl": rep["efl"],
                         "seconds": time.time() - t, "chosen": info["chosen"],
                         "finals": info["finals"], "prescription": describe(p, lay, spec),
                         "layout": layout_dict(lay), "catalog": str(AGF),
                         "spec": {"efl": spec.efl, "fnum": spec.fnum,
                                  "half_field_deg": spec.half_field_deg,
                                  "max_distortion": spec.max_distortion,
                                  "max_track": spec.max_track},
                         "params": {k: v.tolist() for k, v in to_numpy(p).items()}})
            clear_compiled()  # programs for other element counts are not reused
            print(f"{name}: mean {rep['mean_wfe']:.3f} max {rep['max_wfe']:.3f} waves "
                  f"(from {info['chosen']}; {time.time() - t:.0f} s)", flush=True)
            (OUT / "results.json").write_text(json.dumps(rows, indent=1))

    lines = ["| design | best architecture | mean RMS WFE [waves] | max RMS WFE [waves] | "
             "mean RMS spot [um] | max distortion | time [s] |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['design']} | {r['chosen']} | {r['mean_wfe']:.3f} | "
                     f"{r['max_wfe']:.3f} | {r['mean_spot_um']:.1f} | "
                     f"{100 * r['max_distortion']:.2f}% | {r['seconds']:.0f} |")
    lines += ["", "f = 50 mm, +/-20 deg, F/d/C, all-spherical, preferred SCHOTT glasses; "
              "dense 21-field grid."]
    for r in rows:
        lines += ["", f"## {r['design']}", "", "```", r["prescription"], "```"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")

    # layouts: one row per aperture
    fn = sorted({r["fnum"] for r in rows}, reverse=True)
    ns = sorted({r["n_elem"] for r in rows})
    fig, axes = plt.subplots(len(fn), len(ns), figsize=(4.2 * len(ns), 2.9 * len(fn)),
                             facecolor=P.SURFACE, squeeze=False)
    for a, f in enumerate(fn):
        for b, n in enumerate(ns):
            name = f"F/{f:g}, {n} elements"
            r = next(x for x in rows if x["design"] == name)
            p, lay, spec = designs[name]
            P.draw_layout(axes[a, b], p, lay, spec,
                          title=f"{name}   {r['mean_wfe']:.3f} / {r['max_wfe']:.3f} waves")
    fig.tight_layout()
    fig.savefig(OUT / "layouts.png", dpi=150, facecolor=P.SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=P.SURFACE)
    P._style(ax)
    for j, f in enumerate(fn):
        rr = sorted([r for r in rows if r["fnum"] == f], key=lambda r: r["n_elem"])
        ax.plot([r["n_elem"] for r in rr], [r["mean_wfe"] for r in rr], color=P.SERIES[j],
                lw=2, marker="o", label=f"F/{f:g} mean")
        ax.plot([r["n_elem"] for r in rr], [r["max_wfe"] for r in rr], color=P.SERIES[j],
                lw=1.2, ls="--", marker="o", ms=3, label=f"F/{f:g} worst field")
    ax.axhline(0.07, color=P.MUTED, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("elements", fontsize=8, color=P.INK)
    ax.set_ylabel("RMS wavefront error [waves]", fontsize=8, color=P.INK)
    ax.grid(color=P.GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=P.INK)
    fig.tight_layout()
    fig.savefig(OUT / "performance.png", dpi=160, facecolor=P.SURFACE)
    plt.close(fig)
    print("\n".join(lines[: len(rows) + 2]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
