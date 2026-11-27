"""Continue a finished design: one more element, or a cemented doublet.

Starts from the 3-element all-spherical phase-first design of design_triplet.py
(examples/output/results.json) and, on the surface engine,

* adds an air-spaced element: a zero-power phase screen is inserted into every air
  space (and in front), each grows into a glass singlet, the best is kept;
* cements a new glass onto each element in turn (a zero-thickness layer grown into a
  cemented doublet), the best is kept.

    python examples/continue_design.py     # ~15 min on 4 CPU cores

Writes examples/output/continue/: results.md, layouts.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from novaoptics.engine import design as D  # noqa: E402
from novaoptics.engine.plots import draw_system  # noqa: E402
from novaoptics.gui import runner as R  # noqa: E402
from novaoptics.plot import SURFACE  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output" / "continue"


def prescription(problem, params):
    lines = [f"{'surf':<5}{'R [mm]':>11}{'t [mm]':>9}  {'material':<12}{'label'}"]
    for i, (s, p) in enumerate(zip(problem.system.surfaces, params["s"])):
        c = float(p["c"])
        r = "inf" if abs(c) < 1e-12 else f"{1 / c:.3f}"
        lines.append(f"{i:<5}{r:>11}{float(p['t']):>9.3f}  {s.material:<12}{s.label}"
                     + ("  (stop)" if s.stop else ""))
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    row = json.loads((HERE / "output" / "results.json").read_text())[1]
    d = R.from_json(row)
    region = R.load_catalog(R.DEFAULT_AGF, d.spec)
    prob, P, spec = D.from_element(d.params, d.layout, d.spec, region=region)
    designs = [("3 elements (start)", prob, P, 0.0)]

    t = time.time()
    p4, P4, res = D.add_element(prob, P, spec, log=print)
    designs.append(("+1 air-spaced element", p4, P4, time.time() - t))

    t = time.time()
    rears = [i for i, s in enumerate(prob.system.surfaces)
             if s.interact == "refract" and prob.system.surfaces[i - 1].material
             not in D.AIR and s.material in D.AIR]

    def cem(rear):
        q, Q = D.cement(prob, P, spec, rear, log=print)
        return D.cost(q, Q, spec), rear, q, Q

    tried = [cem(r) for r in rears]  # each tries 8 glasses concurrently
    for c, rear, q, Q in tried:
        print(f"   cement behind surface {rear}: mean RMS WFE {D.mean_wfe(q, Q, spec):.3f}")
    c, rear, qc, Qc = min(tried, key=lambda x: x[0])
    designs.append((f"cemented (behind surface {rear})", qc, Qc, time.time() - t))

    lines = ["| design | mean RMS WFE [waves] | max RMS WFE [waves] | time [s] |",
             "|---|---|---|---|"]
    for name, q, Q, sec in designs:
        rep = D.dense_report(q, Q, spec)
        lines.append(f"| {name} | {rep['mean_wfe']:.3f} | {rep['max_wfe']:.3f} | {sec:.0f} |")
    lines += ["", "f = 50 mm, F/4, +/-20 deg, F/d/C; dense 21-field grid; preferred SCHOTT "
              "glasses."]
    for name, q, Q, _ in designs:
        lines += ["", f"## {name}", "", "```", prescription(q, Q), "```"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))

    fig, axes = plt.subplots(1, len(designs), figsize=(5 * len(designs), 3.2),
                             facecolor=SURFACE)
    for ax, (name, q, Q, _) in zip(axes, designs):
        rep = D.dense_report(q, Q, spec)
        draw_system(ax, Q, q.system, spec, problem=q,
                    title=f"{name}   {rep['mean_wfe']:.3f} / {rep['max_wfe']:.3f} waves")
    fig.tight_layout()
    fig.savefig(OUT / "layouts.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
