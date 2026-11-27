"""Multi-configuration designs on the surface engine.

1. Zoom: three singlets (+ - +) in model glasses; the two spacings are variables of
   each configuration, the focal-length targets differ (35 / 50 / 70 mm); everything
   else is shared.
2. Spectrum from defocus: a doublet with *intentional* axial color.  Each wavelength
   is its own configuration, evaluated alone in its own image plane; the image planes
   are prescribed (linear in wavelength, 0.4 mm per 100 nm) and the lens is optimized
   to focus every wavelength on its plane.

    python examples/multiconfig.py      # ~10 min on 4 CPU cores

Writes examples/output/multiconfig/: results.md, zoom.png, spectral.png.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from novaoptics.engine import Builder, Spec  # noqa: E402
from novaoptics.engine import design as D  # noqa: E402
from novaoptics.engine import merit as EM  # noqa: E402
from novaoptics.engine.plots import draw_system  # noqa: E402
from novaoptics.engine.trace import focal_length  # noqa: E402
from novaoptics.gui.runner import engine_prescription  # noqa: E402
from novaoptics.plot import SURFACE  # noqa: E402

OUT = Path(__file__).parent / "output" / "multiconfig"
D_LINE = 0.5875618


def zoom():
    b = Builder()
    b.add(c=1 / 60, t=4.0, material="model", nd=1.62, vd=60.0, label="G1")
    b.add(c=-1 / 120, t=10.0)
    b.add(c=-1 / 25, t=2.0, material="model", nd=1.70, vd=35.0, label="G2")
    b.add(c=1 / 60, t=10.0)
    b.stop(t=1.0)
    b.add(c=1 / 30, t=4.0, material="model", nd=1.62, vd=60.0, label="G3")
    b.add(c=-1 / 60, t=40.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=8.0, fields=((0.0, 0.0), (0.0, 4.0), (0.0, 6.0)))
    over = ((1, "t"), (3, "t"))
    configs = (EM.Config("wide", over, efl=35.0), EM.Config("mid", over, efl=50.0),
               EM.Config("tele", over, efl=70.0))
    prob = EM.Problem(system, configs, EM.Targets(max_track=120.0))
    P = EM.make_params(prm, prob, {(0, 1, "t"): 18.0, (0, 3, "t"): 12.0,
                                   (1, 1, "t"): 12.0, (1, 3, "t"): 8.0,
                                   (2, 1, "t"): 6.0, (2, 3, "t"): 4.0})
    # first the spacings alone (a paraxial start), then everything
    P, _ = D.optimize_design(prob, P, spec, D.variables(prob, curvatures=False,
                                                        thicknesses=False, glasses=False),
                             iters=100)
    P, _ = D.optimize_design(prob, P, spec, iters=300)
    prob, P = D.snap_glasses(prob, P, spec)
    return "zoom 35-70 mm", prob, P, spec


def spectral():
    wls = (0.45, 0.55, 0.65, 0.75, 0.85)
    b = Builder()
    b.stop(t=0.0)
    b.add(c=1 / 30, t=5.0, material="model", nd=1.70, vd=30.0, label="L1")
    b.add(c=-1 / 40, t=2.0, material="model", nd=1.52, vd=64.0, label="L2")
    b.add(c=-1 / 200, t=48.0)
    b.image()
    system, prm = b.build()
    spec = Spec(epd=6.0, fields=((0.0, 0.0), (0.0, 2.0)), wavelengths=wls,
                wl_weights=(1.0,) * len(wls), ref=2)
    prob = EM.Problem(system, targets=EM.Targets(efl=50.0))
    P = EM.make_params(prm, prob)
    prob, P = D.per_wavelength_focus(prob, P, spec)
    # prescribed image planes: 0.4 mm per 100 nm, centered on 650 nm
    for c, wl in enumerate(wls):
        P["mc"][c]["3:t"] = 48.0 + 4.0 * (wl - 0.65)
    free = D.variables(prob, configs=False)  # lens free, image planes fixed
    free["s"][3] = {"c": True}  # the space to the image is set per configuration
    P, _ = D.optimize_design(prob, P, spec, free, iters=300)
    prob, P = D.snap_glasses(prob, P, spec)
    return "spectrum from defocus", prob, P, spec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for job in (zoom, spectral):
        t = time.time()
        name, prob, P, spec = job()
        rms = EM.rms_wfe(P, prob, spec)
        lines += [f"## {name}", "", f"{time.time() - t:.0f} s", "",
                  "| configuration | EFL [mm] | RMS WFE per field [waves] | image distance [mm] |",
                  "|---|---|---|---|"]
        for c, conf in enumerate(prob.configs):
            sp = EM.config_spec(spec, conf)
            surfs = EM.config_surfaces(P, prob, c)
            efl = float(focal_length(surfs, prob.system, sp))
            img = float(surfs[len(surfs) - 2]["t"])
            lines.append(f"| {conf.name} | {efl:.2f} | "
                         f"{' '.join(f'{v:.3f}' for v in rms[c])} | {img:.3f} |")
        lines += ["", "```", engine_prescription(prob, P), "```", ""]
        n = len(prob.configs)
        fig, axes = plt.subplots(n, 1, figsize=(7, 2.1 * n), facecolor=SURFACE)
        for c, ax in enumerate(np.atleast_1d(axes)):
            draw_system(ax, P, prob.system, spec, problem=prob, config=c,
                        title=f"{name}: {prob.configs[c].name}")
        fig.tight_layout()
        fig.savefig(OUT / ("zoom.png" if job is zoom else "spectral.png"), dpi=140,
                    facecolor=SURFACE)
        plt.close(fig)
        print("\n".join(lines[-(n + 8):]), flush=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
