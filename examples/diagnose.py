"""Diagnostics for the designs in examples/output/results.json and for the morph.

    python examples/diagnose.py            # ~15 min on 4 cores (re-runs the pipeline)
    python examples/diagnose.py --designs  # final-design diagnostics only, ~1 min

Writes examples/output/diagnostics/: report.md, seidel_elements.png,
morph_seidel.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from novaoptics import continuation as C  # noqa: E402
from novaoptics import diagnostics as D  # noqa: E402
from novaoptics.plot import GRID, INK, MUTED, SERIES, SURFACE, _style  # noqa: E402
from novaoptics.system import Layout, Spec  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output" / "diagnostics"
SHOW = ("W040", "W131", "W222", "W220P", "W020", "W111")  # terms plotted per element
SHOW_LABEL = {"W040": "spherical W040", "W131": "coma W131", "W222": "astigmatism W222",
              "W220P": "Petzval W220P", "W020": "axial color W020 (F-C)",
              "W111": "lateral color W111 (F-C)"}


def load_designs(glass):
    rows = json.loads((HERE / "output" / "results.json").read_text())
    out = {}
    for r in rows:
        p = {k: jnp.asarray(v) for k, v in r["params"].items()}
        lay = Layout(n_elem=3, stop_index=1, aspheres="aspheric" in r["design"],
                     glass=glass if "g" in p else None)
        out[r["design"]] = (p, lay)
    return out


def design_report(designs, spec):
    lines = ["# Design diagnostics", "",
             "Seidel and primary-color coefficients in d-line waves at full field and "
             "aperture (W220 = sagittal field curvature, W220P = its Petzval part). "
             "`sum|.|` adds the magnitudes of all surface contributions: how much "
             "aberration the design cancels.", ""]
    summary = []
    for name, (p, lay) in designs.items():
        sd = D.seidel(p, lay, spec)
        ac = D.aberration_content(sd, spec)
        fit = D.aberration_fit(p, lay, spec)
        c = dict(zip(fit["names"], fit["coef"][1]))
        summary.append((name, ac, c, sd))
        lines += [f"## {name}", "", "```", D.seidel_table(p, lay, spec), "```", "",
                  "Real-ray Hopkins fit through fifth order (d-line waves):", "", "```",
                  D.fit_table(p, lay, spec), "```", ""]
    # compact comparison
    head = ("| design | sum abs W040 | sum abs W131 | sum abs W222 | sum abs W220P | "
            "W040 / W060 fit (d) | exact axial color | exact lateral color |")
    lines[4:4] = [head, "|---|---|---|---|---|---|---|---|"] + [
        f"| {n} | {ac['abs_sum'][0]:.1f} | {ac['abs_sum'][1]:.1f} | {ac['abs_sum'][2]:.1f} | "
        f"{ac['abs_sum'][7]:.1f} | {c['W040']:+.2f} / {c['W060']:+.2f} | "
        f"{1e3 * sd['axial_color_mm']:+.0f} um | {1e3 * sd['lateral_color_mm']:+.1f} um |"
        for n, ac, c, sd in summary] + [""]
    return "\n".join(lines)


def plot_elements(designs, names, spec, path):
    """Element-by-element Seidel contributions of two designs, one panel per term."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), facecolor=SURFACE)
    idx = {n: k for k, n in enumerate(D.W_NAMES)}
    data = {}
    for n in names:
        p, lay = designs[n]
        sd = D.seidel(p, lay, spec)
        groups = D.element_contributions(sd)
        lam = spec.ref_wavelength * 1e-3
        data[n] = {g: D._to_w(np.asarray(v), lam) for g, v in groups.items()}
    elems = sorted(next(iter(data.values())))
    x = np.arange(len(elems) + 1)
    w = 0.36
    for ax, term in zip(axes.ravel(), SHOW):
        _style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
        for j, n in enumerate(names):
            vals = [data[n][e][idx[term]] for e in elems]
            vals.append(sum(vals))
            ax.bar(x + (j - 0.5) * w, vals, width=w - 0.04, color=SERIES[j],
                   label=n if term == SHOW[0] else None)
        ax.set_xticks(x, elems + ["total"], fontsize=8)
        ax.set_title(SHOW_LABEL[term], fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("waves", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("waves", fontsize=8, color=INK)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def morph_diagnostics(spec, glass, gaps_at=(0.25, 0.5, 0.75, 1.0)):
    p, lay, rec, info = C.design_from_phase(spec, aspheres=True, verbose=False, glass=glass,
                                            snap="gradual", stop_shift=True)
    cand = next(c for c in info["candidates"] if c["name"] == info["chosen"])
    rec = cand["rec"]
    rec_s = C.Recorder(spec, verbose=False)
    C.stage_spherize(cand["params"], cand["layout"], spec, rec_s)
    snaps = rec.snaps + rec_s.snaps
    traj = []
    for sn in snaps:
        sd = D.seidel(sn.params, sn.layout, spec)
        ac = D.aberration_content(sd, spec)
        traj.append({"stage": sn.stage, "mean_wfe": sn.mean_wfe, "total": ac["total"],
                     "abs_sum": ac["abs_sum"]})
    gaps = []
    mats = [sn for sn in rec.snaps if sn.stage == "materialize" and "z" in sn.extra]
    for s_target in gaps_at:
        sn = min(mats, key=lambda x: abs(x.extra["cfg"]["s"] - s_target))
        g = D.path_gap(sn, spec)
        g["index"] = snaps.index(sn)
        gaps.append(g)
        print(f"path gap at s={g['s']:.2f}: mean WFE recorded {g['recorded'][1]:.3f}, "
              f"converged {g['converged'][1]:.3f}, free curvature {g['free_curvature'][1]:.3f} "
              f"waves | cost {g['recorded'][0]:.3g} -> {g['converged'][0]:.3g} -> "
              f"{g['free_curvature'][0]:.3g}", flush=True)
    return traj, gaps, info


def plot_morph(traj, gaps, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
    xs = np.arange(len(traj))
    a1.plot(xs, [t["mean_wfe"] for t in traj], color=SERIES[0], lw=2, marker="o", ms=3,
            label="recorded along the path")
    a1.scatter([g["index"] for g in gaps], [g["converged"][1] for g in gaps], color=SERIES[1],
               s=36, zorder=3, label="same parameterization, converged")
    a1.scatter([g["index"] for g in gaps], [g["free_curvature"][1] for g in gaps],
               color=SERIES[2], marker="D", s=30, zorder=3, label="+ free curvature corrections")
    a1.set_yscale("log")
    a1.set_ylabel("mean RMS WFE [waves]", fontsize=8, color=INK)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK)
    k = {n: i for i, n in enumerate(D.W_NAMES)}
    for j, term in enumerate(("W040", "W131", "W222", "W220P")):
        a2.plot(xs, [t["abs_sum"][k[term]] for t in traj], color=SERIES[j], lw=2,
                label=f"sum |{term}| over surfaces")
    a2.set_yscale("log")
    a2.set_ylabel("aberration content [waves]", fontsize=8, color=INK)
    a2.set_xlabel("pipeline step", fontsize=8, color=INK)
    a2.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)
    stages = [t["stage"] for t in traj]
    start = 0
    for i in range(1, len(stages) + 1):
        if i == len(stages) or stages[i] != stages[start]:
            for ax in (a1, a2):
                ax.axvline(start - 0.5, color=MUTED, lw=0.6, ls=":")
            nth = len(set(stages[:start]))
            a1.text((start + i - 1) / 2, 1.02 + 0.06 * (nth % 2), stages[start],
                    transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
            start = i
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(designs_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    spec = Spec()
    from design_triplet import load_catalog

    glass = load_catalog(spec)
    designs = load_designs(glass)
    report = design_report(designs, spec)
    plot_elements(designs, ["phase-first, spherical (asphere fade)", "conventional, spherical"],
                  spec, OUT / "seidel_elements.png")
    if not designs_only:
        traj, gaps, info = morph_diagnostics(spec, glass)
        (OUT / "morph.json").write_text(json.dumps(
            {"traj": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()}
                      for t in traj], "gaps": gaps}, indent=1))
        plot_morph(traj, gaps, OUT / "morph_seidel.png")
        report += "\n## Morph path gaps\n\n| s | recorded | converged | + free curvature |\n" \
                  "|---|---|---|---|\n" + "\n".join(
                      f"| {g['s']:.2f} | {g['recorded'][1]:.3f} | {g['converged'][1]:.3f} | "
                      f"{g['free_curvature'][1]:.3f} |" for g in gaps) + \
                  "\n\nMean RMS WFE in waves on the design image surface (with virtual " \
                  "Petzval).\n"
    (OUT / "report.md").write_text(report)
    print(report[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", action="store_true")
    main(ap.parse_args().designs)
