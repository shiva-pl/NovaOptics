"""GUI back end (Qt-free runner), cancellation, and an offscreen smoke test of the window."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from novaoptics import continuation as C
from novaoptics import progress
from novaoptics.gui import runner as R
from novaoptics.system import Spec

RESULTS = Path(__file__).parents[1] / "examples" / "output" / "results.json"
needs_results = pytest.mark.skipif(not (RESULTS.exists() and R.DEFAULT_AGF.exists()),
                                   reason="example results / catalog not present")


@needs_results
def test_saved_design_reproduces_its_numbers(tmp_path):
    row = json.loads(RESULTS.read_text())[2]  # conventional, aspheric: stop moved to front
    d = R.evaluate(R.from_json(row))
    assert d.layout.stop_index == row["layout"]["stop_index"]
    assert d.perf["mean_wfe"] == pytest.approx(row["mean_wfe"], rel=1e-6)
    R.save([d], tmp_path / "d.json")
    (e,) = R.load(tmp_path / "d.json")
    assert e.layout == d.layout
    for k in d.params:
        np.testing.assert_allclose(np.asarray(e.params[k]), np.asarray(d.params[k]))
    assert e.perf["mean_wfe"] == pytest.approx(d.perf["mean_wfe"])


def test_cancel_stops_the_pipeline():
    seen = []

    def listener(rec, snap):
        seen.append(snap.stage)
        progress.request_cancel()

    progress.reset_cancel()
    progress.add_listener(listener)
    try:
        with pytest.raises(progress.Cancelled):
            C.design_from_phase(Spec(), patterns={"+-+": C.PATTERNS["+-+"]}, top_k=1,
                                verbose=False)
    finally:
        progress.remove_listener(listener)
        progress.reset_cancel()
    # stops at the next optimizer call: at most the un-optimized split is recorded
    assert seen[0] == "ideal" and set(seen) <= {"ideal", "distribute"} and len(seen) <= 2


@needs_results
def test_window_smoke():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from novaoptics import plot as P
    from novaoptics.gui.app import MainWindow, View

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    d = R.evaluate(R.from_json(json.loads(RESULTS.read_text())[0]))
    w.add_design(d)
    assert w.table.rowCount() == 1 and w.selected() == [d]
    assert "N-PK51" in w.rx.toPlainText()
    w.tabs.setCurrentWidget(w.merit_plot)  # trajectory of a path-less design: a message
    v = View(d.params, d.layout, d.spec, 1.0, "test")
    w._on_geometry(v, P.layout_geometry(d.params, d.layout, d.spec))
    assert w.layout_plot.figure.axes
    from novaoptics.gui.app import _analysis

    res = _analysis((d, 4))  # what the analysis tab computes, here synchronously
    assert 0 < res["psf"]["strehl"] < 1 and res["spot"]["rms_um"] > 0
    w._analysis_shown = (d, 4)
    w._on_analysis((d, 4), res)
    assert w.analysis_status.text().startswith("Strehl")
    opts = w.options()
    assert opts.workflow == "phase-first" and opts.spec().efl == 50.0
    w.close()
    app.processEvents()
