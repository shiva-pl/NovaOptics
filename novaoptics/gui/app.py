"""NovaOptics desktop GUI (PySide6).

    python -m novaoptics.gui [designs.json ...]

Left: specification and pipeline options.  Center: the system layout with a slider
over every recorded pipeline step (live while a run is going: watch phase screens
turn into glass), the merit trajectory, image-quality plots, the prescription and
the log.  Bottom: the finished designs; select one or several to compare them.

The pipeline runs in a background thread (its optimizations in a thread pool, see
parallel.py); ray geometry for drawing is computed in a second background thread, so
the window stays responsive.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QObject, Qt, QTimer, Signal

import matplotlib  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import plot as P  # noqa: E402
from .. import progress  # noqa: E402
from ..continuation import PATTERNS  # noqa: E402
from . import runner as R  # noqa: E402
from .lensdata import LensDataEditor  # noqa: E402

ARCH_NOTES = {"+-+": "Cooke-like", "-++": "negative front", "++-": "telephoto-like",
              "+++": "all positive"}
LATEST = "live: every step, in arrival order"


@dataclass(eq=False)
class View:
    """One system to draw."""

    params: dict
    layout: object
    spec: object
    beta: float
    title: str
    engine: object = None  # an EngineDesign: drawn as a surface system
    config: int = 0


def _view(snap, spec, label):
    note = f" · {snap.note}" if snap.note else ""
    return View(snap.params, snap.layout, spec, snap.beta,
                f"{label} · {snap.stage} {snap.step} · mean RMS WFE {snap.mean_wfe:.3f} waves"
                f"{note}")


class Bridge(QObject):
    """Signals from worker threads to the GUI thread (queued connections)."""

    snap = Signal(str, object)
    design = Signal(object)
    log = Signal(str)
    refresh = Signal(object)  # a design whose evaluation changed
    finished = Signal(object)  # None, "cancelled" or a traceback string


class Renderer(QObject):
    """Runs `fn(request)` for the most recently requested item, off the GUI thread.

    Requests that arrive while it is busy replace each other: only the newest is
    computed.  Used for layout ray geometry and for the image analyses.
    """

    ready = Signal(object, object)

    def __init__(self, fn, name="novaoptics-render"):
        super().__init__()
        self._fn = fn
        self._cv = threading.Condition()
        self._pending = None
        threading.Thread(target=self._loop, daemon=True, name=name).start()

    def request(self, item):
        with self._cv:
            self._pending = item
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                item, self._pending = self._pending, None
            try:
                out = self._fn(item)
            except Exception as e:  # noqa: BLE001 - shown in the plot
                out = e
            self.ready.emit(item, out)


def _geometry(view):
    if view.engine is not None:
        from ..engine.plots import system_geometry

        system, surfs, spec = _engine_model(view.engine, view.config)
        return system_geometry(surfs, system, spec)
    return P.layout_geometry(view.params, view.layout, view.spec, view.beta)


def _engine_model(d, config=0):
    """(system, surface params, spec) of a design as a surface-engine system (element
    designs are converted once and cached on the design)."""
    if isinstance(d, R.EngineDesign):
        from ..engine import merit as EM

        c = min(config, len(d.problem.configs) - 1)
        return (d.problem.system, EM.config_surfaces(d.params, d.problem, c),
                EM.config_spec(d.spec, d.problem.configs[c]))
    if getattr(d, "_engine", None) is None:
        from ..engine import convert

        sp = d.spec
        d._engine = convert.from_layout(d.params, d.layout, sp.epd,
                                        [f * sp.half_field_deg for f in sp.field_fracs],
                                        sp.wavelengths)
    return d._engine


def _analysis(req):
    from ..engine import analysis as EA

    d, k, c = req
    system, params, spec = _engine_model(d, c)
    return EA.analyze(params, system, spec, spec.fields[min(k, len(spec.fields) - 1)])


class PlotPanel(QtWidgets.QWidget):
    """A matplotlib figure with the navigation toolbar."""

    def __init__(self, figsize=(8, 4), parent=None):
        super().__init__(parent)
        # explicit dpi: a site matplotlibrc may set a print dpi that inflates screen text
        self.figure = Figure(figsize=figsize, dpi=100, facecolor=P.SURFACE, layout="tight")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def message(self, text):
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=P.MUTED, fontsize=10)
        self.canvas.draw_idle()


def _spin(lo, hi, val, step=1.0, dec=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def _ispin(lo, hi, val):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NovaOptics — phase-first lens design")
        self.resize(1400, 900)
        self.designs: list[R.Design] = []
        self.paths: dict[str, list] = {}  # layout-tab sources: name -> snapshots
        self.path_spec: dict = {}
        self.live_names: list[str] = []
        self.snap_label: dict = {}  # id(snapshot) -> architecture label (live runs)
        self.running = False
        self.t_start = 0.0
        self._dirty = False
        self.bridge = Bridge()
        self.bridge.snap.connect(self._on_snap)
        self.bridge.design.connect(self._on_design)
        self.bridge.log.connect(self._on_log)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.refresh.connect(self._refresh_design)
        self.renderer = Renderer(_geometry)
        self.renderer.ready.connect(self._on_geometry)
        self.analyzer = Renderer(_analysis, "novaoptics-analysis")
        self.analyzer.ready.connect(self._on_analysis)
        self._analysis_shown = None

        self._build_menu()
        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.addWidget(self._build_options())
        right = QtWidgets.QSplitter(Qt.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_table())
        right.setSizes([650, 220])
        split.addWidget(right)
        split.setSizes([340, 1060])
        self.setCentralWidget(split)
        self._build_status()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(250)
        self._update_enabled()
        self.layout_plot.message("Run a design (Ctrl+R) or open saved designs (Ctrl+O).")
        self.merit_plot.message("The merit along the pipeline appears here.")
        self.perf_plot.message("Image quality of the selected designs appears here.")

    # ------------------------------------------------------------------ construction
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Open designs…", self.open_designs)
        a.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open))
        a = m.addAction("&Save designs…", self.save_designs)
        a.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save))
        m.addAction("&Export current figure…", self.export_figure)
        m.addSeparator()
        m.addAction("&Import .zmx…", self.import_zmx)
        m.addAction("E&xport .zmx…", self.export_zmx)
        m.addSeparator()
        a = m.addAction("&Quit", self.close)
        a.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Quit))
        m = self.menuBar().addMenu("&Run")
        self.act_run = m.addAction("&Run", self.start_run)
        self.act_run.setShortcut("Ctrl+R")
        self.act_stop = m.addAction("&Stop", self.stop_run)
        self.act_stop.setShortcut("Ctrl+.")
        m = self.menuBar().addMenu("&Design")
        m.addAction("Edit as &surfaces (copy)", lambda: self.edit_as_surfaces())
        m.addSeparator()
        m.addAction("&Optimize", self.optimize_selected)
        m.addAction("&Add an element", self.add_element_selected)
        m.addAction("&Cement a glass to an element…", self.cement_selected)
        m.addAction("&Materialize a phase screen…", self.materialize_selected)
        m.addAction("&Dematerialize a lens into a phase screen…", self.dematerialize_selected)
        m.addAction("&Snap model glasses to the catalog", self.snap_selected)
        m.addAction("&Per-wavelength focus (one image plane per wavelength)",
                    self.per_wavelength_selected)
        m = self.menuBar().addMenu("&Help")
        m.addAction("&About", self.about)

    def _build_options(self):
        o = R.RunOptions()
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        g = QtWidgets.QGroupBox("Specification")
        f = QtWidgets.QFormLayout(g)
        self.efl = _spin(1, 10000, o.efl, 1, 2, " mm")
        self.fnum = _spin(0.5, 64, o.fnum, 0.1, 2)
        self.hfov = _spin(0.1, 80, o.half_field_deg, 1, 1, "°")
        self.maxdist = _spin(0, 50, 100 * o.max_distortion, 0.5, 1, " %")
        self.track = _spin(1, 10000, o.max_track, 1, 1, " mm")
        f.addRow("Focal length", self.efl)
        f.addRow("F-number", self.fnum)
        f.addRow("Half field", self.hfov)
        f.addRow("Max distortion", self.maxdist)
        f.addRow("Max track", self.track)
        wl = QtWidgets.QLabel("F, d, C (486.1, 587.6, 656.3 nm), object at infinity")
        wl.setWordWrap(True)
        f.addRow("Spectrum", wl)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("Workflow")
        f = QtWidgets.QVBoxLayout(g)
        self.wf_phase = QtWidgets.QRadioButton("Phase-first (no starting design)")
        self.wf_conv = QtWidgets.QRadioButton("Conventional (Cooke-type multi-start)")
        self.wf_phase.setChecked(True)
        self.aspheres = QtWidgets.QCheckBox("Aspheres on rear surfaces")
        self.aspheres.setChecked(o.aspheres)
        self.spherize = QtWidgets.QCheckBox("Then fade aspheres out (all-spherical design)")
        self.spherize.setChecked(o.spherize)
        self.stop_shift = QtWidgets.QCheckBox("Stop shift on the glass system")
        self.stop_shift.setChecked(o.stop_shift)
        self.hops = QtWidgets.QCheckBox("Bending hops (basin hopping, slower)")
        self.hops.setChecked(o.bending_hops)
        for w in (self.wf_phase, self.wf_conv, self.aspheres, self.spherize, self.stop_shift,
                  self.hops):
            f.addWidget(w)
            if isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self._update_enabled)
        v.addWidget(g)

        self.arch_box = QtWidgets.QGroupBox("Architectures (power patterns)")
        f = QtWidgets.QGridLayout(self.arch_box)
        self.arch = {}
        for k, name in enumerate(PATTERNS):
            cb = QtWidgets.QCheckBox(f"{name}  {ARCH_NOTES.get(name, '')}")
            cb.setChecked(name in o.architectures)
            self.arch[name] = cb
            f.addWidget(cb, k // 2, k % 2)
        self.top_k = _ispin(1, len(PATTERNS), o.top_k)
        self.top_k.setToolTip("How many architectures (best in phase space) to materialize")
        f.addWidget(QtWidgets.QLabel("Materialize best"), 2, 0)
        f.addWidget(self.top_k, 2, 1)
        self.n_elem = _ispin(3, 12, o.n_elem)
        self.n_elem.setToolTip("Architectures start with 3 screens; GROW splits screens in "
                               "phase space until the design has this many elements")
        f.addWidget(QtWidgets.QLabel("Elements"), 3, 0)
        f.addWidget(self.n_elem, 3, 1)
        v.addWidget(self.arch_box)

        self.conv_box = QtWidgets.QGroupBox("Conventional multi-start")
        f = QtWidgets.QFormLayout(self.conv_box)
        self.n_starts = _ispin(1, 200, o.n_starts)
        self.seed = _ispin(0, 2**31 - 1, o.seed)
        f.addRow("Starts", self.n_starts)
        f.addRow("Seed", self.seed)
        v.addWidget(self.conv_box)

        g = QtWidgets.QGroupBox("Glass")
        f = QtWidgets.QVBoxLayout(g)
        self.use_cat = QtWidgets.QCheckBox("Real catalog glasses (preferred only)")
        self.use_cat.setChecked(o.catalog is not None)
        self.use_cat.toggled.connect(self._update_enabled)
        row = QtWidgets.QHBoxLayout()
        self.cat_path = QtWidgets.QLineEdit(o.catalog or "")
        self.cat_path.setPlaceholderText("AGF catalog file")
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_catalog)
        row.addWidget(self.cat_path, 1)
        row.addWidget(browse)
        f.addWidget(self.use_cat)
        f.addLayout(row)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("Manufacturing limits")
        f = QtWidgets.QFormLayout(g)
        self.tmin = _spin(0, 100, o.tmin, 0.5, 2, " mm")
        self.tmax = _spin(0.1, 200, o.tmax, 0.5, 2, " mm")
        self.emin = _spin(0, 100, o.emin, 0.5, 2, " mm")
        self.gmin = _spin(0, 100, o.gmin, 0.5, 2, " mm")
        f.addRow("Min center thickness", self.tmin)
        f.addRow("Max center thickness", self.tmax)
        f.addRow("Min edge thickness", self.emin)
        f.addRow("Min air gap", self.gmin)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("Advanced")
        f = QtWidgets.QFormLayout(g)
        self.morph_iters = _ispin(5, 500, o.morph_iters)
        self.glass_pull = _spin(0, 10, o.glass_pull, 0.05, 3)
        self.threads = _ispin(1, 256, os.cpu_count() or 1)
        self.threads.setToolTip("Independent optimizations run concurrently")
        f.addRow("LM iterations per morph step", self.morph_iters)
        f.addRow("Glass pull during morph", self.glass_pull)
        f.addRow("Threads", self.threads)
        v.addWidget(g)
        v.addStretch(1)

        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.start_run)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_run)
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_stop)
        v.addLayout(row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(320)
        return scroll

    def _build_tabs(self):
        self.tabs = QtWidgets.QTabWidget()
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self.layout_plot = PlotPanel((9, 4.5))
        lay.addWidget(self.layout_plot, 1)
        row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.setMinimumWidth(260)
        self.source.currentTextChanged.connect(self._source_changed)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slider_changed)
        self.follow = QtWidgets.QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.setToolTip("Jump to the newest step while a run is going")
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setMinimumWidth(90)
        row.addWidget(QtWidgets.QLabel("Path"))
        row.addWidget(self.source)
        row.addWidget(self.slider, 1)
        row.addWidget(self.step_label)
        row.addWidget(self.follow)
        self.layout_config = QtWidgets.QComboBox()
        self.layout_config.setToolTip("configuration (multi-configuration designs)")
        self.layout_config.currentIndexChanged.connect(self._config_changed)
        row.addWidget(self.layout_config)
        lay.addLayout(row)
        self.tabs.addTab(w, "Layout")

        self.merit_plot = PlotPanel((9, 4))
        self.tabs.addTab(self.merit_plot, "Merit")
        self.perf_plot = PlotPanel((9, 6))
        self.tabs.addTab(self.perf_plot, "Performance")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.field_box = QtWidgets.QComboBox()
        self.field_box.currentIndexChanged.connect(lambda _: self._refresh_analysis())
        self.analysis_status = QtWidgets.QLabel("")
        row.addWidget(QtWidgets.QLabel("Field"))
        row.addWidget(self.field_box)
        self.analysis_config = QtWidgets.QComboBox()
        self.analysis_config.currentIndexChanged.connect(self._config_changed)
        row.addWidget(self.analysis_config)
        row.addWidget(self.analysis_status, 1)
        lay.addLayout(row)
        self.analysis_plot = PlotPanel((9, 7))
        lay.addWidget(self.analysis_plot, 1)
        self.analysis_tab = w
        self.tabs.addTab(w, "Image analysis")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.rx = QtWidgets.QPlainTextEdit(readOnly=True)
        self.rx.setFont(mono)
        self.rx.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.rx, "Prescription")
        self.lensdata = LensDataEditor()
        self.lensdata.changed.connect(self._lensdata_changed)
        self.lensdata.convert_requested.connect(self.edit_as_surfaces)
        self.tabs.addTab(self.lensdata, "Lens data")
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.log.setMaximumBlockCount(50000)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.log, "Log")
        self.tabs.currentChanged.connect(lambda _: (self._refresh_merit(),
                                                    self._refresh_analysis()))
        return self.tabs

    def _build_table(self):
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["design", "mean RMS WFE [waves]", "max RMS WFE [waves]", "mean spot [µm]",
             "max distortion", "EFL [mm]", "time [s]"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        return self.table

    def _build_status(self):
        sb = self.statusBar()
        self.status = QtWidgets.QLabel("Ready")
        self.elapsed = QtWidgets.QLabel("")
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(160)
        self.busy.hide()
        sb.addWidget(self.status, 1)
        sb.addPermanentWidget(self.elapsed)
        sb.addPermanentWidget(self.busy)

    # ------------------------------------------------------------------ options
    def _update_enabled(self, *_):
        phase = self.wf_phase.isChecked()
        self.arch_box.setEnabled(phase and not self.running)
        self.conv_box.setEnabled(not phase and not self.running)
        self.spherize.setEnabled(phase and self.aspheres.isChecked())
        self.hops.setEnabled(phase)
        self.cat_path.setEnabled(self.use_cat.isChecked())
        self.btn_run.setEnabled(not self.running)
        self.act_run.setEnabled(not self.running)
        self.btn_stop.setEnabled(self.running)
        self.act_stop.setEnabled(self.running)

    def options(self) -> R.RunOptions:
        cat = self.cat_path.text().strip() if self.use_cat.isChecked() else None
        return R.RunOptions(
            workflow="phase-first" if self.wf_phase.isChecked() else "conventional",
            efl=self.efl.value(), fnum=self.fnum.value(), half_field_deg=self.hfov.value(),
            max_distortion=self.maxdist.value() / 100, max_track=self.track.value(),
            aspheres=self.aspheres.isChecked(),
            spherize=self.spherize.isChecked() and self.spherize.isEnabled(),
            catalog=cat or None,
            architectures=tuple(n for n, cb in self.arch.items() if cb.isChecked()),
            top_k=self.top_k.value(), n_elem=self.n_elem.value(),
            stop_shift=self.stop_shift.isChecked(),
            bending_hops=self.hops.isChecked() and self.hops.isEnabled(),
            morph_iters=self.morph_iters.value(), glass_pull=self.glass_pull.value(),
            tmin=self.tmin.value(), tmax=self.tmax.value(), emin=self.emin.value(),
            gmin=self.gmin.value(), n_starts=self.n_starts.value(), seed=self.seed.value())

    def _browse_catalog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Glass catalog", self.cat_path.text() or str(R.REPO),
            "Zemax glass catalogs (*.agf *.AGF);;All files (*)")
        if path:
            self.cat_path.setText(path)
            self.use_cat.setChecked(True)

    # ------------------------------------------------------------------ running
    def start_run(self):
        if self.running:
            return
        opts = self.options()
        if opts.workflow == "phase-first" and not opts.architectures:
            QtWidgets.QMessageBox.warning(self, "NovaOptics", "Select at least one architecture.")
            return
        if opts.catalog and not os.path.exists(opts.catalog):
            QtWidgets.QMessageBox.warning(self, "NovaOptics",
                                          f"Catalog not found:\n{opts.catalog}")
            return
        os.environ["NOVAOPTICS_THREADS"] = str(self.threads.value())
        self.running = True
        self.t_start = time.time()
        self.run_spec = opts.spec()
        for name in self.live_names:  # drop the previous run's live paths
            self.paths.pop(name, None)
        self.live_names = [LATEST]
        self.paths[LATEST] = []
        self.path_spec[LATEST] = self.run_spec
        self._sync_sources(select=LATEST)
        self.follow.setChecked(True)
        self.log.appendPlainText(f"=== {opts.workflow} run, {time.strftime('%H:%M:%S')}\n{opts}")
        self.status.setText("Starting… (the first run compiles; later runs reuse the cache)")
        self.busy.show()
        self._update_enabled()
        self.tabs.setCurrentIndex(0)
        threading.Thread(target=self._job, args=(opts,), daemon=True,
                         name="novaoptics-run").start()

    def _job(self, opts):
        b = self.bridge
        try:
            R.run(opts, on_snap=b.snap.emit, on_design=b.design.emit, on_log=b.log.emit,
                  as_components=True)
            b.finished.emit(None)
        except progress.Cancelled:
            b.finished.emit("cancelled")
        except Exception:  # noqa: BLE001 - reported in the GUI
            b.finished.emit(traceback.format_exc())

    def stop_run(self):
        if self.running:
            progress.request_cancel()
            self.status.setText("Stopping after the current optimizer step…")

    def _on_snap(self, label, snap):
        name = f"live: {label}"
        if name not in self.paths:
            self.paths[name] = []
            self.path_spec[name] = self.run_spec
            self.live_names.append(name)
        self.paths[name].append(snap)
        self.paths[LATEST].append(snap)
        self.snap_label[id(snap)] = label
        self._last = (label, snap)
        self._dirty = True

    def _on_design(self, d):
        self.add_design(d, select=not self.table.selectedItems())

    def _on_log(self, line):
        self.log.appendPlainText(line)

    def _on_finished(self, result):
        self.running = False
        self.busy.hide()
        self._update_enabled()
        dt = time.time() - self.t_start
        if result is None:
            self.status.setText(f"Finished in {dt:.0f} s")
        elif result == "cancelled":
            self.status.setText(f"Stopped after {dt:.0f} s")
            self.log.appendPlainText("Run stopped.")
        else:
            self.status.setText("Run failed (see Log)")
            self.log.appendPlainText(result)
            QtWidgets.QMessageBox.critical(self, "NovaOptics", result.strip().splitlines()[-1])
        progress.reset_cancel()
        self._tick()

    def _tick(self):
        if self.running:
            self.elapsed.setText(f"{time.time() - self.t_start:.0f} s")
        if not self._dirty:
            return
        self._dirty = False
        self._sync_sources()
        label, sn = self._last
        if self.running:
            self.status.setText(f"{label}: {sn.stage} {sn.step} · mean RMS WFE "
                                f"{sn.mean_wfe:.3f} waves · {sn.note}")
        cur = self.source.currentText()
        if cur in self.live_names:
            n = len(self.paths[cur])
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(n - 1, 0))
            self.slider.blockSignals(False)
            if self.follow.isChecked() and self.slider.value() != n - 1:
                self.slider.setValue(n - 1)
            else:
                self._update_step_label()
        if self.tabs.currentWidget() is self.merit_plot:
            self._refresh_merit()

    # ------------------------------------------------------------------ designs
    def add_design(self, d: R.Design, select=True):
        self.designs.append(d)
        q = d.perf
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [d.name, f"{q['mean_wfe']:.3f}", f"{q['max_wfe']:.3f}", f"{q['mean_spot_um']:.1f}",
                f"{100 * q['max_distortion']:.2f} %", f"{q['efl']:.3f}", f"{d.seconds:.0f}"]
        for c, val in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(val)
            if c:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, c, it)
        name = self._design_path_name(d)
        if isinstance(d, R.EngineDesign):  # pipeline path (if any), then the design
            self.paths[name] = list(d.snaps) + [_EngineSnap(d)]
        elif d.snaps:
            self.paths[name] = d.snaps
        else:
            self.paths[name] = [_FinalSnap(d)]
        self.path_spec[name] = d.spec
        self._sync_sources()
        if select:
            self.table.selectRow(r)

    def _design_path_name(self, d):
        k = self.designs.index(d) + 1
        return f"design {k}: {d.name}"

    def selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.designs[r] for r in rows]

    def _selection_changed(self):
        sel = self.selected()
        if not sel:
            return
        d = sel[0]
        self._sync_sources(select=self._design_path_name(d))
        self.slider.setValue(self.slider.maximum())
        self._refresh_perf(sel)
        self._refresh_rx(sel)
        self._refresh_merit()
        self._sync_configs(d)
        self._sync_fields(d)
        self._refresh_analysis()
        self.lensdata.show_design(d)

    # ------------------------------------------------------------------ image analysis
    def _sync_fields(self, d):
        if isinstance(d, R.EngineDesign):
            _, _, sp = _engine_model(d, max(self.analysis_config.currentIndex(), 0))
            labels = [f"{x:g}°, {y:g}°" if x else f"{y:g}°" for x, y in sp.fields]
        else:
            labels = [f"{f * d.spec.half_field_deg:.1f}°" for f in d.spec.field_fracs]
        if [self.field_box.itemText(i) for i in range(self.field_box.count())] != labels:
            k = self.field_box.currentIndex()
            self.field_box.blockSignals(True)
            self.field_box.clear()
            self.field_box.addItems(labels)
            self.field_box.setCurrentIndex(min(max(k, 0), len(labels) - 1))
            self.field_box.blockSignals(False)

    def _refresh_analysis(self):
        sel = self.selected()
        if self.tabs.currentWidget() is not self.analysis_tab or not sel:
            return
        key = (sel[0], self.field_box.currentIndex(), max(self.analysis_config.currentIndex(), 0))
        if key == self._analysis_shown or key[1] < 0:
            return
        self._analysis_shown = key
        self.analysis_status.setText(f"computing {sel[0].name} …")
        self.analyzer.request(key)

    def _on_analysis(self, key, res):
        d, k, _ = key
        if key != self._analysis_shown:
            return  # a newer request is on its way
        fig = self.analysis_plot.figure
        if isinstance(res, Exception):
            self.analysis_plot.message(f"analysis failed: {res}")
            self.analysis_status.setText("")
            return
        from ..engine.plots import analysis_figure

        analysis_figure(fig, res["spot"], res["opd"], res["psf"], res["mtf"],
                        f"{d.name} · field {self.field_box.itemText(k)}")
        self.analysis_plot.canvas.draw_idle()
        self.analysis_status.setText(
            f"Strehl {res['psf']['strehl']:.3f} · RMS spot {res['spot']['rms_um']:.2f} µm · "
            f"working F/{res['spot']['fno_working']:.2f}")

    def _refresh_rx(self, sel):
        out = []
        for d in sel:
            q = d.perf
            if isinstance(d, R.EngineDesign):
                head = (f"EFL {q['efl']:.3f} mm   EPD {d.spec.epd:g} mm   "
                        f"{len(d.problem.configs)} configuration(s)   ")
            else:
                head = (f"EFL {q['efl']:.3f} mm   F/{d.spec.fnum:g}   "
                        f"±{d.spec.half_field_deg:g}°   ")
            out += [f"{d.name}", "=" * len(d.name), d.note,
                    head +
                    f"mean / max RMS WFE {q['mean_wfe']:.3f} / {q['max_wfe']:.3f} waves   "
                    f"mean spot {q['mean_spot_um']:.1f} µm   max distortion "
                    f"{100 * q['max_distortion']:.2f} %", "", d.prescription, "", ""]
        self.rx.setPlainText("\n".join(out))

    def _refresh_perf(self, sel):
        draw_performance(self.perf_plot.figure, sel)
        self.perf_plot.canvas.draw_idle()

    def _refresh_merit(self):
        if self.tabs.currentWidget() is not self.merit_plot:
            return
        if self.running or not self.selected():
            series = [(n.replace("live: ", ""), self.paths[n]) for n in self.live_names
                      if n != LATEST and self.paths.get(n)]
            title = "this run (one line per architecture)"
        else:
            series = [(d.name, d.snaps) for d in self.selected() if d.snaps]
            title = "selected designs"
        fig = self.merit_plot.figure
        fig.clear()
        if not series:
            self.merit_plot.message("No pipeline path recorded for this selection.")
            return
        # Recorders of materialized architectures start with the shared ideal stage.
        series.sort(key=lambda s: -len(s[1]))
        ax = fig.add_subplot()
        P.draw_trajectory(ax, series)
        ax.set_title(title, fontsize=9, color=P.INK, loc="left")
        self.merit_plot.canvas.draw_idle()

    # ------------------------------------------------------------------ layout path
    def _sync_sources(self, select=None):
        names = list(self.paths)
        cur = select or self.source.currentText()
        self.source.blockSignals(True)
        if [self.source.itemText(i) for i in range(self.source.count())] != names:
            self.source.clear()
            self.source.addItems(names)
        if cur in names:
            self.source.setCurrentText(cur)
        self.source.blockSignals(False)
        if select:
            self._source_changed(self.source.currentText())

    def _source_changed(self, name):
        snaps = self.paths.get(name, [])
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(snaps) - 1, 0))
        self.slider.setValue(max(len(snaps) - 1, 0))
        self.slider.blockSignals(False)
        self._slider_changed(self.slider.value())

    def _slider_changed(self, i):
        name = self.source.currentText()
        snaps = self.paths.get(name, [])
        self._update_step_label()
        if not snaps:
            return
        sn = snaps[min(i, len(snaps) - 1)]
        if isinstance(sn, _EngineSnap):
            d = sn.design
            c = max(self.layout_config.currentIndex(), 0)
            title = f"{d.name} · mean RMS WFE {d.perf.get('mean_wfe', float('nan')):.3f} waves"
            if len(d.problem.configs) > 1:
                title += f" · {self.layout_config.currentText()}"
            self.renderer.request(View(None, None, d.spec, 1.0, title, engine=d, config=c))
            return
        label = (self.snap_label.get(id(sn)) or getattr(sn, "label", None)
                 or name.replace("live: ", ""))
        self.renderer.request(_view(sn, self.path_spec[name], label))

    def _update_step_label(self):
        n = len(self.paths.get(self.source.currentText(), []))
        self.step_label.setText(f"{min(self.slider.value() + 1, n)} / {n}")

    def _on_geometry(self, view, geo):
        fig = self.layout_plot.figure
        fig.clear()
        ax = fig.add_subplot()
        if isinstance(geo, Exception):
            ax.axis("off")
            ax.text(0.5, 0.5, f"cannot trace this system:\n{geo}", ha="center", va="center",
                    color=P.MUTED, transform=ax.transAxes)
        else:
            if view.engine is not None:
                from ..engine.plots import draw_system

                system, surfs, spec = _engine_model(view.engine, view.config)
                draw_system(ax, surfs, system, spec, title=view.title, geometry=geo)
                labels = [f"{np.hypot(*f):.0f}°" for f in spec.fields]
            else:
                P.draw_layout(ax, view.params, view.layout, view.spec, view.beta,
                              title=view.title, geometry=geo)
                labels = [f"{f * view.spec.half_field_deg:.0f}°" for f in view.spec.field_fracs]
            handles = [matplotlib.lines.Line2D([], [], color=P.SERIES[i % len(P.SERIES)],
                                               lw=1.5, label=lab)
                       for i, lab in enumerate(labels)]
            handles += [matplotlib.lines.Line2D([], [], color=P.PHASE, lw=1.8,
                                                ls=(0, (2, 1.5)), label="ideal phase"),
                        matplotlib.patches.Rectangle((0, 0), 1, 1, fc=P.GLASS, ec=P.GLASS_EDGE,
                                                     label="glass")]
            ax.legend(handles=handles, loc="lower left", ncol=len(handles), frameon=False,
                      fontsize=7.5, labelcolor=P.INK)
        self.layout_plot.canvas.draw_idle()

    # ------------------------------------------------------------------ configurations
    def _sync_configs(self, d):
        names = ([c.name or f"config {i + 1}" for i, c in enumerate(d.problem.configs)]
                 if isinstance(d, R.EngineDesign) else [])
        for box in (self.layout_config, self.analysis_config):
            if [box.itemText(i) for i in range(box.count())] != names:
                box.blockSignals(True)
                box.clear()
                box.addItems(names)
                box.blockSignals(False)
            box.setVisible(len(names) > 1)

    def _config_changed(self, i):
        for box in (self.layout_config, self.analysis_config):
            if box.currentIndex() != i and i >= 0:
                box.blockSignals(True)
                box.setCurrentIndex(i)
                box.blockSignals(False)
        sel = self.selected()
        if sel:
            self._sync_fields(sel[0])
        self._slider_changed(self.slider.value())
        self._refresh_analysis()

    # ------------------------------------------------------------------ design operations
    def _engine_selected(self, convert=True):
        sel = self.selected()
        if not sel:
            QtWidgets.QMessageBox.information(self, "NovaOptics", "Select a design first.")
            return None
        d = sel[0]
        if not isinstance(d, R.EngineDesign):
            if not convert:
                return None
            d = R.evaluate_engine(R.engine_from_design(d))
            self.add_design(d, select=False)
        return d

    def edit_as_surfaces(self, d=None):
        d = d or (self.selected() or [None])[0]
        if d is None:
            return
        e = R.evaluate_engine(R.engine_from_design(d)) if not isinstance(d, R.EngineDesign) \
            else d
        if e is not d:
            self.add_design(e)
        self.tabs.setCurrentWidget(self.lensdata)

    def _action(self, label, fn):
        """Run fn() -> EngineDesign in the background; the result is added as a design."""
        if self.running:
            return
        self.running = True
        self.t_start = time.time()
        self.busy.show()
        self._update_enabled()
        self.status.setText(f"{label}…")
        self.log.appendPlainText(f"=== {label}, {time.strftime('%H:%M:%S')}")
        b = self.bridge

        def job():
            progress.reset_cancel()
            try:
                import contextlib

                with contextlib.redirect_stdout(R._Lines(b.log.emit)):
                    out = fn()
                for d in (out if isinstance(out, list) else [out]):
                    R.evaluate_engine(d)
                    b.design.emit(d)
                b.finished.emit(None)
            except progress.Cancelled:
                b.finished.emit("cancelled")
            except Exception:  # noqa: BLE001 - reported in the GUI
                b.finished.emit(traceback.format_exc())

        threading.Thread(target=job, daemon=True, name="novaoptics-action").start()

    def _derived(self, d, prob, params, label, seconds):
        return R.EngineDesign(f"{d.name} · {label}", prob, params, d.spec, seconds,
                              f"{label} (from {d.name})", d.catalog, free=None)

    def optimize_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            free = d.free or ED.variables(d.problem)
            p, info = ED.optimize_design(d.problem, d.params, d.spec, free, iters=200)
            print(f"cost {info['history'][0]:.4g} -> {info['cost']:.4g} in {info['iters']} "
                  f"iterations")
            e = self._derived(d, d.problem, p, "optimized", time.time() - t)
            e.free = d.free
            return e

        self._action(f"optimizing {d.name}", fn)

    def add_element_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p, res = ED.add_element(d.problem, d.params, d.spec, log=print)
            return self._derived(d, prob, p, "+1 element", time.time() - t)

        self._action(f"adding an element to {d.name} (every air space is tried)", fn)

    def cement_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        S = d.problem.system.surfaces
        rears = [i for i, s in enumerate(S) if i > 0 and s.interact == "refract"
                 and S[i - 1].material not in ("air", "", "vacuum")
                 and S[i - 1].interact == "refract"]
        if not rears:
            QtWidgets.QMessageBox.information(self, "NovaOptics", "No glass element found.")
            return
        items = [f"surface {i}: rear of {S[i - 1].label or 'element'} ({S[i - 1].material})"
                 for i in rears]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Cement",
                                                  "Cement a new glass behind:", items, 0, False)
        if not ok:
            return
        rear = rears[items.index(item)]
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p = ED.cement(d.problem, d.params, d.spec, rear)
            return self._derived(d, prob, p, f"cemented at surface {rear}", time.time() - t)

        self._action(f"cementing a glass behind surface {rear} of {d.name}", fn)

    def _pick_component(self, d, kinds, title):
        from ..engine.components import group

        comps = [(k, c) for k, c in enumerate(group(d.problem.system, d.params["s"]))
                 if c.kind in kinds]
        if not comps:
            QtWidgets.QMessageBox.information(self, "NovaOptics", f"No {' / '.join(kinds)} "
                                              "component in this design.")
            return None
        items = [f"{k}: {c.kind} {c.label} (surfaces {c.start}-{c.end})" for k, c in comps]
        pre = self.lensdata.selected_component()
        cur = next((i for i, (k, _) in enumerate(comps) if k == pre), 0)
        item, ok = QtWidgets.QInputDialog.getItem(self, title, "Component:", items, cur, False)
        return comps[items.index(item)][0] if ok else None

    def materialize_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        k = self._pick_component(d, ("screen",), "Materialize")
        if k is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p = ED.materialize(d.problem, d.params, d.spec, k, log=print)
            return self._derived(d, prob, p, f"component {k} materialized", time.time() - t)

        self._action(f"materializing component {k} of {d.name}", fn)

    def dematerialize_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        k = self._pick_component(d, ("lens",), "Dematerialize")
        if k is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p = ED.dematerialize(d.problem, d.params, d.spec, k)
            return self._derived(d, prob, p, f"component {k} as a phase screen", time.time() - t)

        self._action(f"dematerializing component {k} of {d.name}", fn)

    def snap_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p = ED.snap_glasses(d.problem, d.params, d.spec)
            return self._derived(d, prob, p, "catalog glasses", time.time() - t)

        self._action(f"snapping the model glasses of {d.name}", fn)

    def per_wavelength_selected(self):
        d = self._engine_selected()
        if d is None:
            return
        from ..engine import design as ED

        def fn():
            t = time.time()
            prob, p = ED.per_wavelength_focus(d.problem, d.params, d.spec)
            p, _ = ED.optimize_design(prob, p, d.spec,
                                      ED.variables(prob, curvatures=False, thicknesses=False),
                                      iters=100)
            return self._derived(d, prob, p, "per-wavelength focus", time.time() - t)

        self._action(f"one image plane per wavelength for {d.name}", fn)

    def _lensdata_changed(self, d):
        self.status.setText(f"Re-evaluating {d.name}…")

        def job():
            try:
                R.evaluate_engine(d)
                self.bridge.log.emit(f"lens data of {d.name} applied")
            except Exception:  # noqa: BLE001
                self.bridge.log.emit(traceback.format_exc())
            self.bridge.refresh.emit(d)

        threading.Thread(target=job, daemon=True).start()

    def _refresh_design(self, d):
        r = self.designs.index(d)
        q = d.perf
        vals = [d.name, f"{q['mean_wfe']:.3f}", f"{q['max_wfe']:.3f}", f"{q['mean_spot_um']:.1f}",
                f"{100 * q['max_distortion']:.2f} %", f"{q['efl']:.3f}", f"{d.seconds:.0f}"]
        for c, val in enumerate(vals):
            self.table.item(r, c).setText(val)
        self._analysis_shown = None
        self._selection_changed()
        self.status.setText(f"{d.name}: mean RMS WFE {q['mean_wfe']:.3f} waves")

    def import_zmx(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import Zemax lens", str(R.REPO), "Zemax lens (*.zmx);;All files (*)")
        if not path:
            return
        try:
            d = R.evaluate_engine(R.engine_from_zmx(path))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "NovaOptics", f"Cannot import {path}:\n{e}")
            return
        self.add_design(d)
        self.status.setText(f"Imported {path}")

    def export_zmx(self):
        d = self._engine_selected()
        if d is None:
            return
        if len(d.problem.configs) > 1:
            QtWidgets.QMessageBox.information(
                self, "NovaOptics", "Multi-configuration export is not supported yet; the "
                "first configuration is written.")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Zemax lens",
                                                        "design.zmx", "Zemax lens (*.zmx)")
        if not path:
            return
        from ..engine import zmx
        from ..engine import merit as EM

        try:
            zmx.save(path, d.problem.system, EM.config_surfaces(d.params, d.problem, 0),
                     EM.config_spec(d.spec, d.problem.configs[0]), d.name)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "NovaOptics", f"Cannot export:\n{e}")
            return
        self.status.setText(f"Exported {path}")

    # ------------------------------------------------------------------ files
    def open_designs(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open designs", str(R.REPO / "examples" / "output"),
                "Designs (*.json);;All files (*)")
        if not path:
            return
        cat = self.cat_path.text().strip() or None
        try:
            designs = R.load(path, default_catalog=cat)
            self.status.setText(f"Evaluating {len(designs)} designs…")
            QtWidgets.QApplication.processEvents()
            for d in designs:
                if not d.perf:
                    R.evaluate_engine(d) if isinstance(d, R.EngineDesign) else R.evaluate(d)
                self.add_design(d, select=False)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "NovaOptics", f"Cannot open {path}:\n{e}")
            return
        self.table.selectRow(self.table.rowCount() - len(designs))
        self.status.setText(f"Opened {len(designs)} designs from {path}")

    def save_designs(self):
        designs = self.selected() or self.designs
        if not designs:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save designs", "designs.json",
                                                        "Designs (*.json)")
        if path:
            R.save(designs, path)
            self.status.setText(f"Saved {len(designs)} designs to {path}")

    def export_figure(self):
        w = self.tabs.currentWidget()
        panel = {self.merit_plot: self.merit_plot, self.perf_plot: self.perf_plot}.get(
            w, self.layout_plot)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", "figure.png",
                                                        "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            panel.figure.savefig(path, dpi=200, facecolor=P.SURFACE)

    def about(self):
        QtWidgets.QMessageBox.about(
            self, "About NovaOptics",
            "<b>NovaOptics</b><br>Optical design by morphing ideal phase screens into "
            "real glass while the merit function stays minimized.<br><br>"
            "Differentiable real-ray tracing in JAX; Levenberg–Marquardt with exact "
            "Jacobians.")

    def closeEvent(self, e):
        if self.running:
            progress.request_cancel()
        super().closeEvent(e)


@dataclass(eq=False)
class _EngineSnap:
    """Layout-path entry of an engine design (drawn as a surface system)."""

    design: object
    label: str = ""
    mean_wfe: float = float("nan")


@dataclass(eq=False)
class _FinalSnap:
    """Stand-in snapshot for a loaded design without a recorded path."""

    design: R.Design

    def __getattr__(self, k):
        d = self.__dict__["design"]
        return {"params": d.params, "layout": d.layout, "beta": 1.0, "stage": "final",
                "step": 0, "note": "", "mean_wfe": d.perf.get("mean_wfe", float("nan")),
                "label": d.name}[k]


def draw_performance(fig, designs):
    """RMS WFE, RMS spot, field curves and distortion vs field for several designs."""
    fig.clear()
    if not designs:
        return
    axes = fig.subplots(2, 2)
    for ax in axes.flat:
        P._style(ax)
        ax.grid(color=P.GRID, lw=0.6)
    a, b, c, e = axes.flat
    for j, d in enumerate(designs):
        q, col = d.perf, P.SERIES[j % len(P.SERIES)]
        a.plot(q["field_deg"], q["rms_wfe"], color=col, lw=2, label=d.name)
        b.plot(q["field_deg"], q["rms_spot_um"], color=col, lw=2)
        if np.all(np.isfinite(q["tangential"])):
            c.plot(q["tangential"], q["curve_deg"], color=col, lw=2)
            c.plot(q["sagittal"], q["curve_deg"], color=col, lw=2, ls="--")
        nz = q["field_deg"] > 0  # distortion is undefined (0/0) on axis
        e.plot(100 * q["distortion"][nz], q["field_deg"][nz], color=col, lw=2)
        for m, cq in enumerate(q.get("configs", [])[1:], start=1):  # further configurations
            ls = [":", "-.", (0, (5, 2)), (0, (1, 1))][(m - 1) % 4]
            a.plot(cq["field_deg"], cq["rms_wfe"], color=col, lw=1.5, ls=ls,
                   label=f"{d.name}: {cq['name']}")
            b.plot(cq["field_deg"], cq["rms_spot_um"], color=col, lw=1.5, ls=ls)
            nz = cq["field_deg"] > 0
            e.plot(100 * cq["distortion"][nz], cq["field_deg"][nz], color=col, lw=1.5, ls=ls)
    a.axhline(0.07, color=P.MUTED, lw=1, ls=":")
    a.text(0.2, 0.072, "Maréchal 0.07", fontsize=7.5, color=P.MUTED, va="bottom")
    a.set(xlabel="field angle [deg]", ylabel="RMS wavefront error [waves]")
    b.set(xlabel="field angle [deg]", ylabel="RMS spot radius [µm]")
    c.axvline(0, color=P.MUTED, lw=0.8)
    c.set(xlabel="focus shift [mm] (solid T, dashed S)", ylabel="field angle [deg]")
    e.axvline(0, color=P.MUTED, lw=0.8)
    e.set(xlabel="distortion [%]", ylabel="field angle [deg]")
    for ax in axes.flat:
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.xaxis.label.set_color(P.INK)
        ax.yaxis.label.set_color(P.INK)
    a.legend(frameon=False, fontsize=7.5, labelcolor=P.INK)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("NovaOptics")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    for path in argv[1:]:
        QTimer.singleShot(0, lambda p=path: win.open_designs(p))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
