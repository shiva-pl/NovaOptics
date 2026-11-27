"""Lens data editor for surface-engine designs: surfaces, configurations, specification.

Rows are surfaces (the object at infinity is implied, the last row is the image).
Radius and thickness can be marked variable for "Optimize".  Configurations
(multi-configuration / zoom / per-wavelength focus) are rows of the second table:
each operand column is a surface parameter that the configuration overrides
("3:t" = thickness after surface 3), with optional own wavelengths and focal length.
"Apply" rebuilds the design from the tables.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from ..engine import merit as EM
from ..engine.system import Spec, Surface, System, surface_params
from . import runner as R

KINDS = ["standard", "even asphere", "Qbfs (type 0)", "Qcon (type 1)", "mirror", "stop",
         "coord break", "image"]
SHAPE = {"standard": "conic", "even asphere": "even", "Qbfs (type 0)": "qbfs",
         "Qcon (type 1)": "qcon", "mirror": "conic", "stop": "conic", "coord break": "cb",
         "image": "conic"}
COLS = ["label", "type", "radius [mm]", "thickness [mm]", "material", "conic", "vary R",
        "vary t"]


def _kind(s: Surface):
    if s.shape == "cb":
        return "coord break"
    if s.interact == "reflect":
        return "mirror"
    if s.stop and s.interact == "none":
        return "stop"
    if s.interact == "none":
        return "image"
    return {"conic": "standard", "even": "even asphere", "qbfs": "Qbfs (type 0)",
            "qcon": "Qcon (type 1)"}[s.shape]


def _fmt_r(c):
    return "inf" if abs(c) < 1e-12 else f"{1.0 / c:.6g}"


def _parse_r(text):
    t = text.strip().lower()
    if t in ("", "inf", "infinity", "0"):
        return 0.0
    return 1.0 / float(t)


def _floats(text):
    return [float(x) for x in text.replace(",", " ").replace(";", " ").split()]


class LensDataEditor(QtWidgets.QWidget):
    """Edits an EngineDesign in place; emits `changed(design)` after Apply."""

    changed = Signal(object)
    convert_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.design = None
        self.rows = []  # source surface index per row (None: new)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.info = QtWidgets.QLabel("Select a design.")
        self.info.setWordWrap(True)
        self.convert_btn = QtWidgets.QPushButton("Edit as surfaces (copy)")
        self.convert_btn.clicked.connect(lambda: self.convert_requested.emit(self.design))
        self.convert_btn.hide()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.info, 1)
        top.addWidget(self.convert_btn)
        v.addLayout(top)

        form = QtWidgets.QHBoxLayout()
        self.epd = QtWidgets.QLineEdit()
        self.fields = QtWidgets.QLineEdit()
        self.fields.setToolTip("field angles x,y [deg] separated by ';'  e.g. 0,0; 0,14; 0,20")
        self.wls = QtWidgets.QLineEdit()
        self.wls.setToolTip("wavelengths [um]; the middle one is the reference")
        self.efl = QtWidgets.QLineEdit()
        self.efl.setToolTip("focal-length target [mm] (empty: none)")
        self.track = QtWidgets.QLineEdit()
        for lab, w in (("EPD", self.epd), ("fields [deg]", self.fields),
                       ("wavelengths [µm]", self.wls), ("EFL", self.efl),
                       ("max track", self.track)):
            form.addWidget(QtWidgets.QLabel(lab))
            form.addWidget(w, 2 if lab in ("fields [deg]", "wavelengths [µm]") else 1)
        v.addLayout(form)

        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        v.addWidget(self.table, 3)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Insert surface", self.insert_row), ("Delete surface", self.delete_row)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        v.addWidget(QtWidgets.QLabel("Configurations (operand columns: surface:parameter, "
                                     "e.g. 6:t; wavelengths per configuration optional)"))
        self.ctable = QtWidgets.QTableWidget(0, 0)
        self.ctable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.ctable, 1)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Add configuration", self.add_config),
                         ("Remove configuration", self.remove_config),
                         ("Add operand", self.add_operand)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.apply)
        row.addWidget(self.apply_btn)
        v.addLayout(row)
        self._set_enabled(False)

    def _set_enabled(self, on):
        for w in (self.table, self.ctable, self.apply_btn, self.epd, self.fields, self.wls,
                  self.efl, self.track):
            w.setEnabled(on)

    # ------------------------------------------------------------------ load
    def show_design(self, d):
        self.design = d
        if not isinstance(d, R.EngineDesign):
            self.info.setText(f"{d.name}: a phase-first element design.  Edit a copy as a "
                              "surface system to change surfaces, cement elements or add "
                              "configurations.")
            self.convert_btn.show()
            self._set_enabled(False)
            self.table.setRowCount(0)
            self.ctable.setRowCount(0)
            return
        self.convert_btn.hide()
        self._set_enabled(True)
        self.info.setText(f"{d.name}  ({d.note})" if d.note else d.name)
        prob, P, spec = d.problem, d.params, d.spec
        self.epd.setText(f"{spec.epd:g}")
        self.fields.setText("; ".join(f"{x:g},{y:g}" for x, y in spec.fields))
        self.wls.setText(" ".join(f"{w:g}" for w in spec.wavelengths))
        self.efl.setText("" if prob.targets.efl is None else f"{prob.targets.efl:g}")
        self.track.setText("" if prob.targets.max_track is None else f"{prob.targets.max_track:g}")
        free = d.free or {"s": [{} for _ in prob.system.surfaces]}
        self.table.setRowCount(0)
        self.rows = []
        for i, (s, p) in enumerate(zip(prob.system.surfaces, P["s"])):
            self._add_row(i, s, p, free["s"][i] if i < len(free["s"]) else {})
        self._load_configs()

    def _add_row(self, at, s: Surface, p, fr, src=None):
        self.table.insertRow(at)
        self.rows.insert(at, at if src is None else src)
        self.table.setItem(at, 0, QtWidgets.QTableWidgetItem(s.label))
        kind = QtWidgets.QComboBox()
        kind.addItems(KINDS)
        kind.setCurrentText(_kind(s))
        self.table.setCellWidget(at, 1, kind)
        cb = s.shape == "cb"
        mat = s.material
        if mat == "model":
            mat = f"{float(p['nd']):.5g}/{float(p['vd']):.4g}"
        vals = ["" if cb else _fmt_r(float(p["c"])), f"{float(p['t']):.6g}",
                "" if s.interact != "refract" else ("" if mat == "air" else mat),
                "" if cb else f"{float(p['k']):.6g}"]
        for c, val in enumerate(vals, start=2):
            self.table.setItem(at, c, QtWidgets.QTableWidgetItem(val))
        for c, key in ((6, "c"), (7, "t")):
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if fr.get(key) else Qt.Unchecked)
            self.table.setItem(at, c, it)

    def _load_configs(self):
        prob, P = self.design.problem, self.design.params
        ops = []
        for conf in prob.configs:
            for o in conf.overrides:
                if o not in ops:
                    ops.append(o)
        heads = ["name"] + [f"{i}:{k}" for i, k in ops] + ["wavelengths [µm]", "EFL"]
        self.ctable.setColumnCount(len(heads))
        self.ctable.setHorizontalHeaderLabels(heads)
        self.ctable.setRowCount(len(prob.configs))
        for r, conf in enumerate(prob.configs):
            self.ctable.setItem(r, 0, QtWidgets.QTableWidgetItem(conf.name or f"{r + 1}"))
            for j, (i, k) in enumerate(ops, start=1):
                key = EM.key_name(i, k)
                v = P["mc"][r].get(key)
                self.ctable.setItem(r, j, QtWidgets.QTableWidgetItem(
                    "" if v is None else f"{float(v):.6g}"))
            wl = "" if conf.wavelengths is None else " ".join(f"{w:g}" for w in conf.wavelengths)
            self.ctable.setItem(r, len(ops) + 1, QtWidgets.QTableWidgetItem(wl))
            self.ctable.setItem(r, len(ops) + 2, QtWidgets.QTableWidgetItem(
                "" if conf.efl is None else f"{conf.efl:g}"))

    # ------------------------------------------------------------------ edits
    def insert_row(self):
        if not self.design or not isinstance(self.design, R.EngineDesign):
            return
        at = max(self.table.currentRow(), 0)
        s = Surface(label="new")
        self._add_row(at, s, surface_params(s), {}, src=None)
        self.rows[at] = None

    def delete_row(self):
        r = self.table.currentRow()
        if r >= 0 and self.table.rowCount() > 2:
            self.table.removeRow(r)
            del self.rows[r]

    def add_config(self):
        r = self.ctable.rowCount()
        if self.ctable.columnCount() == 0:
            self.ctable.setColumnCount(3)
            self.ctable.setHorizontalHeaderLabels(["name", "wavelengths [µm]", "EFL"])
        self.ctable.insertRow(r)
        self.ctable.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{r + 1}"))
        for c in range(1, self.ctable.columnCount()):  # copy the previous row's values
            prev = self.ctable.item(r - 1, c) if r > 0 else None
            self.ctable.setItem(r, c, QtWidgets.QTableWidgetItem(prev.text() if prev else ""))

    def remove_config(self):
        r = self.ctable.currentRow()
        if r >= 0:
            self.ctable.removeRow(r)

    def add_operand(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Add operand", "surface:parameter (parameter: t, c, k, dx, dy, tx, ty, tz, "
            "nd, vd)", text="0:t")
        if not ok or ":" not in text:
            return
        i, k = text.split(":")
        i = int(i)
        if self.ctable.columnCount() == 0:
            self.add_config()
        col = self.ctable.columnCount() - 2  # before wavelengths and EFL
        self.ctable.insertColumn(col)
        self.ctable.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(f"{i}:{k.strip()}"))
        base = self.design.params["s"][i][k.strip()] if i < len(self.design.params["s"]) else 0
        for r in range(self.ctable.rowCount()):
            self.ctable.setItem(r, col, QtWidgets.QTableWidgetItem(f"{float(base):.6g}"))

    # ------------------------------------------------------------------ apply
    def build(self):
        """(problem, params, spec, free) from the tables."""
        d = self.design
        old_S, old_P = d.problem.system.surfaces, d.params["s"]
        S, P, free_s = [], [], []
        n = self.table.rowCount()
        for r in range(n):
            src = self.rows[r]
            base = old_S[src] if src is not None else Surface()
            bp = old_P[src] if src is not None else surface_params(base)
            kind = self.table.cellWidget(r, 1).currentText()
            label = (self.table.item(r, 0).text() if self.table.item(r, 0) else "")
            txt = [(self.table.item(r, c).text() if self.table.item(r, c) else "")
                   for c in range(2, 6)]
            shape = SHAPE[kind]
            interact = {"mirror": "reflect", "stop": "none", "image": "none",
                        "coord break": "none"}.get(kind, "refract")
            mat_text = txt[2].strip()
            prev_mat = S[-1].material if S else "air"
            vals = {}
            if interact == "refract":
                if "/" in mat_text:
                    nd, vd = (float(x) for x in mat_text.split("/"))
                    material, vals = "model", {"nd": nd, "vd": vd}
                else:
                    material = mat_text or "air"
            else:
                material = prev_mat
            same = shape == base.shape
            s = dataclasses.replace(
                base, shape=shape, interact=interact, material=material, label=label,
                stop=(kind == "stop") or (base.stop and kind not in ("image", "coord break")
                                          and interact != "none"),
                n_terms=base.n_terms if same else 0, cb_order=base.cb_order)
            if r == n - 1:
                s = dataclasses.replace(s, interact="none", stop=False)
            p = surface_params(s)
            for k_, v_ in bp.items():
                if k_ in p and np.shape(p[k_]) == np.shape(v_):
                    p[k_] = v_
            if shape != "cb":
                p["c"] = jnp.asarray(_parse_r(txt[0]))
                p["k"] = jnp.asarray(float(txt[3] or 0.0))
            p["t"] = jnp.asarray(float(txt[1] or 0.0))
            for k_, v_ in vals.items():
                p[k_] = jnp.asarray(v_)
            S.append(s)
            P.append(p)
            fr = {}
            if self.table.item(r, 6).checkState() == Qt.Checked and shape != "cb":
                fr["c"] = True
            if self.table.item(r, 7).checkState() == Qt.Checked and r < n - 1:
                fr["t"] = True
            if material == "model":
                fr["nd"] = fr["vd"] = True
            free_s.append(fr)
        stops = [i for i, s in enumerate(S) if s.stop]
        if len(stops) != 1:
            raise ValueError(f"exactly one stop is needed (found {len(stops)})")
        system = System(tuple(S), d.problem.system.z_launch)

        # configurations
        heads = [self.ctable.horizontalHeaderItem(c).text()
                 for c in range(self.ctable.columnCount())]
        ops = [h for h in heads if ":" in h]
        configs, mc = [], []
        for r in range(self.ctable.rowCount()):
            cell = lambda c: (self.ctable.item(r, c).text().strip()
                              if self.ctable.item(r, c) else "")
            over, vals = [], {}
            for h in ops:
                c = heads.index(h)
                if cell(c):
                    i, k = h.split(":")
                    over.append((int(i), k))
                    vals[EM.key_name(int(i), k)] = jnp.asarray(float(cell(c)))
            wl = _floats(cell(heads.index("wavelengths [µm]"))) if "wavelengths [µm]" in heads \
                else []
            efl = cell(heads.index("EFL")) if "EFL" in heads else ""
            configs.append(EM.Config(name=cell(0), overrides=tuple(over),
                                     wavelengths=tuple(wl) or None,
                                     wl_weights=(1.0,) * len(wl) if wl else None,
                                     efl=float(efl) if efl else None))
            mc.append(vals)
        if not configs:
            configs, mc = [EM.Config()], [{}]
        fields = tuple(tuple(_floats(f)[:2] + [0.0] * (2 - len(_floats(f)[:2])))
                       for f in self.fields.text().split(";") if f.strip())
        wls = tuple(_floats(self.wls.text()))
        spec = Spec(epd=float(self.epd.text()), fields=fields or ((0.0, 0.0),),
                    wavelengths=wls or (0.5875618,), wl_weights=(1.0,) * max(len(wls), 1))
        t = d.problem.targets
        targets = dataclasses.replace(
            t, efl=float(self.efl.text()) if self.efl.text().strip() else None,
            max_track=float(self.track.text()) if self.track.text().strip() else None)
        prob = EM.Problem(system, tuple(configs), targets)
        params = {"s": P, "mc": mc}
        free = {"s": free_s, "mc": [{k: True for k in m} for m in mc]}
        return prob, params, spec, free

    def apply(self):
        try:
            prob, params, spec, free = self.build()
        except Exception as e:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "Lens data", str(e))
            return
        d = self.design
        d.problem, d.params, d.spec, d.free = prob, params, spec, free
        self.changed.emit(d)
