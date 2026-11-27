"""Zemax .zmx import / export."""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from novaoptics.engine import trace as T
from novaoptics.engine import zmx

RESULTS = Path(__file__).parents[1] / "examples" / "output" / "results.json"

SAMPLE = """VERS 190513 80 123457 L123457
MODE SEQ
NAME test lens
UNIT IN X W X CM MR CPMM
ENPD 0.5
FTYP 0 0 2 2 0 0 0
XFLN 0 0 0 0 0 0 0 0 0 0 0 0
YFLN 0 5 0 0 0 0 0 0 0 0 0 0
FWGN 1 1 1 1 1 1 1 1 1 1 1 1
WAVM 1 0.55 1
WAVM 2 0.65 0.5
WAVM 3 0.55 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ INFINITY
SURF 1
  STOP
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0.2
SURF 2
  COMM "lens front"
  TYPE EVENASPH
  CURV 0.5 0 0 0 0 ""
  CONI -0.3
  PARM 1 0.001
  PARM 2 0.002
  DISZ 0.25
  GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0
  DIAM 0.4 1 0 0 1 ""
SURF 3
  TYPE STANDARD
  CURV -0.1 0 0 0 0 ""
  DISZ 1.0
SURF 4
  TYPE COORDBRK
  PARM 1 0
  PARM 2 0
  PARM 3 45
  PARM 4 0
  PARM 5 0
  PARM 6 0
  DISZ 0
SURF 5
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0
  GLAS MIRROR 0 0 0 0 0 0 0 0 0 0
SURF 6
  TYPE COORDBRK
  PARM 3 45
  DISZ -1.5
SURF 7
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0
"""


def test_import_utf16_units_mirror_and_breaks(tmp_path):
    f = tmp_path / "sample.zmx"
    f.write_bytes(b"\xff\xfe" + SAMPLE.encode("utf-16-le"))
    system, params, spec, info = zmx.load(f)
    S = system.surfaces
    assert info["name"] == "test lens"
    assert spec.epd == pytest.approx(12.7)
    assert spec.wavelengths == (0.55, 0.65) and spec.iref == 0  # FTYP: 2 wavelengths
    assert spec.fields == ((0.0, 0.0), (0.0, 5.0))
    assert S[0].stop and S[1].shape == "even" and S[1].material == "N-BK7"
    assert float(params[1]["c"]) == pytest.approx(0.5 / 25.4)
    assert float(params[1]["k"]) == pytest.approx(-0.3)
    assert float(params[1]["a2"]) == pytest.approx(0.001 / 25.4)  # r^2: 1/length
    assert float(params[1]["coef"][0]) == pytest.approx(0.002 / 25.4**3)
    assert float(params[1]["t"]) == pytest.approx(0.25 * 25.4)
    assert S[3].shape == "cb" and float(params[3]["tx"]) == 45.0
    assert S[4].interact == "reflect" and S[4].material == "air"
    assert float(params[5]["t"]) == pytest.approx(-1.5 * 25.4)
    # the folded beam reaches the image
    xy = T.aim_chief(params, system, (0.0, 0.0), 0.55)
    r = T.trace_ray(params, system, 0.55, *T.launch(xy, (0.0, 0.0), 0.0, 0.5, spec.epd, system))
    assert bool(r["ok"])
    assert abs(float(r["dir"][1])) > 0.9  # travelling along y after the fold


@pytest.mark.skipif(not RESULTS.exists(), reason="example results not present")
def test_export_import_round_trip(tmp_path):
    from novaoptics.engine import convert
    from novaoptics.gui import runner as R

    d = R.from_json(json.loads(RESULTS.read_text())[0])  # aspheric: EVENASPH export
    s1, p1, e1 = convert.from_layout(d.params, d.layout, d.spec.epd, [0, 20])
    zmx.save(tmp_path / "x.zmx", s1, p1, e1)
    s2, p2, e2, _ = zmx.load(tmp_path / "x.zmx")
    assert [s.material for s in s1.surfaces] == [s.material for s in s2.surfaces]
    for f in ((0.0, 0.0), (0.0, 20.0)):
        a = T.trace_ray(p1, s1, 0.5875618, *T.launch(T.aim_chief(p1, s1, f, 0.5875618), f, 0.3,
                                                     0.8, e1.epd, s1))["pts"][-1]
        b = T.trace_ray(p2, s2, 0.5875618, *T.launch(T.aim_chief(p2, s2, f, 0.5875618), f, 0.3,
                                                     0.8, e2.epd, s2))["pts"][-1]
        assert float(jnp.max(jnp.abs(a - b))) < 1e-8


@pytest.mark.parametrize("change,msg", [
    ("  DISZ INFINITY", "  DISZ 100"), ("FTYP 0 0 2", "FTYP 1 0 2"),
    ("  TYPE EVENASPH", "  TYPE TOROIDAL"), ("PWAV 1", "PWAV 1\nMNUM 3 1")])
def test_unsupported_content_is_refused(change, msg):
    with pytest.raises(zmx.ZmxError):
        zmx.from_text(SAMPLE.replace(change, msg, 1))
