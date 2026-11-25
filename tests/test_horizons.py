import datetime as dt
import json

import pytest

from gravity_sandbox import horizons

# Trimmed from a real API response (Earth-Moon barycentre, 2026-11-25).
SAMPLE_RESULT = """
Target body name: Earth-Moon Barycenter (3)       {source: DE441}
Center body name: Solar System Barycenter (0)     {source: DE441}
Reference frame : Ecliptic of J2000.0
*******************************************************************************
            JDTDB,            Calendar Date (TDB),                      X,                      Y,                      Z,                     VX,                     VY,                     VZ,
*******************************************************************************
$$SOE
2461369.500000000, A.D. 2026-Nov-25 00:00:00.0000,  6.840829765127650E+07,  1.300727306257762E+08,  4.558325551509857E+03, -2.686013122333976E+01,  1.371811555446543E+01, -9.424536862141508E-04,
2461370.500000000, A.D. 2026-Nov-26 00:00:00.0000,  6.607715869939561E+07,  1.312377965050650E+08,  4.478225907146931E+03, -2.710005727530719E+01,  1.325026700174740E+01, -9.116618800524279E-04,
$$EOE
*******************************************************************************
"""


def test_parse_vectors_reads_the_first_row():
    position, velocity = horizons.parse_vectors(SAMPLE_RESULT)
    assert position == pytest.approx([6.840829765127650e7, 1.300727306257762e8, 4.558325551509857e3])
    assert velocity == pytest.approx([-2.686013122333976e1, 1.371811555446543e1, -9.424536862141508e-4])


def test_parse_vectors_rejects_a_response_without_data():
    with pytest.raises(ValueError):
        horizons.parse_vectors("No ephemeris for target \"Planet X\"")


def test_build_url_requests_barycentric_ecliptic_vectors():
    url = horizons.build_url("3", dt.date(2026, 11, 25))
    for fragment in ("COMMAND=%273%27", "CENTER=%27500%400%27", "REF_PLANE=%27ECLIPTIC%27",
                     "START_TIME=%272026-11-25%27", "STOP_TIME=%272026-11-26%27", "OUT_UNITS=%27KM-S%27"):
        assert fragment in url


def test_bundled_snapshot_has_every_body():
    snapshot = json.loads(horizons.SNAPSHOT_FILE.read_text(encoding="utf-8"))
    assert set(snapshot["bodies"]) == set(horizons.HORIZONS_IDS)
    for state in snapshot["bodies"].values():
        assert len(state["position_km"]) == 3 and len(state["velocity_km_s"]) == 3
