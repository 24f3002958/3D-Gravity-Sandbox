"""Real planet positions and velocities from NASA/JPL Horizons.

Standard library only. API documentation:
https://ssd-api.jpl.nasa.gov/doc/horizons.html

Refresh the bundled snapshot (data/solar_system_snapshot.json) with:

    python -m gravity_sandbox.horizons              # today's date (UTC)
    python -m gravity_sandbox.horizons 2030-01-01   # any date
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SNAPSHOT_FILE = Path(__file__).resolve().parent.parent / "data" / "solar_system_snapshot.json"

# Horizons target IDs. Planets with moons use their system barycentre ("3" is the
# Earth-Moon barycentre), which is the right point to follow when a planet and
# its moons are simulated as a single body.
HORIZONS_IDS = {
    "Sun": "10",
    "Mercury": "1",
    "Venus": "2",
    "Earth": "3",
    "Mars": "4",
    "Jupiter": "5",
    "Saturn": "6",
    "Uranus": "7",
    "Neptune": "8",
}


def build_url(target_id: str, date: dt.date) -> str:
    """URL for one state vector (km, km/s) relative to the Solar System barycentre,
    in the ecliptic frame, at 00:00 TDB on ``date``."""
    params = {
        "format": "json",
        "COMMAND": f"'{target_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@0'",
        "START_TIME": f"'{date.isoformat()}'",
        "STOP_TIME": f"'{(date + dt.timedelta(days=1)).isoformat()}'",
        "STEP_SIZE": "'1 d'",
        "VEC_TABLE": "'2'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def parse_vectors(result: str) -> tuple[list[float], list[float]]:
    """First (x, y, z) [km] and (vx, vy, vz) [km/s] row of a Horizons CSV vector table,
    i.e. the lines between $$SOE and $$EOE."""
    if "$$SOE" not in result or "$$EOE" not in result:
        raise ValueError("Horizons response has no $$SOE/$$EOE data block")
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    row = next((line for line in block.splitlines() if line.strip()), None)
    if row is None:
        raise ValueError("Horizons data block is empty")
    fields = [field.strip() for field in row.split(",")]
    numbers = [float(value) for value in fields[2:8]]    # skip the Julian date and calendar date
    return numbers[:3], numbers[3:]


def fetch_state(target_id: str, date: dt.date, timeout: float = 30.0) -> tuple[list[float], list[float]]:
    with urllib.request.urlopen(build_url(target_id, date), timeout=timeout) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"Horizons error for target {target_id}: {payload['error']}")
    return parse_vectors(payload["result"])


def fetch_solar_system(date: dt.date | None = None) -> dict:
    """State vectors for the Sun and planets, in the format presets.solar_system() reads."""
    date = date or dt.datetime.now(dt.timezone.utc).date()
    bodies = {}
    for name, target_id in HORIZONS_IDS.items():
        position, velocity = fetch_state(target_id, date)
        bodies[name] = {"horizons_id": target_id, "position_km": position, "velocity_km_s": velocity}
    return {
        "source": "NASA/JPL Horizons API",
        "url": API_URL,
        "epoch": f"{date.isoformat()} 00:00 TDB",
        "center": "Solar System Barycenter (500@0)",
        "frame": "Ecliptic of J2000.0 (ICRF)",
        "bodies": bodies,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Sun and planet state vectors from JPL Horizons.")
    parser.add_argument("date", nargs="?", type=dt.date.fromisoformat, help="YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("-o", "--output", type=Path, default=SNAPSHOT_FILE, help="output JSON file")
    args = parser.parse_args(argv)
    snapshot = fetch_solar_system(args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(snapshot['bodies'])} bodies for {snapshot['epoch']} to {args.output}")


if __name__ == "__main__":
    main()
