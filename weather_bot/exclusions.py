"""Tactical station-level trading exclusions for extreme-weather events.

Bias correction and the per-station σ_residual that drives sizing are
calibrated on *normal* weather. When a tropical cyclone / hurricane /
atmospheric river hits a station, both assumptions break:

  - Forecast spread explodes (σ_ensemble 3-5× typical)
  - The trained bias number no longer applies (regime is different)
  - Markets get noisy and wide
  - Retail front-runs on satellite imagery, not ECMWF

Rather than try to trade through that, we exclude the affected stations
until the event passes. Two mechanisms (used together):

  A. **Manual exclusions** (this file) — hand-edited
     `data/excluded_stations.json` list. Use when you see an event coming.

  B. **Automatic σ-anomaly check** (in safety.py) — bot detects regime
     shift from its own data when `sigma_ensemble_c > N × sigma_residual_c`.

External warning feeds (NWS / NHC / GDACS) would be a "Option C" upgrade
to catch events the user doesn't know about; see
`project_structural_gaps_pre_live.md`.

## File format

`data/excluded_stations.json` is a JSON array of objects:

```json
[
  {
    "station_id": "RPLL",
    "target": "max",
    "expires": "2026-05-15",
    "reason": "Typhoon Mawar — ensemble spread 4.2°C vs 1.1°C typical"
  },
  {
    "station_id": "KMIA",
    "target": "*",
    "expires": "2026-05-12",
    "reason": "Atmospheric river — skip both max and min"
  }
]
```

Fields:
  - `station_id` (required): ICAO code matching `weather_bot.locations`
  - `target` (required): `"max"`, `"min"`, or `"*"` (= both)
  - `expires` (required): ISO date. Entries past this are silently ignored.
  - `reason` (optional but recommended): free-text rationale

Edit by hand, or use the `add` helper described in
`project_structural_gaps_pre_live.md`. Re-runs of `load_active_exclusions`
read the file fresh each time, so updates take effect on the next cron.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

DEFAULT_EXCLUSIONS_PATH = Path("data/excluded_stations.json")


def load_active_exclusions(
    today: date,
    path: Path = DEFAULT_EXCLUSIONS_PATH,
) -> set[tuple[str, str]]:
    """Return the set of (station_id, target) pairs currently excluded.

    Missing file → empty set. Malformed file → empty set (silent — we don't
    want a JSON typo to crash the cron). Entries past their `expires` date
    are silently dropped.

    `target == "*"` in the file expands to both `(sid, "max")` and
    `(sid, "min")` in the returned set.
    """
    if not path.exists():
        return set()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if not isinstance(entries, list):
        return set()

    active: set[tuple[str, str]] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        sid = e.get("station_id")
        target = e.get("target")
        expires_str = e.get("expires")
        if not (isinstance(sid, str) and isinstance(target, str)
                and isinstance(expires_str, str)):
            continue
        try:
            expires = date.fromisoformat(expires_str)
        except ValueError:
            continue
        if today > expires:
            continue
        if target == "*":
            active.add((sid, "max"))
            active.add((sid, "min"))
        elif target in ("max", "min"):
            active.add((sid, target))
        # else: silently ignore unrecognised target value
    return active
