"""Read side of the projections store.

The slate precompute (``sim/slate.py``) writes a versioned, per-week layout:

    data/projections/<week>/<snapshot_id>.json   # immutable run
    data/projections/<week>/latest.json          # copy of newest
    data/projections/<week>/index.json           # manifest

Everything that *consumes* projections (the inspector, the plotter, the Flask
playground) goes through :class:`ProjectionsStore` rather than reading files
directly. This is the seam for the planned move to a hosted DB (DigitalOcean):
a ``DBProjectionsStore`` implementing the same methods can replace the
file-backed reader without touching any caller.

Methods are intentionally small and read-only:
    - list_weeks()                      -> ["week4", ...]
    - list_snapshots(week)              -> [ {snapshot_id, generated_at, ...}, ...]
    - load(week, snapshot_id=None)      -> full snapshot dict (default: latest)
    - get_player(week, name, snapshot_id=None) -> player record or None
    - search_players(week, query, ...)  -> fuzzy-ish name matches (typeahead)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_STORE_DIR = Path("data/projections")


import re

_WEEK_FILE_RE = re.compile(r"_week(\d+)\.json$")


def write_weeks_manifest(
    data_dir: Path | str, pattern_prefix: str
) -> dict:
    """Write a ``weeks.json`` manifest of every shipped week asset in ``data_dir``.

    The static apps serve prior weeks by lazy-loading a different week's JSON on
    demand; this manifest is how the front end discovers which weeks exist and
    which is latest, without hardcoding filenames. Each week is its own file (one
    loads at a time), so listing prior weeks here adds nothing to the default
    page load.

    ``pattern_prefix`` is the asset basename stem, e.g. ``"projections"`` (public:
    ``projections_week7.json``) or ``"demo"`` (league: ``demo_week7.json``). Files
    are matched as ``<prefix>_week<N>.json``; ``N`` orders them and the highest is
    marked ``latest``. Returns the manifest dict (also written to
    ``<data_dir>/weeks.json``).
    """
    data_dir = Path(data_dir)
    weeks = []
    for f in sorted(data_dir.glob(f"{pattern_prefix}_week*.json")):
        m = _WEEK_FILE_RE.search(f.name)
        if not m:
            continue
        weeks.append({"week": int(m.group(1)), "file": f.name})
    weeks.sort(key=lambda w: w["week"])
    latest = weeks[-1]["file"] if weeks else None
    manifest = {"latest": latest, "weeks": weeks}
    (data_dir / "weeks.json").write_text(json.dumps(manifest) + "\n")
    return manifest


class ProjectionsStore:
    """File-backed reader over the versioned projections layout."""

    def __init__(self, store_dir: Path | str = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)

    # -- discovery ---------------------------------------------------------

    def list_weeks(self) -> list[str]:
        """Week labels that have at least one snapshot, newest-ish first."""
        if not self.store_dir.exists():
            return []
        weeks = [
            p.name
            for p in self.store_dir.iterdir()
            if p.is_dir() and (p / "index.json").exists()
        ]
        return sorted(weeks)

    def list_snapshots(self, week: str) -> list[dict]:
        """Snapshot manifest entries for a week (oldest -> newest)."""
        index_path = self.store_dir / week / "index.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text()).get("snapshots", [])

    def latest_snapshot_id(self, week: str) -> Optional[str]:
        index_path = self.store_dir / week / "index.json"
        if not index_path.exists():
            return None
        return json.loads(index_path.read_text()).get("latest")

    # -- loading -----------------------------------------------------------

    def load(self, week: str, snapshot_id: Optional[str] = None) -> dict:
        """Load a full snapshot dict; defaults to ``latest.json`` for the week."""
        week_dir = self.store_dir / week
        if snapshot_id is None:
            path = week_dir / "latest.json"
        else:
            path = week_dir / f"{snapshot_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no snapshot at {path}")
        return json.loads(path.read_text())

    def load_path(self, path: Path | str) -> dict:
        """Load a snapshot from an explicit file path (back-compat convenience)."""
        return json.loads(Path(path).read_text())

    # -- player access -----------------------------------------------------

    def get_player(
        self, week: str, name: str, snapshot_id: Optional[str] = None
    ) -> Optional[dict]:
        return self.load(week, snapshot_id).get("players", {}).get(name)

    @staticmethod
    def record_group(rec: dict) -> str:
        """Position group for a stored record: ``"qb"`` or ``"skill"``.

        Prefers the ``position_group`` field written by the slate. Falls back to
        inferring from ``markets_present`` (a passing line => QB) so snapshots
        produced before the field existed still classify correctly.
        """
        g = rec.get("position_group")
        if g in ("qb", "skill"):
            return g
        mp = rec.get("markets_present", [])
        if "passing_yards" in mp or "passing_tds" in mp:
            return "qb"
        return "skill"

    def player_names(
        self,
        week: str,
        snapshot_id: Optional[str] = None,
        *,
        group: Optional[str] = None,
    ) -> list[str]:
        """Player names for a week/snapshot, optionally scoped to a group.

        ``group`` is ``"qb"``, ``"skill"``, or ``None``/``"all"`` for everyone.
        """
        players = self.load(week, snapshot_id).get("players", {})
        if group in (None, "all"):
            return sorted(players.keys())
        return sorted(
            name for name, rec in players.items() if self.record_group(rec) == group
        )

    def search_players(
        self,
        week: str,
        query: str,
        *,
        snapshot_id: Optional[str] = None,
        limit: int = 10,
        group: Optional[str] = None,
    ) -> list[str]:
        """Simple substring/prefix typeahead over player names.

        Deliberately dependency-free (case-insensitive substring, prefix-first
        ordering). ``group`` scopes the candidates to QBs or skill players (see
        :meth:`player_names`). The richer rapidfuzz-based matching in the roadmap
        can slot in behind this same signature later.
        """
        q = query.strip().lower()
        names = self.player_names(week, snapshot_id, group=group)
        if not q:
            return names[:limit]
        starts = [n for n in names if n.lower().startswith(q)]
        contains = [n for n in names if q in n.lower() and n not in starts]
        return (starts + contains)[:limit]
