"""Resolve a Yahoo fantasy league id by season.

Yahoo issues a **new league id every season** even for the same league (a
renewed league keeps its name but gets a fresh numeric id). Rather than make the
caller remember which id goes with which year, we keep a season->id dictionary
in ``config/league_ids.json`` and resolve from it.

Precedence (highest first):
  1. an explicit id passed on the CLI (``--league-id``) — always wins;
  2. the ``FFMC_LEAGUE_ID`` environment variable;
  3. the season lookup in ``config/league_ids.json``.

If none resolve, ``resolve_league_id`` raises ``LeagueIdError`` with a message
telling the user to pass ``--league-id`` or add the season to the config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_LEAGUE_IDS = Path("config/league_ids.json")
# Falls back to the committed example if you have not created your own.
_EXAMPLE_LEAGUE_IDS = Path("config/league_ids.example.json")


class LeagueIdError(ValueError):
    """Raised when a league id cannot be resolved for a season."""


def load_league_ids(path: Path | str | None = DEFAULT_LEAGUE_IDS) -> dict[str, str]:
    """Load the season->league-id map (string season keys -> string ids).

    Returns an empty dict if the file is missing (resolution then relies on the
    explicit arg / env var). Ignores the leading ``_comment`` key.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    leagues = data.get("leagues", {})
    return {str(k): str(v) for k, v in leagues.items()}


def resolve_league_id(
    *,
    explicit: Optional[str] = None,
    season: Optional[int] = None,
    path: Path | str | None = DEFAULT_LEAGUE_IDS,
    env: Optional[dict] = None,
) -> str:
    """Resolve a league id by precedence: explicit > env > season lookup.

    ``explicit`` is whatever the user passed on the CLI (``None`` if omitted).
    ``season`` keys the config lookup. ``env`` defaults to ``os.environ`` and is
    injectable for tests. Raises :class:`LeagueIdError` if nothing resolves.
    """
    if explicit:
        return str(explicit)

    env = os.environ if env is None else env
    env_id = env.get("FFMC_LEAGUE_ID")
    if env_id:
        return str(env_id)

    mapping = load_league_ids(path)
    if season is not None and str(season) in mapping:
        return mapping[str(season)]

    known = ", ".join(sorted(mapping)) or "(none configured)"
    raise LeagueIdError(
        f"No league id for season={season}. Pass --league-id, set "
        f"FFMC_LEAGUE_ID, or add the season to {path}. "
        f"Known seasons: {known}."
    )
