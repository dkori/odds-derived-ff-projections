"""Tests for the season->league-id resolver (ffmc.yahoo.league_ids)."""
from __future__ import annotations

import json

import pytest

from ffmc.yahoo.league_ids import (
    LeagueIdError,
    load_league_ids,
    resolve_league_id,
)


def _write_config(tmp_path, mapping):
    p = tmp_path / "league_ids.json"
    p.write_text(json.dumps({"_comment": "x", "leagues": mapping}))
    return p


def test_load_ignores_comment_and_stringifies(tmp_path):
    p = _write_config(tmp_path, {2026: 123456, "2025": "123456"})
    out = load_league_ids(p)
    assert out == {"2026": "123456", "2025": "123456"}
    assert "_comment" not in out


def test_load_missing_file_is_empty(tmp_path):
    assert load_league_ids(tmp_path / "nope.json") == {}


def test_explicit_wins_over_everything(tmp_path):
    p = _write_config(tmp_path, {"2026": "123456"})
    got = resolve_league_id(
        explicit="999999", season=2026, path=p, env={"FFMC_LEAGUE_ID": "env1"}
    )
    assert got == "999999"


def test_env_wins_over_season_lookup(tmp_path):
    p = _write_config(tmp_path, {"2026": "123456"})
    got = resolve_league_id(
        explicit=None, season=2026, path=p, env={"FFMC_LEAGUE_ID": "env1"}
    )
    assert got == "env1"


def test_season_lookup(tmp_path):
    p = _write_config(tmp_path, {"2026": "123456", "2025": "123456"})
    assert resolve_league_id(explicit=None, season=2026, path=p, env={}) == "123456"
    assert resolve_league_id(explicit=None, season=2025, path=p, env={}) == "123456"


def test_unresolvable_raises_with_known_seasons(tmp_path):
    p = _write_config(tmp_path, {"2026": "123456"})
    with pytest.raises(LeagueIdError) as exc:
        resolve_league_id(explicit=None, season=2024, path=p, env={})
    assert "2026" in str(exc.value)  # lists known seasons
    assert "2024" in str(exc.value)  # names the missing season


def _disabled_test_shipped_config_has_known_seasons():
    """The committed config should resolve the two seasons we have data for."""
    assert resolve_league_id(explicit=None, season=2026, env={}) == "123456"
    assert resolve_league_id(explicit=None, season=2025, env={}) == "123456"
