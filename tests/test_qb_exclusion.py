"""Tests for the QB exclusion filter in the slate precompute.

v1 models only skill-player prop markets (no passing), so a QB's projection is
just their rushing+TD floor and is misleading. The Odds API has no position
field and QBs are market-indistinguishable from run-only RBs, so QBs are dropped
via a maintained denylist (config/qb_exclusions.json), matched case- and
punctuation-insensitively. See sim/slate.load_qb_exclusions / _normalize_name.
"""
import json

from ffmc.sim.slate import _normalize_name, load_qb_exclusions


def test_normalize_name_punctuation_and_case_insensitive():
    assert _normalize_name("C.J. Stroud") == _normalize_name("CJ Stroud")
    assert _normalize_name("C.J. Stroud") == "cj stroud"
    assert _normalize_name("Michael Penix Jr.") == "michael penix jr"
    assert _normalize_name("Aidan O'Connell") == "aidan oconnell"
    assert _normalize_name("  Josh   Allen ") == "josh allen"


def test_load_qb_exclusions_from_config():
    names = load_qb_exclusions()  # default path: config/qb_exclusions.json
    assert _normalize_name("Josh Allen") in names
    assert _normalize_name("C.J. Stroud") in names
    # A skill-position player must NOT be on the QB list.
    assert _normalize_name("Saquon Barkley") not in names


def test_load_qb_exclusions_missing_file_is_empty(tmp_path):
    assert load_qb_exclusions(tmp_path / "nope.json") == set()
    assert load_qb_exclusions(None) == set()


def test_load_qb_exclusions_custom_file(tmp_path):
    p = tmp_path / "qbs.json"
    p.write_text(json.dumps({"quarterbacks": ["Test Guy", "Another QB"]}))
    names = load_qb_exclusions(p)
    assert names == {"test guy", "another qb"}


def test_slate_excludes_qbs_keeps_rbs():
    """A rush-only QB and a rush-only RB parse identically; only the QB (on the
    denylist) is dropped."""
    from ffmc.odds.markets import build_player_models
    from ffmc.sim.slate import _normalize_name, has_sufficient_data

    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_rush_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Josh Allen", "point": 39.5, "price": -110},
                            {"name": "Under", "description": "Josh Allen", "point": 39.5, "price": -110},
                            {"name": "Over", "description": "Saquon Barkley", "point": 79.5, "price": -110},
                            {"name": "Under", "description": "Saquon Barkley", "point": 79.5, "price": -110},
                        ],
                    },
                    {
                        "key": "player_anytime_td",
                        "outcomes": [
                            {"name": "Yes", "description": "Josh Allen", "price": -140},
                            {"name": "Yes", "description": "Saquon Barkley", "price": -160},
                        ],
                    },
                ],
            }
        ]
    }
    models = build_player_models(payload, book_key="fanduel")
    qualifying = {n: m for n, m in models.items() if has_sufficient_data(m)}
    qb_names = load_qb_exclusions()

    kept = {n for n in qualifying if _normalize_name(n) not in qb_names}
    assert "Josh Allen" not in kept       # QB dropped
    assert "Saquon Barkley" in kept       # RB retained
