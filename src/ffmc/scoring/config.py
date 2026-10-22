"""League scoring configuration.

Scoring is data, not code: PPR / half-PPR / standard differ only in constants,
and eventually these can be pulled from a Yahoo league's settings. A StatLine is
one simulated outcome for a player; ScoringConfig.score() turns it into points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StatLine:
    """One player's (simulated) statistical outcome for a game."""

    receptions: float = 0.0
    receiving_yards: float = 0.0
    rushing_yards: float = 0.0
    touchdowns: float = 0.0  # rushing + receiving TDs combined (anytime-TD market)
    passing_yards: float = 0.0
    passing_tds: float = 0.0  # scored separately from rushing/receiving TDs (QBs)


@dataclass(frozen=True)
class ScoringConfig:
    """Points awarded per unit of each stat.

    Defaults are full-PPR standard values. Use the factory helpers for the
    common presets.
    """

    points_per_reception: float = 1.0
    points_per_receiving_yard: float = 0.1  # 1 pt / 10 yds
    points_per_rushing_yard: float = 0.1
    points_per_touchdown: float = 6.0  # rushing/receiving (anytime-TD) count
    # QB passing. Defaults match the common 1 pt / 25 pass yds + 4 pt / pass TD.
    # Rushing yards/TDs for QBs reuse the fields above (position-independent).
    points_per_passing_yard: float = 0.04  # 1 pt / 25 yds
    points_per_passing_td: float = 4.0

    def score(self, line: StatLine) -> float:
        return (
            line.receptions * self.points_per_reception
            + line.receiving_yards * self.points_per_receiving_yard
            + line.rushing_yards * self.points_per_rushing_yard
            + line.touchdowns * self.points_per_touchdown
            + line.passing_yards * self.points_per_passing_yard
            + line.passing_tds * self.points_per_passing_td
        )


def ppr() -> ScoringConfig:
    return ScoringConfig(points_per_reception=1.0)


def half_ppr() -> ScoringConfig:
    return ScoringConfig(points_per_reception=0.5)


def standard() -> ScoringConfig:
    return ScoringConfig(points_per_reception=0.0)


PRESETS = {"ppr": ppr, "half_ppr": half_ppr, "standard": standard}


# The canonical trio the front end offers as a scoring toggle. The three differ
# ONLY in points-per-reception (0 / 0.5 / 1.0), so they can be scored from a
# single set of sampled stat lines (see ffmc.sim.montecarlo.simulate_player_multi)
# with no extra sampling and no extra Odds API cost. Order is standard -> half ->
# full so a receiver's distribution shifts monotonically rightward across them.
# The label keys match PRESETS so stores/manifests stay self-describing.
SCORING_TRIO: tuple[str, ...] = ("standard", "half_ppr", "ppr")

# The toggle's default selection (0.5 PPR, this league's scoring).
DEFAULT_TRIO_SCORING = "half_ppr"


def scoring_trio() -> dict[str, "ScoringConfig"]:
    """The canonical standard / 0.5-PPR / full-PPR configs, keyed by label."""
    return {name: from_preset(name) for name in SCORING_TRIO}


def from_preset(name: str) -> ScoringConfig:
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        raise ValueError(
            f"Unknown scoring preset '{name}'. Options: {', '.join(PRESETS)}"
        )
    return PRESETS[key]()


# --- user-editable JSON scoring file --------------------------------------
#
# Scoring is data. A JSON file lets a user (or, later, the front end) edit league
# settings without touching code. This also gives the browser front-end a single
# canonical schema to read/write. The default file is 0.5 PPR.

import json
from pathlib import Path

DEFAULT_SCORING_PATH = Path("config/scoring.json")

_FIELD_KEYS = (
    "points_per_reception",
    "points_per_receiving_yard",
    "points_per_rushing_yard",
    "points_per_touchdown",
    "points_per_passing_yard",
    "points_per_passing_td",
)


def to_dict(config: ScoringConfig) -> dict:
    return {k: getattr(config, k) for k in _FIELD_KEYS}


def from_dict(data: dict) -> ScoringConfig:
    unknown = set(data) - set(_FIELD_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown scoring keys: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(_FIELD_KEYS)}"
        )
    return ScoringConfig(**{k: float(data[k]) for k in data})


def load_scoring_file(path: Path | str) -> ScoringConfig:
    """Load a scoring config from a JSON file."""
    return from_dict(json.loads(Path(path).read_text()))


def save_scoring_file(config: ScoringConfig, path: Path | str) -> Path:
    """Write a scoring config to a JSON file (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(config), indent=2) + "\n")
    return path


def ensure_default_scoring_file(path: Path | str = DEFAULT_SCORING_PATH) -> Path:
    """Create the default (0.5 PPR) scoring file if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        save_scoring_file(half_ppr(), path)
    return path


def resolve_scoring(
    *, scoring_file: Optional[str] = None, preset: Optional[str] = None
) -> tuple[ScoringConfig, str]:
    """Resolve a ScoringConfig + a label for stores/plots.

    Precedence: an explicit ``scoring_file`` wins; else a named ``preset``; else
    the default scoring file (created at 0.5 PPR on first use). Returning the
    label keeps the projections store honest about what scoring produced it.
    """
    if scoring_file:
        return load_scoring_file(scoring_file), f"file:{scoring_file}"
    if preset:
        return from_preset(preset), preset
    path = ensure_default_scoring_file()
    return load_scoring_file(path), f"file:{path}"
