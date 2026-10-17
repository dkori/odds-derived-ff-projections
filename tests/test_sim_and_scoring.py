import json
from pathlib import Path

import numpy as np

from ffmc.distributions.poisson import PoissonTouchdownDistribution
from ffmc.odds.markets import build_player_models
from ffmc.scoring.config import StatLine, from_preset, half_ppr, ppr, standard
from ffmc.sim.montecarlo import PlayerModel, simulate_player


def test_scoring_presets():
    line = StatLine(receptions=5, receiving_yards=80, rushing_yards=0, touchdowns=1)
    assert abs(ppr().score(line) - (5 + 8 + 6)) < 1e-9
    assert abs(half_ppr().score(line) - (2.5 + 8 + 6)) < 1e-9
    assert abs(standard().score(line) - (0 + 8 + 6)) < 1e-9


def test_from_preset_invalid():
    try:
        from_preset("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_simulate_player_td_only_matches_expectation():
    # A player with only a Poisson TD dist, PPR (6 pts/TD).
    model = PlayerModel(name="TD Guy", touchdowns=PoissonTouchdownDistribution(0.5))
    dist = simulate_player(model, ppr(), n_sims=100_000, seed=7)
    # E[points] = 6 * lambda = 6 * ln2 ~ 4.16
    assert abs(dist.mean - 6 * np.log(2)) < 0.1


def test_end_to_end_sample_event():
    payload = json.loads(
        (Path(__file__).parent.parent / "data/samples/sample_event.json").read_text()
    )
    models = build_player_models(payload, book_key="draftkings")
    assert "Sample WR" in models
    dist = simulate_player(models["Sample WR"], ppr(), n_sims=100_000, seed=3)
    # Sanity: a WR with a ~60yd receiving line, ~4-5 catches, ~45% TD should land
    # in a plausible PPR range.
    assert 8.0 < dist.mean < 22.0
    assert dist.floor < dist.median < dist.ceiling
