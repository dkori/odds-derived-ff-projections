import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

from ffmc.odds.markets import build_player_models
from ffmc.scoring.config import ppr
from ffmc.sim.correlations import (
    CorrelationSpec,
    PlayerFeatures,
    spearman_to_gaussian,
)
from ffmc.sim.montecarlo import simulate_player


def _model():
    payload = json.loads(
        (Path(__file__).parent.parent / "data/samples/sample_event.json").read_text()
    )
    return build_player_models(payload, book_key="draftkings")["Sample WR"]


def _rb_model():
    payload = json.loads(
        (Path(__file__).parent.parent / "data/samples/sample_rb_event.json").read_text()
    )
    return build_player_models(payload, book_key="draftkings")["Sample RB"]


def test_spearman_to_gaussian_identity():
    assert abs(spearman_to_gaussian(0.0)) < 1e-12
    assert abs(spearman_to_gaussian(1.0) - 1.0) < 1e-9
    assert spearman_to_gaussian(0.8) > 0.8


def test_correlation_matrix_is_valid_psd():
    # Across the whole rush_share spectrum, the matrix must stay valid.
    for share in [0.0, 0.2, 0.5, 0.8, 1.0]:
        f = PlayerFeatures(rush_share=share, ypc=11.0, td_per_100_yds=0.8)
        m = CorrelationSpec.from_features(f).gaussian_matrix()
        assert m.shape == (4, 4)
        assert np.allclose(m, m.T)
        assert np.allclose(np.diag(m), 1.0)
        assert np.all(np.linalg.eigvalsh(m) > -1e-9)


def test_correlated_preserves_marginals():
    model = _model()
    indep = simulate_player(model, ppr(), n_sims=200_000, seed=5, correlated=False)
    corr = simulate_player(model, ppr(), n_sims=200_000, seed=5, correlated=True)
    assert abs(indep.mean - corr.mean) < 0.25


def test_correlated_widens_spread():
    model = _model()
    indep = simulate_player(model, ppr(), n_sims=200_000, seed=5, correlated=False)
    corr = simulate_player(model, ppr(), n_sims=200_000, seed=5, correlated=True)
    assert (corr.ceiling - corr.floor) > (indep.ceiling - indep.floor)


def test_realized_rank_correlation_matches_target():
    # rec<->yds should realize near 0.80 through the copula.
    model = _model()
    rng = np.random.default_rng(11)
    cov = CorrelationSpec.from_features(PlayerFeatures(ypc=13.0)).gaussian_matrix()
    sub = cov[np.ix_([0, 1], [0, 1])]
    z = rng.multivariate_normal([0, 0], sub, size=300_000)
    u = norm.cdf(z)
    rec = model.receptions.invert(u[:, 0])
    yds = model.receiving_yards.invert(u[:, 1])
    assert abs(spearmanr(rec, yds).statistic - 0.80) < 0.05


# --- position-agnostic behavior: everything driven by rush_share -------------

def test_pure_receiver_has_no_rushing_coupling():
    spec = CorrelationSpec.from_features(PlayerFeatures(rush_share=0.0, ypc=13.0))
    assert spec.rush_td == 0.024      # intercept only; ~0
    assert abs(spec.rush_recyds - 0.017) < 1e-9
    # And a receiver still has a solid receiving-yards<->TD coupling.
    assert spec.yds_td > 0.30


def test_pure_runner_couples_rushing_to_tds():
    spec = CorrelationSpec.from_features(PlayerFeatures(rush_share=1.0, ypc=8.0))
    # rush_yds<->TD approaches the measured ~0.40 for a run-dominant profile.
    assert spec.rush_td > 0.45 - 0.06
    # rec_yds<->TD has fallen off relative to a pure receiver.
    assert spec.yds_td < 0.20


def test_rush_td_is_monotonic_in_rush_share():
    shares = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = [
        CorrelationSpec.from_features(PlayerFeatures(rush_share=s)).rush_td
        for s in shares
    ]
    assert all(b >= a for a, b in zip(vals, vals[1:]))  # non-decreasing
    assert vals[-1] > vals[0] + 0.2  # meaningfully rises


def test_dual_threat_is_between_extremes():
    pure_rec = CorrelationSpec.from_features(PlayerFeatures(rush_share=0.0))
    dual = CorrelationSpec.from_features(PlayerFeatures(rush_share=0.5))
    pure_run = CorrelationSpec.from_features(PlayerFeatures(rush_share=1.0))
    assert pure_rec.rush_td < dual.rush_td < pure_run.rush_td  # smooth handoff


def test_ypc_tilts_receiver_ydstd():
    deep = CorrelationSpec.from_features(PlayerFeatures(rush_share=0.0, ypc=16.0))
    short = CorrelationSpec.from_features(PlayerFeatures(rush_share=0.0, ypc=8.0))
    assert deep.yds_td > short.yds_td
    assert deep.rec_yds == short.rec_yds == 0.80


def test_predictions_are_clamped():
    from ffmc.sim.correlations import RUSH_TD_CLAMP, RECYDS_TD_CLAMP

    crazy = CorrelationSpec.from_features(
        PlayerFeatures(rush_share=5.0, ypc=999.0)  # nonsense inputs
    )
    assert RUSH_TD_CLAMP[0] <= crazy.rush_td <= RUSH_TD_CLAMP[1]
    assert RECYDS_TD_CLAMP[0] <= crazy.yds_td <= RECYDS_TD_CLAMP[1]


def test_features_from_expectations_rush_share():
    # 70 rush yds, 30 rec yds -> rush_share 0.7
    f = PlayerFeatures.from_expectations(
        exp_receptions=3.0,
        exp_receiving_yards=30.0,
        exp_rushing_yards=70.0,
        exp_touchdowns=0.6,
    )
    assert abs(f.rush_share - 0.7) < 1e-9
    assert abs(f.ypc - 10.0) < 1e-9
    # Pure receiver: no rushing line -> rush_share 0.
    f2 = PlayerFeatures.from_expectations(4.0, 60.0, None, 0.4)
    assert f2.rush_share == 0.0


def test_expected_values_are_sane():
    model = _model()
    assert 3.0 < model.receptions.expected_value() < 7.0
    assert 45.0 < model.receiving_yards.expected_value() < 95.0
    assert 0.1 < model.touchdowns.expected_value() < 1.0


def test_rb_end_to_end_preserves_mean_and_widens():
    model = _rb_model()
    indep = simulate_player(model, ppr(), n_sims=200_000, seed=9, correlated=False)
    corr = simulate_player(model, ppr(), n_sims=200_000, seed=9, correlated=True)
    assert abs(indep.mean - corr.mean) < 0.4
    assert (corr.ceiling - corr.floor) > (indep.ceiling - indep.floor)


def test_rb_resolves_high_rush_share_and_couples_tds():
    # The sample RB (E[rush]~71, E[rec_yds]~39) should resolve to high rush_share
    # and a materially positive rush_yds<->TD copula entry.
    from ffmc.sim.montecarlo import _resolve_spec

    model = _rb_model()
    dists = {
        "receptions": model.receptions,
        "receiving_yards": model.receiving_yards,
        "rushing_yards": model.rushing_yards,
        "touchdowns": model.touchdowns,
    }
    spec = _resolve_spec(model, dists)
    assert spec.rush_td > 0.25
    assert spec.gaussian_matrix()[2, 3] > 0.25  # rush_yds<->TD entry
