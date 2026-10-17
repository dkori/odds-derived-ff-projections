# ARCHITECTURE

Module-by-module reference for `src/ffmc/`. Read [../README.md](../README.md)
first for the big picture and the design thesis (minimal assumptions: we invert
the market's own prices and do no predictive modeling on top of them). For the
correlation model specifically, see [CORRELATIONS.md](CORRELATIONS.md); for the
modeling decisions and their assumptions, see [DECISIONS.md](DECISIONS.md).

Data flows: **odds → distributions → sim → score**, with `scoring` and `cli`
supporting. Design principles: each stat's marginal distribution is independent
of the odds source; scoring is injected config; the odds source is swappable;
correlation is layered on top of the marginals without altering them.

---

## `odds/` — odds ingestion and de-vigging

### `odds/devig.py`
Pure, dependency-light functions (no numpy needed).
- `american_to_implied_prob(odds)` — American odds → raw (vig-inclusive)
  implied probability. Also `decimal_to_implied_prob`.
- `devig_pair(over_odds, under_odds, method)` → `DevigResult(over, under, overround)`.
  The two raw implied probs sum to `overround` (> 1, that's the vig).
  **Multiplicative** de-vig rescales each so they sum to 1, preserving ratio.
  `DevigMethod` is an enum with room for Shin's/power methods later
  (`MULTIPLICATIVE` is the only paired-market one implemented).
- `devig_prob_from_pair(...)` — convenience returning just the de-vigged P(over).
- **One-sided (Over-only) ladder calibration** — alt lines are posted Over-only,
  so they can't be paired-de-vigged. `oneside_power_exponent(p_raw_main,
  p_true_main)` finds `pi` s.t. `p_raw_main**pi == p_true_main`;
  `calibrate_oneside_ladder(raw_probs, p_raw_main=, p_true_main=)` applies that
  power map to a whole ladder of raw Over probs, pinned by the de-vigged main
  line. Deflates longshots (tail) more than favorites (favorite-longshot bias).
  Without a two-sided anchor it returns the raw probs unchanged. See
  DECISIONS.md ("Over-only alt de-vig").

### `odds/markets.py`
Turns a raw Odds API event payload into `PlayerModel`s.
- Market-key constants: `MARKET_RECEPTIONS`, `MARKET_RECEIVING_YDS`,
  `MARKET_RUSHING_YDS` (each includes the `_alternate` variant),
  `MARKET_ANYTIME_TD`, plus the QB passing markets; `ALL_MARKETS` is their
  union. Centralized here so a rename is one line.
- `build_player_models(payload, book_key)` → `{player_name: PlayerModel}`. For
  each stat, `_ladder_for` collects lines per threshold and builds a P(X > t)
  ladder; receptions → `DiscreteLadderDistribution`, yards →
  `ContinuousLadderDistribution`, TDs → the TD-count distribution.
- **Over-only alt handling (critical):** alternate markets are posted
  **Over-only** (no paired Under). `_ladder_for` accepts each Over price as a
  direct raw P(X > point) anchor (via `american_to_implied_prob`) — this is what
  gives 10–15 anchors/player. It also finds the two-sided **main line** (the
  threshold nearest P=0.5 that has both sides), de-vigs it, and **calibrates the
  whole one-sided ladder to it** with `calibrate_oneside_ladder` (power
  transform), pinning the main-line threshold to its exact de-vigged value.
  Requiring both sides (an earlier version) dropped every alt line and left a
  single anchor — the bug that made projections meaningless; see DECISIONS.md.
- Anytime-TD is a single P(≥1 TD) Yes/No anchor; on books that post them,
  `player_tds_over` "Over 1.5/2.5/3.5" extend it into a multi-TD ladder.

### `odds/client.py`
`OddsClient` — The Odds API v4 wrapper with on-disk caching.
- Caches every response to `data/cache/<hash>.json`; a live HTTP call happens
  only on a cache miss (or `force_refresh`). Protects the free-tier quota and
  lets all dev run offline.
- `list_events(sport)` — upcoming games (free, no props, 0 credits).
- `event_player_props(event_id, markets, ...)` — player props live on the
  **event-odds endpoint**, not the bulk odds endpoint. Optional `bookmakers`
  filter. Reads `ODDS_API_KEY` from env. Prints `x-requests-remaining`.
- Cost is `[unique markets returned] × [regions]` per game. Empty responses
  cost 0.

---

## `distributions/` — ladder → sampleable distribution

Each posted line, after de-vig, gives a point on the survival function
`S(t) = P(X > t)`, i.e. a CDF point `F(t) = 1 - S(t)`. With several alternate
lines we get several CDF points and reconstruct a full, sampleable distribution.
All classes expose `sample(n, rng)`, `invert(u)` (inverse-CDF from supplied
uniforms — this is what lets the copula drive them), and `expected_value()`.

### `distributions/ladder.py`
- `LadderPoint(threshold, prob_over)` — one de-vigged posted line.
- `ContinuousLadderDistribution` (yards): interior CDF via **monotone PCHIP**
  interpolation (shape-preserving, keeps density ≥ 0); **super-exponential
  (Weibull) survival tails** beyond the posted range, with per-stat shape
  constants measured from nflverse exceedances (`TAIL_SHAPE_RECEIVING=1.15`,
  `TAIL_SHAPE_RUSHING=1.10`, `TAIL_SHAPE_PASSING=1.15`); values floored at 0.
  `expected_value()` integrates the survival function. Inverse-transform
  sampling on a dense CDF grid.
- `DiscreteLadderDistribution` (receptions): fits a **negative-binomial** to the
  posted CDF anchors (a half-line at 4.5 pins P(X ≤ 4)). NB is chosen because it
  is a single coherent overdispersed count model — giving a principled PMF
  *between* integers (no arbitrary uniform fill) and a proper decaying upper
  tail (no residual mass clamped at k_max). Fit is least-squares on (mu, alpha);
  degrades to near-Poisson with a single anchor.

### `distributions/td_count.py`
TD counts from the TD market ladder. Books post `player_anytime_td` = P(≥1 TD)
and (on some books) `player_tds_over` = P(≥2/3/4). These are de-vigged (anytime-TD
is Yes-only, so one-sided probs are corrected for an assumed per-side hold) and a
**negative binomial** is fit to the ladder, so the multi-TD tail is priced by the
market. See DECISIONS.md ("TDs from a de-vigged TD ladder").

### `distributions/poisson.py`
- `PoissonTouchdownDistribution(prob_anytime_td)` — the single-anchor fallback
  used when only the anytime-TD market is available. Since books post only
  P(≥1 TD), we back out `lambda = -ln(1 - p)` and sample from Poisson(lambda).
  `lambda_from_anytime_td(p)` is the standalone conversion. `analysis/
  td_dispersion_check.py` confirmed this is well-calibrated given only the
  single anchor.

---

## `scoring/` — league scoring

### `scoring/config.py`
- `StatLine(...)` — one outcome (receptions, receiving/rushing/passing yards,
  rush+receive TD count, passing TD count). NOTE: the rush+receive `touchdowns`
  are **combined** (the anytime-TD market prices one number); passing TDs are a
  **separate** count scored separately.
- `ScoringConfig(points_per_*)` with `.score(line)`. Factories `ppr()`,
  `half_ppr()`, `standard()`; `from_preset(name)` and `PRESETS`. Dataclass
  defaults are full-PPR, but see the file-based config below for the app default.
- **User-editable JSON scoring file**: `to_dict`/`from_dict`,
  `load_scoring_file`/`save_scoring_file`, `ensure_default_scoring_file`
  (creates `config/scoring.json` at **0.5 PPR** if missing), and
  `resolve_scoring(scoring_file=, preset=)` → `(ScoringConfig, label)` with
  precedence **file > preset > default-file**. Stores record the label so a
  store is honest about the scoring it was built with.

---

## `sim/` — Monte Carlo and correlation

### `sim/montecarlo.py`
- `PlayerModel(name, ...)` — a player's stat distributions; any may be `None`
  (no line offered).
- `ScoreDistribution(name, samples)` — wraps the simulated point array with
  `mean`, `median`, `std`, `quantile(q)`, `floor` (p10), `ceiling` (p90),
  `prob_over(threshold)`, `summary()`.
- `simulate_player(model, scoring, n_sims=50_000, seed=None, correlated=False)`
  — the entry point. `correlated=False` (default) samples each stat
  independently. `correlated=True` uses the Gaussian copula.
- Internals: `_simulate_independent` draws each marginal separately.
  `_simulate_correlated` builds the correlation matrix (via `_resolve_spec`,
  which computes usage features from the marginals' `expected_value()`s and
  calls `CorrelationSpec.from_features`), takes the **sub-matrix over only the
  stats the player actually has**, draws correlated normals, maps them to
  uniforms via `Phi`, and pushes each uniform through that stat's `invert(u)`.
  This leaves every marginal exactly unchanged.

### `sim/correlations.py`
The correlation model — **see [CORRELATIONS.md](CORRELATIONS.md) for the full
reasoning; do not edit without reading it.** In brief: position-agnostic,
continuous functions of odds-derived usage (chiefly `rush_share`). Key pieces:
`spearman_to_gaussian` (converts measured rank corr to the copula's Pearson
parameter), `PlayerFeatures` (usage features from marginal expectations),
`CorrelationSpec.from_features` (the single usage→correlation mapping), and
`gaussian_matrix()` (assembles + PSD-projects the matrix). The coefficients are
the one part of the pipeline *fit to history* (nflverse) rather than read from
the odds; it is opt-in (`--correlated`), modest (R²~0.15 for the useful one),
and never alters a marginal.

### `sim/slate.py`
The **precompute batch job** (not run at request time).
- `precompute_slate(...)` — lists events in a commence-time window (or explicit
  `event_ids`), fetches each game's props for one book (cached), simulates every
  **qualifying** player, and writes one **versioned snapshot** via
  `write_snapshot`. Returns a `SlateResult` (per-game coverage report).
  `force_refresh` bypasses the odds cache (spends credits); cached games cost 0.
- `write_snapshot(...)` — writes the **versioned, week-preserving layout** (never
  overwrites a prior run): `data/projections/<week>/<snapshot_id>.json`
  (immutable), `.../latest.json` (a copy of the newest), and `.../index.json`
  (a manifest). Lets you re-run a week as lines fill in and keep prior weeks.
- `has_sufficient_data(model)` — True iff the player has ≥1 of receptions /
  receiving-yards / rushing-yards (drops anytime-TD-only noise).
- Store schema per player: `mean, median, std, floor_p10, ceiling_p90`,
  `quantiles{levels[], values[]}` (a **dense, tail-dense grid of ~105 levels**),
  `histogram{counts[], edges[], n}` (**64 bins**), `markets_present, game,
  commence_time, event_id`; top level records `week_label, snapshot_id,
  generated_at, book, scoring, correlated, n_sims, games[]`.

### `sim/store.py`
`ProjectionsStore` — the **read side** and the seam for a possible future
DB-backed reader. File-backed reader over the versioned layout: `list_weeks`,
`list_snapshots`, `latest_snapshot_id`, `load(week, snapshot_id=None)` (defaults
to latest), `get_player`, `player_names`, and `search_players` (substring/prefix
typeahead). All consumers (the plotter, the Flask app) go through this so a
`DBProjectionsStore` can replace it without touching callers.

### `sim/viz.py`
Figure construction shared by the plotter and the Flask app:
`density_from_histogram(rec)` (stored histogram → x/density curve) and
`build_figure(store, series, mode=...)`. Modes:
- `"density"` (default) — overlay for ≤3 players, small multiples for 4+.
- `"cdf_std"` — the standard CDF (x=points, y=cumulative probability).
- `"diff"` — **2 players only**; x=percentile, y=points difference, oriented so
  the **higher-mean player is positive**. Public helper **`diff_series(series)`**
  returns `(levels_pct, diff, hi_name, lo_name)` and interpolates a mismatched
  quantile grid.

`percentile_at(rec, x)` inverts the stored dense quantile grid (used for hover).
Plotly is imported lazily. `tests/test_viz_charts.py` locks the math.

### `web/`
A local Flask **playground** (dev tool) for eyeballing the charts.
`create_app()` exposes `/` plus JSON routes `/api/weeks`,
`/api/snapshots/<week>`, `/api/search`, `/api/figure`; it reads **only** through
`ProjectionsStore`. Run with `pip install -e ".[web]"` then `python -m ffmc.web`.

---

## `yahoo/` (matchup-page scraper — `yahoo` extra)

Personal-use scraper for a Yahoo Fantasy league's **matchup pages** and
**players directory**, which are server-rendered and carry, per player, Yahoo's
**projected** points (and on matchup pages, the **actual** points). Yahoo retired
third-party Fantasy *API* access in mid-2026; the page is the remaining route.
You supply **your own** session cookie (`$YAHOO_COOKIE`) and **your own** league
id (`--league-id`, `$FFMC_LEAGUE_ID`, or `config/league_ids.json`). Respect
Yahoo's Terms of Service; this is a low-volume personal-use tool.

- `parser.py` — pure HTML → structured rows (no network, fully testable).
  `parse_matchup_html(html, week=...) -> MatchupPage`; `PlayerLine` carries
  `player_id, name, nfl_team, position, roster_slot, team_side, projected,
  actual, owner`. `parse_matchup_team_names(html)` returns the fantasy team
  names and degrades to `(None, None)` if the layout changes.
- `fetcher.py` — cached, cookie-authenticated fetch, season-aware snapshot
  writer, and **soft-block detection** (a 200 with no table is a login wall).
- `crawler.py` — `LeagueCrawler.crawl_week()` loops teams, dedupes players by
  `player_id`, skips the opponent's redundant page, uses a polite jittered delay
  before real network calls, bounded retry/backoff, and **halts on a block**
  rather than recording empty rosters.
- `directory.py` — the league players-page position scraper (QB/RB/WR/TE +
  Yahoo projection + NFL team), paginated.
- `league_ids.py` — resolves a season→league-id by precedence
  (`--league-id` > `$FFMC_LEAGUE_ID` > `config/league_ids.json`). Ships with
  `config/league_ids.example.json`; create your own gitignored copy.

CLI: `ffmc yahoo` (one page) and `ffmc crawl` (a league/week, `--dry-run` to
preview URLs with zero network).

---

## `cli/main.py`
Console script `ffmc` (defined in `pyproject.toml`). Subcommands:
- `simulate --event <path.json>` — simulate every player in a saved payload
  (runs on the committed synthetic samples with no API key).
- `events` — list upcoming events (needs key or cache; 0 credits).
- `fetch --event-id <id>` — fetch live props for an event and simulate.
- `slate --week-label <label>` — the precompute batch job (see `sim/slate.py`).
  `--dry-run` previews games + worst-case credit cost without spending;
  `--force-refresh` re-pulls fresh lines.
- `yahoo` / `crawl` — the Yahoo scraper (see `yahoo/`).

Common flags: `--scoring ppr|half_ppr|standard` and **`--scoring-file <path>`**
(precedence: file > preset > default `config/scoring.json` at 0.5 PPR), `--sims`,
`--seed`, `--book`, `--thresholds 10,15,20` (P(>x) columns), `--sport`,
`--correlated` (opt into the copula; default independent). There is
intentionally **no `--role` flag** — correlations are derived from the odds.

---

## `analysis/` (not shipped in the package)
Standalone scripts that pull public nflverse data to **justify the few
assumptions the engine does make** — they are how the measured constants were
obtained, included so the work is reproducible:
- `nflverse_correlation_check.py` — measures per-player rank correlations and how
  they vary with usage; produces the coefficients in `sim/correlations.py`.
- `yards_tail_shape_check.py` — fits the Weibull upper-tail exponent from real
  exceedances; produces the `TAIL_SHAPE_*` constants in `distributions/ladder.py`.
- `td_dispersion_check.py` — checks TD-count overdispersion and confirms the
  single-anchor Poisson fallback is well-calibrated.

Install with `pip install -e ".[analysis]"` (adds pandas + pyarrow). See
[CORRELATIONS.md](CORRELATIONS.md) and [DECISIONS.md](DECISIONS.md).
