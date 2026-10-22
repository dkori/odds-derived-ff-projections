# odds-derived-ff-projections

The goal of this project was to improve on single-value point estimates that sites like yahoo provide to see if we can instead estimate a probability distribution of likely fantasy scores. A probability distribution gives us more information about player floors/ceilings and upsides. To keep the methodology simple, we derive these score probability distributions as **purely as possible from sportsbook
player-prop odds** and only added a few assumptions on top of those odds where necessary. 

The idea is that sportsbook lines already represent some of the best available forecasting information, and the goal of this project is not to improve on those. The idea instead is to see if we can use those to make something useful for Fantasy roster decisions. 

Note that this project **does not perform any predictive analytics**, we have not tried to fit our own model to directly predict scores, we instead treat the sports book posted odds as a source of truth. Basically we're taking odds that are already out there, and trying to convert them to fantasy scores as faithfully as possible. This project involved some analysis of historical stats to inform some model assumptions, specifically the correlations between different types of stats and the shapes of the tails of the distributions (likelihoods for extremely high stat lines). 

## High level methodology : 

1. Obtain over/under prop lines, where available, for scorable stats for every player (rushing/receiving/passing yards, receptions, touchdowns). For most stats, there will be a main line with over/under odds, which we can use to estimate and remove the sportsbook's vig, and a series of alt props that only contain "over" odds. 
2. Using these odds, we create probability distributions for each stat. Note that some of these stats are obviously correlated, such as receiving yards and receptions, so we estimate joint distributions for those (more details below).
3. We run a monte carlo simulation (randomly generating stat values based on these distributions), and calculate the fantasy scores implied by them.

## Modeling Assumptions

While this project tries to keep judgment calls to a minimum, there were some necessary assumptions along the way. The the full rationale for each is in
[docs/DECISIONS.md](docs/DECISIONS.md).

1. **Market efficiency.** The whole approach assumes the book's lines are a good
   estimate of true probabilities. Defensible, but the projections inherit any
   market bias. This is the one assumption you can't opt out of.

2. **Vig removal** Sportsbooks take a cut from every posted over/under line (the odds don't sum to 1), and we don't necessarily know where that cut is coming from. There are multiple methods for removing vigs, and this project opts for a simple
   multiplicative (proportional) de-vig. Unlike other methods, this ignores favorite-longshot bias;
   Shin's or a power method would differ. The de-vig interface is pluggable
   (`odds/devig.py`).

3. **Over-only alternate lines need calibration.** While we have both over and under odds for the main props, which make de-vigging more straightforward, Alt prop ladders are posted Over-only (no paired Under), so they can't be paired-de-vigged directly. To get around this, after we de-vig the two-sided *main* line for one clean anchor, which gives us a solid estimate of the total probability of outcomes on either side of that line, we fit a power transform to calibrate the rest of the ladder to it.

4. **Tail extrapolation beyond the posted lines.** It is reasonably straightforward to estimate the probability between two posted alt lines, but we have to make some assumptions about what the probabilities look like beyond the highest posted line. The highest posted line for a player might be over 200 yards at a very small probability, but there's some non-zero theoretical probability of them getting a million yards. Outcomes past the highest/lowest posted line require extrapolating the tail of the CDF. To make a data driven estimate here, we pulled historical nfl data to understand the true shape of extreme outlier stat lines, and used them to estimate a constant for a super-exponential tail We use a super-exponential (Weibull) tail (`analysis/
   yards_tail_shape_check.py`). 

5. **Count distributions for receptions and TDs.** Receptions are fit with a
   negative binomial (a single coherent overdispersed count model). Touchdowns
   are fit with a negative binomial to the de-vigged TD ladder
   (`player_anytime_td` + `player_tds_over`); with only the anytime-TD anchor we
   fall back to Poisson, which `analysis/td_dispersion_check.py` confirms is
   well-calibrated in that case. Rush and receive TDs are pooled into one count
   because the anytime-TD market prices exactly one number.

6. **Single bookmaker.** The engine reads one book's lines, currently FanDuel, not a consensus
   across books. A consensus/fair-line blend would be more robust.

7. **Correlations between stats** The package was originally built to treat all stats as independent, but in practice things like receiving yards and receptions are obviously correlated. We looked at historical player stat lines to understand the correlation between different stats. We found receiving yards and receptions are generally uniformly correlated across all players (some players have more or fewer yards per reception, but for almost all players the correlation between catches and receiving yards is roughly the same, around 0.80). For other stats, we found the correlations were much less constant, so we estimate a player-specific correlation based on that player's expected stats as implied by their odds for that game — mainly how their expected yards split between rushing and receiving. For example, if a player has a high expected number of touchdowns relative to their expected total yardage, we assume their touchdowns and yardage are more tightly correlated in the simulation. If players have low TD odds relative to their yardage odds, we assume a weaker correlation between the two, see [docs/CORRELATIONS.md](docs/CORRELATIONS.md).

Note that the entire model is position agnostic with **no position label and no WR/RB branch anywhere**. We only join in positions from Yahoo after running the simulations.

## What it models

Stats: **receptions, receiving yards, rushing yards, touchdowns**, plus **QB
passing yards and passing TDs**. (No kickers; defense/DST is not modeled.)

- **Yards** (receiving / rushing / passing) — continuous.
- **Receptions** — discrete counts (negative binomial).
- **Touchdowns** — discrete counts from the de-vigged anytime-TD (+ multi-TD)
  ladder. QB passing TDs are a **separate** count, scored separately
  (e.g. 4 pts vs 6 for a rush/receive TD).

## How it works

1. **Fetch odds** for a game from [The Odds API](https://the-odds-api.com)
   (free tier, alternate-line markets). Responses are cached locally so
   development doesn't burn the monthly request quota. *(Or load a committed
   sample fixture and run with no API key at all.)*
2. **De-vig** each over/under pair so the numbers are a real probability
   distribution. → `odds/devig.py`
3. **Reconstruct a distribution** per stat from the ladder of de-vigged lines,
   made sampleable via an inverse-CDF. → `distributions/`
4. **Score** each simulated stat line with a league scoring config. → `scoring/`
5. **Aggregate** thousands of simulations into a fantasy-score distribution
   (mean, median, quantiles, floor/ceiling, P(> threshold)). → `sim/`

A CLI (`ffmc`) runs all of this offline.

## Quick start (no API key needed)

```bash
pip install -e ".[dev]"

# Simulate the committed SYNTHETIC sample event — independent mode:
ffmc simulate --event data/samples/sample_event.json --book draftkings \
    --scoring ppr --sims 50000 --thresholds 10,15,20

# Same, with the measured-correlation Gaussian copula:
ffmc simulate --event data/samples/sample_event.json --book draftkings \
    --scoring ppr --sims 50000 --correlated
```

To run against live odds or scrape your own Yahoo league, see the full
end-to-end runbook in **[docs/REPRODUCE.md](docs/REPRODUCE.md)** (get your own
Odds API key; optionally use your own Yahoo cookie).

## Project layout

```
src/ffmc/
  odds/           # The Odds API client (cached), American-odds math, de-vig
  distributions/  # ladder -> CDF -> inverse-transform sampler (per stat)
  scoring/        # league scoring config + stat-line -> points
  sim/            # per-player Monte Carlo, correlation copula, slate precompute,
                  #   store reader, figure builder
  web/            # optional local Flask playground to explore distributions
  yahoo/          # optional personal-use Yahoo matchup/directory scraper
  cli/            # the `ffmc` command-line entry point
analysis/         # standalone nflverse studies that MEASURE the few constants
                  #   the engine assumes (tail shape, TD dispersion, correlations)
config/           # scoring.json (0.5 PPR default); league_ids.example.json
data/samples/     # committed SYNTHETIC fixtures for offline dev/tests
tests/            # pytest suite
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module-by-module map.

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — module-by-module reference.
- **[docs/DECISIONS.md](docs/DECISIONS.md)** — every modeling decision and the
  assumption behind it (the long-form version of the ledger above).
- **[docs/CORRELATIONS.md](docs/CORRELATIONS.md)** — the opt-in correlation model
  in depth: what the odds can and cannot tell us, the position-agnostic design,
  the fitted coefficients, and the nflverse analyses that produced them.
- **[docs/REPRODUCE.md](docs/REPRODUCE.md)** — end-to-end runbook to reproduce
  the work with your own Odds API key and (optionally) your own Yahoo cookie.

## A note on scope and honesty

This is a faithful translator of sportsbook odds, not a betting system and not a
"beat the market" projector. The correlation coefficients are the only values
fit to history, and their predictive power is modest; treat per-player
correlations as better-than-uniform estimates, not precise truths. The engine
inherits whatever the market gets wrong.

## License

[AGPL-3.0](LICENSE). You may use, study, share, and modify this freely,
including commercially; if you distribute it or run a modified version as a
network service, you must make your source available under the same license.
Not affiliated with, endorsed by, or connected to The Odds API, Yahoo, or any
sportsbook. Respect The Odds API's terms (don't redistribute raw responses) and
Yahoo's Terms of Service when using the optional scraper with your own account.
