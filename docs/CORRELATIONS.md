# CORRELATIONS

The correlation model is the most nuanced and most-iterated part of this
project. This doc captures **why it is the way it is**, so nobody re-litigates
settled questions or breaks an intentional design. Read it before editing
`src/ffmc/sim/correlations.py`.

## The problem

The default simulation samples each stat (receptions, receiving yards, rushing
yards, TDs) **independently**. In reality they're positively correlated (more
catches → more yards → more scoring chances). Independence **compresses the tails
of the total score** — the mean/median stay right, but the floor and ceiling
look more confident than reality. Correlated mode (opt-in, `--correlated`) fixes
this with a **Gaussian copula**.

## What a Gaussian copula does here

It couples the stats **without changing any marginal**. Per player per sim:
draw correlated standard normals `z ~ N(0, R)`, map to uniforms `u = Φ(z)`, then
push each `u` through that stat's own inverse-CDF (`invert(u)`). Because each
uniform still goes through the same marginal sampler, every stat's odds-implied
distribution is provably unchanged — only the *co-movement*, and therefore the
total-score tails, changes. (The samplers were built around inverse-transform of
a uniform precisely so this drop-in works.)

Rank (Spearman) correlations are converted to the copula's Pearson parameter via
the exact identity `rho_pearson = 2·sin(π·rho_spearman/6)` (`spearman_to_gaussian`),
so simulated *rank* correlations reproduce the measured values.

## The hard truth: odds cannot give you the correlation

A single player's single-game props price each stat's **marginal** in isolation.
They never price the **joint**. Two marginals are consistent with infinitely
many joint distributions (Sklar's theorem), so **you cannot back a correlation
out of the odds alone.** (Same-game-parlay pricing *would* encode correlation,
but The Odds API is not known to expose it, and de-vigging an SGP to isolate a
correlation is itself model-dependent. Not pursued.)

Therefore any co-movement we add is an assumption sourced from **outside** the
odds. The honest options were: (a) ship independence (zero fabricated joint
structure, but known-thin tails), (b) predict correlation from features via a
relationship **calibrated on historical data**. We do (b), calibrated on public
nflverse game logs — the odds supply the *marginals* and the *usage features*;
history supplies the *feature → correlation relationship*.

## Key empirical findings (from `analysis/nflverse_correlation_check.py`)

Measured per-player Spearman rank correlations, then studied how they vary:

- **receptions ↔ receiving yards: strong and uniform.** Mean ~0.80, sd ~0.06,
  range ~0.66–0.91 across all receiver types; no meaningful variation by
  archetype. → a single flat constant **0.80** is justified. (An early
  intuition that deep-ball players would decouple catches from yards was tested
  and **disproved** — even pure deep threats show ~0.8.)
- **receiving yards ↔ TDs: predictable from yards-per-catch (YPC).** Higher YPC
  → tighter coupling (deep players' TDs *are* their big gains). This is the
  real, useful signal for receivers.
- **receptions ↔ TDs: weak and noisy** (near-binary per game). Modeled as a
  near-flat ~0.28 with a tiny, tightly-clamped tilt so we don't overfit noise.
- **rushing yards ↔ TDs: strong (~0.38–0.40) for runners.** This was originally
  (wrongly) modeled as 0 — a real error caught mid-development. It's the
  *dominant* TD driver for a back (goal-line volume).
- **rushing yards ↔ receiving yards/receptions: mildly positive (~0.14).** A
  feared negative game-script effect did NOT materialize.

## Why the model is POSITION-AGNOSTIC (important — do not reintroduce roles)

A crucial requirement: **no position labels, no role buckets.** A possession WR
and a red-zone-only WR share a position but have different profiles; position
throws away exactly the distinguishing info. An earlier version gated on "has a
rushing line" and applied different logic to RBs vs receivers — that is a
position label in disguise and was **removed**.

The final design makes every correlation a **continuous function of odds-derived
usage**, chiefly **`rush_share`** = E[rush yds] / (E[rush yds] + E[rec yds]).
A player's profile *emerges* from their own lines: rush_share 0 (pure receiver)
→ rushing couplings vanish; rush_share ≈ 1 (pure runner) → they're strong;
dual-threats interpolate smoothly. It was **recalibrated by pooling ALL skill
players (193, WR/TE/RB/FB, 2018–2023) with NO position filter** and regressing
each correlation on `rush_share`. Position is not an input at any stage. (The
one unavoidable residual: the historical *players* have positions, but position
is never used in the estimation or the runtime model.)

## The fitted relationships (in `sim/correlations.py`)

All are Spearman rank correlations; `rush_share ∈ [0,1]`; YPC = rec yds / recs.

| Pair | Formula | Fit quality |
|---|---|---|
| receptions ↔ rec yards | `0.80` (flat) | uniform, sd 0.06 |
| rushing yds ↔ TD | `0.024 + 0.489·rush_share`, clamp [0, 0.45] | **R²=0.64** |
| rec yards ↔ TD | `0.364 − 0.265·rush_share` + YPC tilt (faded by receiving share `1−rush_share`) | R²=0.33 |
| rush yds ↔ rec yds / receptions | `0.017 + 0.161·rush_share`, clamp [0, 0.30] | R²=0.12 |
| receptions ↔ TD | `0.237 + 0.089·(TD per 100 yds)`, clamp [0.24, 0.38] | R²≈0.04 (near-flat) |

Endpoints reproduce both the WR and RB numbers we'd measured separately, but now
from one formula. Example (one code path, no position): rush_share 0 → rush↔TD
0.02, recyds↔TD 0.41; rush_share 0.5 → 0.27 / 0.20; rush_share 1.0 → 0.45 / 0.10.

## Sensitivity: does correlation even matter? (`analysis/correlation_sensitivity.py`)

An asymmetric screen. Result: correlation **barely moves the mean/median** (a
point projection is fine under independence) but **materially widens the tails**
(p10–p90 spread grows ~30%+ from independent to strongly correlated). So it
matters specifically for floor/ceiling/risk decisions. This is exactly why the
copula exists and why it's the tails that change. Note the screen can only rule
correlation *out* as negligible (it isn't); it can't tell you the *true* value —
that needs the historical calibration above.

## Caveats to keep stated (don't quietly drop)

- Predictive power is real but **modest** for the TD pairs (R² 0.04–0.33); strong
  only for rush↔TD (0.64). Per-player outputs are better-than-uniform estimates,
  not precise truths.
- Correlations calibrated on **top-usage players**; low-volume players may differ.
- TDs are a **single pooled count** (rush + receive) because the anytime-TD
  market prices one number. rush↔TD couples rushing yards to *total* TDs — which
  is exactly what was measured, so it's internally consistent.
- The **Gaussian copula has thin tail dependence** — it under-models "everything
  clicks at once" games. A t-copula would be better (roadmap).
- **Cross-player** correlation (a WR and his own QB; two RBs in one backfield) is
  **not** modeled at all. Everything here is within a single player.

## If you want to change the correlations

1. Re-run `analysis/nflverse_correlation_check.py` (edit its seasons/filters).
2. Update the fitted constants at the top of `sim/correlations.py`.
3. Keep it position-agnostic: express any new relationship as a function of
   odds-computable usage, never a position/role branch.
4. Keep marginals sacred: the copula must never alter a stat's own distribution.
