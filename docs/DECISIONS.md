# DECISIONS

Settled modeling decisions and the assumptions behind them. Read this to avoid
re-litigating questions that were already reasoned through, or silently breaking
an intentional choice. Each entry: the decision, why, and the known limitation.

## QB passing modeled with existing machinery; pass TDs are a SEPARATE count
**Added 2026-10-01.** QBs are now modeled (the `config/qb_exclusions.json`
denylist is a slate-level default filter toggled by `--include-qbs`;
`build_player_models` always builds QB models). All four QB-scoring inputs have
clean markets, confirmed live on FanDuel (Steelers@Browns, Broncos@49ers,
Rams@Eagles):
- **Passing yards** (`player_pass_yds` + `_alternate`): two-sided main line + a
  **deep Over-only alt ladder (9–11 anchors, ~100–375 yds)**. Structurally
  identical to receiving/rushing yards ⇒ `ContinuousLadderDistribution` with
  `TAIL_SHAPE_PASSING` (see the yards-tail entry).
- **Passing TDs** (`player_pass_tds` + `_alternate`): a **two-sided** main line
  (0.5 or 1.5) + an Over ladder (0.5/1.5/2.5[/3.5]) ⇒ fits the same
  NB-from-ladder model (`TouchdownCountDistribution`) via a new `_pass_td_anchors`
  parser. Because the main line is two-sided, de-vig is multiplicative with **no
  assumed-hold hack** (unlike the one-sided anytime-TD path).
- **Rushing yards / rushing TDs** reuse the existing skill-player marginals;
  rush values are **position-independent** (rush TD = 6).

**Critical: passing TDs (4 pts) and the anytime-TD rush/receive count (6 pts) are
DISTINCT scored quantities, never pooled.** For a QB the `player_anytime_td`
market prices a *rushing/receiving* TD (e.g. Jalen Hurts' goal-line runs), scored
at 6; passing TDs come from their own market and score at 4. `ScoringConfig`
gained `points_per_passing_yard` (0.04 = 1/25) and `points_per_passing_td` (4.0);
`PlayerModel` gained `passing_yards` + `passing_tds` (sampled **independently** of
the skill-stat Gaussian copula for now — a QB-specific copula, pass yds ↔ pass
TDs and rush ↔ pass, is a documented fast-follow). Verified live: Hurts E[passTD]
1.26 @4 and E[anytime/rush TD] 0.44 @6 add separately; projection mean 18.8.
*Limitation:* QB intra-stat correlations are not yet modeled (passing stats
independent of rushing in the copula); INTs are not scored (not in this league).

## De-vigging is mandatory, and multiplicative for now
Raw implied probabilities include the book's margin and don't sum to 1. We
**must** de-vig before treating them as a distribution, or every projection is
biased. We use **multiplicative** (proportional) de-vig — simplest, ignores
favorite-longshot bias. `DevigMethod` enum leaves room for Shin's/power methods.
*Limitation:* multiplicative de-vig is slightly biased vs. Shin's; acceptable v1.

## Yards continuous, receptions/TDs discrete
Yards are treated as continuous (`ContinuousLadderDistribution`); receptions and
TDs as integer counts. Sampling discrete stats as counts (not rounded
continuous draws) avoids bias.

## Receptions → negative-binomial fit (not CDF interpolation)
An earlier version interpolated the CDF linearly between posted integers (⇒
uniform PMF between them) and clamped residual tail mass at k_max — both were
arbitrary and the second contradicted its own docstring. Replaced with a
**negative-binomial fit** to the posted CDF anchors: one coherent overdispersed
count model giving a principled between-integer shape and a proper decaying tail.
*Limitation:* NB is a 2-param family fit to ≥3 anchors, so it's a least-squares
fit, not exact interpolation — small residuals if a book's anchors aren't
NB-shaped (by design; more honest than forcing an exact match).

## TDs from a de-vigged TD ladder via negative binomial (was: anytime-TD Poisson)
**Superseded 2026-09-30.** Originally we backed a Poisson rate out of the single
anytime-TD prob (`lambda = -ln(1-p)`) and sampled Poisson(lambda). Two problems
made the upper tail too fat: (1) `player_anytime_td` is posted **Yes-only** on
every probed book, so the prob was **never de-vigged** — the hold inflated
`P(≥1)` (Gibbs raw 0.767); (2) Poisson is **one-parameter**, so fixing `P(≥1)`
fixes lambda and therefore *all* higher probabilities — it structurally
over-weights multi-TD games (a de-vigged `P(≥1)≈0.72` implies Poisson `P(≥3)`
far above the market's). Boom draws >55 pts were manufactured from phantom 4–5
TD games.

**Now:** we fetch the TD *ladder* (`player_anytime_td` → `P(≥1)`;
`player_tds_over` "Over j.5" → `P(≥ j+1)`; FanDuel/BetMGM post 1.5/2.5/3.5) and
fit a **negative binomial** to the ≥2 de-vigged survival anchors
(`distributions/td_count.TouchdownCountDistribution`). NB's second (dispersion)
parameter lets it match `P(≥1)` **and** `P(≥2..k)` at once, so the multi-TD tail
is **priced by the market**, not forced by one rate. With a single anchor
(anytime-TD only) it falls back to the old Poisson shape; a hard cap (k_max=4)
guards sampled counts. Rush + receive TDs remain **pooled** into one count (the
market prices one number).

*De-vig of one-sided TD probs (the one explicit assumption):* there is no paired
opposite side (Yes-only / Over-only) and no two-sided *main* TD line to pin a
power transform to, so we deflate each raw TD implied prob by an **assumed
per-side hold, default 6%** (`odds/devig.devig_oneside_td_prob`, `p_true =
p_raw/(1+hold)`; configurable via `build_player_models(td_hold=...)`). If a
genuine two-sided `player_tds`/`player_rush_reception_tds` is present it is
de-vigged multiplicatively and preferred (no assumption). *Limitation:* the 6%
hold is a single assumed constant, not measured per book/line; the NB fit is
least-squares to ≤4 anchors; `P(≥5+)` is extrapolated (capped at 4).

## TD single-anchor Poisson fallback is validated (not replaced)
**Added 2026-10-01.** The multi-anchor TD path (above) fits an NB and is good.
We investigated whether the **single-anchor (anytime-TD-only) Poisson fallback**
also overstates multi-TD games, and whether to replace it with a
history-calibrated NB whose dispersion `α = h(μ)` is borrowed from history
(`μ = −ln(1−P(≥1))` is the one forward-observable quantity).
`analysis/td_dispersion_check.py` measured per-game pooled (rush+receive) TD
counts on nflverse 2018–2023 (n=320 scorers). Finding: **given only `P(≥1)`,
Poisson is already well-calibrated** for the multi-TD conditional rate.
- The pooled dispersion that best reproduces the empirical `P(≥2 | ≥1)` is
  `α* ≈ 0.028` (essentially Poisson), with **0%** RMSE improvement over Poisson.
- Per-player method-of-moments dispersion has **median 0.0** (only 42% of
  scorers are overdispersed at all), and `α = h(μ)` has R² ≈ 0.08.
- At representative rates the empirical and Poisson `P(≥2|≥1)` nearly coincide
  (μ=0.3: 0.145 vs 0.143; μ=0.5: 0.206 vs 0.229).

**Decision: keep the Poisson single-anchor fallback.** Wiring `α* ≈ 0.03` would
be fitting an in-sample constant with no out-of-sample benefit (overfitting) and
would add unexplained complexity. The multi-TD *overstatement* originally
observed was the **Poisson-from-the-full-ladder** bug — one rate forced to fit
`P(≥1)`, `P(≥2)`, `P(≥3)` simultaneously — which the NB-from-ladder fix already
corrected; the single-anchor fallback was not the culprit. *Limitation:*
validation is in-sample 2018–2023 and on pooled TDs; a book posting only
anytime-TD still gives a one-parameter shape (mean fixes everything), but that
shape is now shown to match reality rather than merely assumed.

## Yards upper tail: super-exponential (Weibull-shape) decay, multi-anchor rate
**Refined 2026-09-30; shape MEASURED 2026-10-01.** The upper tail (beyond the
top posted yards line) was a plain exponential whose rate came from only the
**last two** anchors — it decayed too slowly and inflated the ceiling (a
workhorse RB's rushing q99.9 ran ~330–540 yds). `ContinuousLadderDistribution`
now (a) fits the base rate by **log-linear least squares over the last
`tail_fit_anchors` (default 4)** points (robust to one noisy top line) and (b)
decays survival as a **Weibull-style power**:
`S(x) = S_n · exp(−(rate·(x−tn))**tail_shape)` with `tail_shape` (>1 ⇒
faster-than-exponential; 1.0 recovers a plain exponential).

**`tail_shape` is now measured, not guessed, and is per-stat.** The original
`tail_shape = 1.5` was a hand-picked over-correction. `analysis/yards_tail_shape_check.py`
fits the exponent to real nflverse exceedances (games above each player's ~85th
percentile), 2018–2023, separately for receiving and rushing:

| stat | measured Weibull exponent (median / mean) | GPD ξ (mean) | n |
|---|---|---|---|
| receiving | 1.12 / 1.17 (sd 0.28) | −0.57 (80% < 0) | 153 |
| rushing | 1.05 / 1.11 (sd 0.32) | −0.47 (73% < 0) | 49 |
| passing (QB) | 1.14 / 1.21 (sd 0.28) | −0.83 (96% < 0) | 28 |

Two findings drove the decision:
1. **Real exponents cluster near 1.1, not 1.5** — the tails are only mildly
   super-exponential (close to exponential). The GPD cross-check gives ξ < 0 for
   ~75–80% of players (bounded/thin tails), so a modest exponent is safe; 1.5 was
   over-thinning the ceiling.
2. **The exponent does NOT vary with observable interior dispersion.** Regressing
   it on three scale-free interior features (IQR/median, interior CV, interior
   skew proxy) gives R² ≤ 0.04 for both stats. A *conditional* shape would be
   fitting noise, so we use a **pooled constant per stat** — the honest choice.

Wired as module constants `TAIL_SHAPE_RECEIVING = 1.15`,
`TAIL_SHAPE_RUSHING = 1.10`, and `TAIL_SHAPE_PASSING = 1.15` in
`distributions/ladder.py`, passed per-stat at the `ContinuousLadderDistribution`
construction sites in `odds/markets.py`. The same per-stat constant feeds the
single-anchor fallback (below). **Passing (QB) note:** passing yards are the
least heavy-tailed (ξ mean −0.83, 96% bounded) and their alt ladders are deep
(9–11 anchors), so the tail is **low-stakes**: 1.15 vs the measured 1.14 moves
full QB projections by 0.00, and even 1.0 vs 1.5 moves pass-yds q99 by only
~8–16 yds (~0.3–0.6 pts) with the mean unchanged — the interior anchors pin the
distribution. *Limitation:*
still a pooled constant, not validated against realized ceilings out-of-sample;
q99 near a very high posted line is set by the market data, not the
extrapolation. The idea of pinning the tail by "matching the implied mean" was
considered and rejected: one line gives a quantile not a mean, the median is in
the (already-pinned) interior, and for a right-skewed stat the mean lives partly
in the unobserved tail — so no posted anchor constrains the tail shape, which is
why it must be imported from history.

## Alternate lines are Over-only; we accept one-sided anchors
The `_alternate` prop markets (`player_reception_yds_alternate`,
`player_rush_yds_alternate`, `player_receptions_alternate`) are posted
**Over-only** — each alt line has an Over price but no paired Under. These alt
lines are the whole distribution: a book typically posts 10–15 of them per
player, and each Over price is a direct `P(X > point)` anchor. An earlier
`_ladder_for` required *both* Over and Under to form an anchor, which silently
dropped every alt line and left each player with a **single** anchor (just the
two-sided main line). With one anchor there is no distributional information —
only a median — and the fabricated exponential tail dominated the mean and
ceiling. **Fix:** accept Over-only lines as `american_to_implied_prob(price)`
anchors; keep two-sided de-vig for the main line and prefer it at a shared
threshold. This is the difference between the product working and not working.
*Limitation:* see the vig entry below.

## Over-only alt de-vig: power-calibrate the ladder to the two-sided main line
An Over-only alt price includes the book's margin and has no paired Under to
remove it. But the **main line** usually *does* have both sides, so it can be
cleanly de-vigged, giving one measured, trustworthy `P(X > main_line)`. We use
that single clean anchor to calibrate the whole one-sided ladder via a **power
transform**: pick `pi` so `p_raw(main) ** pi == p_true(main)`, then map every
alt anchor `p -> p ** pi`. The power form deflates **longshots (deep-tail lines)
more than favorites**, the correct direction for favorite-longshot bias (a
constant divisor applies one factor to all and can't fix the shape). The
main-line threshold is then pinned to its exact de-vigged value. If a player has
no two-sided main line, the raw Over probs are used unchanged (no basis to
de-vig a purely one-sided market); the transform is monotone so ladder ordering
is always preserved. Implemented in `odds/devig.calibrate_oneside_ladder` /
`oneside_power_exponent`, applied in `odds/markets._ladder_for`.
*Effect (DK Metcalf rec-yds):* E[X] ~52.9 (raw) -> ~49.9 (calibrated), pulling
the implied mean toward the 43.5 main line; the four sampled players each moved
down ~0.3-0.4 pts at 0.5 PPR. *Limitation:* one clean anchor pins a
one-parameter (power) correction, not the entire vig curve; it assumes the
Over-side hold structure the main line reveals extends across the ladder --
weakest in the deep tail, where the power form still degrades gracefully. A full
Shin's-method treatment remains a future option.

## Tail extrapolation is a real assumption
Books post lines over a limited range; outcomes beyond require extrapolating the
CDF tail. Yards use an exponential survival tail (rate from the last posted
segment). With a full alt ladder there are many interior anchors, so the tail
governs only the extreme upper end (beyond the highest posted alt, ~150 yds) and
its influence is small. *Limitation:* genuinely unobserved; matters for ceiling
games. Not validated against realized outcomes. **NB:** this assumption is only
benign because we now keep the full alt ladder — with a single anchor the tail
dominates the distribution (see the single-anchor entry below).

## Single-anchor (main-line-only) continuous stats are conservative + flagged
**Added 2026-09-30.** A continuous stat (yards) posted with only its **main
line** and no alternates — common **early in the week** before alt ladders open —
gives us exactly ONE point `P(X > L)`. One anchor can't determine a shape. The
original fallback used an exponential with `rate = 1/L`, which decays far too
slowly and **fabricated absurd ceilings**: a 47.5-yd receiving line implied a
~315-yd 99th percentile and a bimodal-looking "boom" profile that was pure
artifact (observed on DJ Moore / Dalton Kincaid in the early Week-4 FanDuel pull
— their whole game, Patriots @ Bills, had posted only main lines).

**Fix (two layers):**
1. *Conservative shape.* The single-anchor fallback now models the stat as
   `S(x) = exp(−(rate·(x−floor))**tail_shape)` with `rate` pinned so `S(L)=1−F`
   (see `ContinuousLadderDistribution._single_anchor_rate`) and the same
   per-stat `tail_shape` (measured ~1.15 receiving / ~1.10 rushing; see the
   yards-tail entry above) used for the multi-anchor upper tail. This honours
   the one posted probability exactly while decaying much faster than the old
   `rate = 1/L` exponential — DJ Moore's rec-yds q99 fell ~315 → ~168, median
   stays at the line.
2. *Honesty.* The distribution exposes `n_anchors` / `is_single_anchor`; the
   slate records per-stat `anchor_counts`, `single_anchor_stats`, and a
   `low_confidence` flag in `data_provenance`; the playground shows a "⚠ low
   confidence — main line only (assumed shape)" badge. So a thin-data projection
   is visibly marked, not presented as measured.

*Limitation:* a single anchor is still barely more than a point estimate — the
shape remains an assumption. The real remedy is to **re-pull closer to kickoff**
when alt ladders fill in; the flag exists so users don't trust these in the
meantime. We deliberately do NOT drop single-anchor players (unlike QBs/TD-only)
because a main line is real, usable information — we just mark its uncertainty.

## Scoring is injected config, and lives in an editable JSON file
Scoring is data, not code. Presets (`ppr`/`half_ppr`/`standard`) remain, but the
canonical source is a user-editable JSON file (`config/scoring.json`, keys:
`points_per_reception`, `points_per_receiving_yard`, `points_per_rushing_yard`,
`points_per_touchdown`), initialized to **0.5 PPR / 0.1 per yard / 6 pt TD**.
CLI precedence: `--scoring-file` > `--scoring` preset > the default file (auto-
created on first use). The projections store records the scoring label (e.g.
`file:config/scoring.json`) so a store is honest about what produced it. This is
also the schema the browser front end will read/write, so scoring changes need
no code edits. *Limitation:* v1 covers only the four modeled stats; no bonuses
(e.g. 100-yard games, per-first-down) yet.

## Independence is the DEFAULT; correlation is opt-in
`simulate_player(correlated=False)` by default — the only model with zero
fabricated joint structure. `--correlated` opts into the Gaussian copula. This
is deliberate: nothing changes silently, and independence's bias (thin tails) is
documented rather than "fixed" with numbers the odds don't contain. See
[CORRELATIONS.md](CORRELATIONS.md) for the whole story.

## Correlation model is position-agnostic (hard requirement)
No position labels or role buckets anywhere at runtime. Correlations are
continuous functions of odds-derived usage (`rush_share`, YPC). This was an
explicit user requirement and an earlier position-gated version was removed.
**Do not reintroduce position/role branching.** Details in
[CORRELATIONS.md](CORRELATIONS.md).

## Marginals are sacred
Whatever we do for correlation, each stat's own odds-implied distribution must
stay exactly unchanged. The copula only alters co-movement (it drives the
existing inverse-CDFs with correlated uniforms). Any change that shifts a
marginal is a bug.

## Monte Carlo (not analytic convolution) — because of the copula
For the *independent* case, the total-score distribution could be computed
exactly by FFT-convolving the per-stat PDFs (faster, noise-free). We considered
this. We kept Monte Carlo because the **correlated** case has no simple
closed form (sum of copula-coupled non-Gaussian variables), and we didn't want
two engines. If independent-mode speed/exactness ever matters, an analytic path
could be added just for that case (and would double as a test oracle).

## Single bookmaker
We read one book's lines (`--book`), not a consensus across books. *Limitation:*
a consensus/fair-line blend would be more robust. Not done.

## Market efficiency assumption
The whole approach assumes the book's lines are a good estimate of true
probabilities. Defensible prior, but projections inherit any market bias.

## The Odds API integration is VERIFIED (2026-09-29)
The market keys, endpoint, and response shape in `odds/` were confirmed against
the live endpoint with a real key. A one-game, one-book NFL fetch returned all 7
markets with the exact shape the parser expects and simulated end-to-end. What
was confirmed:
- **Endpoint:** player props live only on `/v4/sports/{sport}/events/{eventId}/odds`
  (one game per call). There is **no per-player query** — a call returns every
  player in the requested markets for that game.
- **Market keys (all correct):** `player_receptions(_alternate)`,
  `player_reception_yds(_alternate)`, `player_rush_yds(_alternate)`,
  `player_anytime_td`. Anytime-TD outcomes are `Yes`/`No` with `point: null`
  (parser defaults the point to 0.5).
- **Outcome shape:** `name` (Over/Under or Yes/No), `description` (player),
  `point` (line), `price` (American odds). Matches `_collect_over_under`.
- **Credit cost:** `cost = [unique markets returned] × [regions]`, billed per
  game. The primary usage markets plus the TD ladder
  (`player_anytime_td` + `player_tds_over`, and optional two-sided
  `player_tds`/`player_rush_reception_tds`) → **~8 credits per game** at one book
  (books return only the subset they price, not the full request list).
  `/events` is free. A full ~16-game slate at one book ≈ **~112–130 credits per
  refresh**; free tier is 500/mo (≈3–4 full-slate refreshes/mo, or upgrade for
  daily). *(Updated 2026-09-30: was 7 markets/7 credits before the TD ladder was
  added for the fat-tail fix.)*

## Quarterbacks are excluded via a maintained denylist (not an odds rule)
v1 models only receptions / receiving-yards / rushing-yards / TDs from
skill-player prop markets; it does **not** fetch or simulate passing yards, pass
TDs, or interceptions. A QB's fantasy score is dominated by passing, so their
projection here would be just a rushing+TD *floor* — structurally wrong and
misleading. We therefore **drop QBs from the store**.

*Why a denylist and not an odds-derived filter:* the Odds API returns **no
position field**, and a QB is **market-indistinguishable** from a run-only RB —
both present as "rushing line + anytime-TD, no receiving" (verified: Josh Allen
sits amid Saquon/Kyren/Jeanty in the rush-only group with no separating feature).
Any structural rule ("rushing + TD, no receptions ⇒ drop") would wrongly delete
legitimate goal-line/early-down backs. So exclusion is a **maintained list**
(`config/qb_exclusions.json`), matched **case- and punctuation-insensitively**
(`sim/slate._normalize_name`, so "C.J. Stroud" == "CJ Stroud"). Applied in
`precompute_slate(exclude_qbs=True)` (default; `ffmc slate --include-qbs` keeps
them). Unknown names are ignored; the per-game report lists who was cut.
*Limitation:* the list needs updating as depth charts change, and it doesn't
generalize to unlisted names. This is a deliberate stopgap — **QBs will get their
own simulation methodology** (passing markets + pass-stat correlations) in a
future version rather than being excluded. See the roadmap.

## Demo chart set: "at least" framing, and why the density plot was demoted
*(decided 2026-10-03, through iterative use of the live demo)*

The static demo converged on **three** charts after starting with four. The
reasoning is worth keeping because it's about *decision usefulness*, not taste:

- **The density (PDF) view is the best picture of the distributions and the worst
  tool for a lineup call.** It shows divergence vividly, but the thing a user
  wants to read off it — "where do these two cross, and what does that mean for
  who I start?" — **isn't interpretable** from crossing density curves. It was
  kept (it's genuinely informative about shape) but moved **last** with an
  explicit caveat.
- **"At least" beats "at most."** A fantasy manager reasons in thresholds: "what's
  the chance they clear the ~12 I need?" That is a **survival** function
  `P(score ≥ x)`, not the textbook CDF `P(score ≤ x)`. Both encode identical
  information, but the complement matches the question, so the primary chart
  plots `y = 1 − level` (downward-sloping).
- **One framing of the survival curve was enough.** We briefly shipped *both*
  orientations — `P(≥x)` vs points, and its inverse (points at a given
  confidence). The inverse answered the same question less directly, so it was
  dropped and **median lines were added to the survival chart** instead, which is
  what the inverse was mostly being used to read.
- **Per-player overlay lines don't scale.** Drawing our-mean and Yahoo's
  projection as vertical lines meant up to **four** labeled lines inside a narrow
  points range — unreadable for similar players. Those numbers moved to the
  player **chips**, where `mean` and `median` are shown together: mean for
  maximizing expected score, median for "I just need them to hit their
  projection."
- **Diff chart must stay readable under domination.** When one player strictly
  dominates, the difference curve never crosses zero, so a naive auto-range hides
  the losing side entirely. The y-range is forced **symmetric** and the line/fill
  is **split into each player's own color** by side, with in-panel labels, so the
  chart is self-explanatory either way.
- **Interaction:** zoom/pan is **disabled** (it fired by accident, especially on
  touch) while hover is **kept** — the hover readout is the actual value-delivery
  mechanism here, the zoom was never useful on a fixed 0–100 percentile axis.

## TD-only entries are noise to be filtered downstream
Because `player_anytime_td` lists *everyone* who could score (linemen, QBs,
D/ST), a slate parse yields non-skill "players" with only a TD marginal. These
are not v1 targets. The precompute filters to players with **sufficient data** —
at least one of receptions / receiving-yards / rushing-yards lines — so a
TD-only entry is dropped. This is a display/quality filter, not a cost lever
(players come bundled per game regardless).

## Backend architecture: batch precompute, front-end reads a static store
The product is a browser front-end, but odds are **not** pulled at request time.
A weekly batch job (`slate` precompute) pulls every game's props once, simulates
all qualifying players, and writes a combined projections store. The front end
(browser plugin) only reads that store and **fuzzy-matches player names** it
scrapes from the page (Yahoo fantasy roster, or a team page) against the store.
Rationale: protects the API quota (one pull serves all users), keeps the UI fast
(no live API/sim latency), and decouples data collection from presentation. Two
front-end designs are under consideration (interactive compare tool; automatic
Yahoo-page overlay) — both consume the same precomputed store, so the backend is
identical for either.

**Deployment (decided 2026-10-02): static-first.** The store is small enough
(one week ≈ 180 KB gzipped) to ship as a **static JSON asset** read by a
client-side app — no server, no per-request sim, no API-key exposure (hosted on
GitHub Pages / Netlify / S3). **Hosting the store in a DB (DigitalOcean) is
therefore long-term and deferred**, not the next step: it only becomes
worthwhile for the **browser extension** (a live, queryable store behind
`DBProjectionsStore`), which is itself downstream. The static demo is the
near-term front-end deliverable (see docs/DEMO_PLAN.md).
