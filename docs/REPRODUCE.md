# REPRODUCE

An end-to-end runbook, from "runs on my machine in 60 seconds with no account"
to "my own live odds and my own Yahoo league." Three tiers; stop wherever your
needs are met.

- **Tier 0** — no accounts, no keys. Runs on committed synthetic fixtures.
- **Tier 1** — your own free Odds API key. Live NFL props → projections store.
- **Tier 2** — your own Yahoo login cookie. Scrape your league's projections.

Everything is local and offline-first: live API responses are cached to
`data/cache/`, so you only spend a credit the first time you pull a given game.

---

## Prerequisites

- Python ≥ 3.10 (verified on 3.13).
- A clean virtual environment is recommended.

```bash
python -m venv .venv && source .venv/bin/activate
```

Install depends on which tier you want:

```bash
pip install -e ".[dev]"                     # Tier 0 (engine + tests)
pip install -e ".[dev,analysis]"            # + the nflverse calibration studies
pip install -e ".[dev,analysis,web]"        # + the local Flask playground
pip install -e ".[dev,analysis,web,yahoo]"  # + the Yahoo scraper (Tier 2)
```

Verify the install:

```bash
pytest           # expect all tests to pass
ffmc --help      # the CLI entry point
```

---

## Tier 0 — no key, synthetic data

The repo ships two **synthetic** sample events (fictional teams/players, in the
Odds API schema). Nothing here touches the network or needs a key.

```bash
# Independent mode:
ffmc simulate --event data/samples/sample_event.json --book draftkings \
    --scoring ppr --sims 50000 --thresholds 10,15,20

# With the measured-correlation Gaussian copula:
ffmc simulate --event data/samples/sample_event.json --book draftkings \
    --scoring ppr --sims 50000 --correlated

# A rushing-heavy sample:
ffmc simulate --event data/samples/sample_rb_event.json --book draftkings \
    --scoring half_ppr --sims 50000 --correlated
```

You'll get, per player: mean / median / floor (p10) / ceiling (p90) and the
P(score ≥ threshold) columns you requested with `--thresholds`.

### (Optional) reproduce the calibration studies

These pull **public nflverse data** and reproduce the few constants the engine
assumes — the honest way to check the assumptions ledger in
[DECISIONS.md](DECISIONS.md):

```bash
pip install -e ".[analysis]"
python analysis/nflverse_correlation_check.py   # the correlation coefficients
python analysis/yards_tail_shape_check.py       # the TAIL_SHAPE_* constants
python analysis/td_dispersion_check.py           # the Poisson-fallback check
```

---

## Tier 1 — your own Odds API key (live NFL props)

### 1. Get a key

Sign up for a free "Starter" key at <https://the-odds-api.com> (500 credits/mo,
includes alternate player-prop markets). Copy `.env.example` to `.env` (which is
gitignored) and fill it in:

```bash
cp .env.example .env
# edit .env so it reads:  ODDS_API_KEY=your_key_here
set -a && source .env && set +a
```

> **Cost model.** `ffmc events` is free (0 credits). Fetching a game's props
> costs `[unique markets returned] × [regions]` ≈ 7 credits/game/book. A full
> ~16-game slate is ~112–130 credits, so the free tier is ~4 full refreshes/mo.
> Every response is cached, so re-running a fetched game costs 0.

### 2. List upcoming games (free)

```bash
ffmc events
```

### 3. Fetch one game's props and simulate

```bash
ffmc fetch --event-id <id-from-events> --book draftkings --scoring ppr --correlated
```

### 4. Precompute a whole week into a versioned store

```bash
# Preview games + worst-case credit cost WITHOUT spending:
ffmc slate --week-label week1 --book draftkings \
    --commence-from <iso8601> --commence-to <iso8601> --correlated --dry-run

# Then actually run it:
ffmc slate --week-label week1 --book draftkings \
    --commence-from <iso8601> --commence-to <iso8601> --correlated
```

This writes a versioned snapshot under `data/projections/week1/`
(`<snapshot_id>.json` + `latest.json` + `index.json`). Re-run as lines fill in;
prior snapshots are preserved.

### 5. Explore the distributions in a browser (optional)

```bash
pip install -e ".[web]"
python -m ffmc.web          # serves http://127.0.0.1:5000
```

The playground reads `data/projections/` through the store, with a week/snapshot
picker, player typeahead, and density / survival charts.

---

## Tier 2 — your own Yahoo cookie (scrape your league)

> The optional Yahoo scraper is for **your own** league, with **your own**
> account, at low volume. It reads server-rendered pages (Yahoo retired
> third-party Fantasy API access in mid-2026). **Respect Yahoo's Terms of
> Service.** Do not share your cookie; treat it like a password.

### 1. Get your session cookie

Log in to Yahoo Fantasy in a browser, open developer tools → Network, and copy
the full `Cookie:` request header from any fantasysports.yahoo.com request.
Provide it via an environment variable (keeps it out of your shell history and
the repo):

```bash
export YAHOO_COOKIE='<the full cookie string>'
```

### 2. Tell it your league id

Yahoo issues a **new league id each season**. Either pass `--league-id`
explicitly, set `$FFMC_LEAGUE_ID`, or create your own league map:

```bash
cp config/league_ids.example.json config/league_ids.json
# edit config/league_ids.json with YOUR season -> league id
# (config/league_ids.json is gitignored; the .example is the only committed one)
```

### 3. Scrape

```bash
# One matchup page:
ffmc yahoo --week 1 --league-id <your-league-id>

# A whole league/week (preview URLs with zero network first):
ffmc crawl --league-id <your-league-id> --season 2026 --weeks 1 --teams 10 --dry-run
ffmc crawl --league-id <your-league-id> --season 2026 --weeks 1 --teams 10
```

Crawled rosters (with Yahoo's own projections + actuals per player) land under
`data/yahoo/<season>/` (gitignored). From here you can line Yahoo's projections
up against the odds-derived store you built in Tier 1 for your own comparison.

---

## Troubleshooting

- **`ODDS_API_KEY is not set`** on `events`/`fetch` — you skipped
  `set -a && source .env && set +a`, or you're asking for a game not in the
  cache. Tier 0 commands never need the key.
- **Yahoo returns a login wall** — the fetcher detects a 200-with-no-table and
  halts rather than recording empty rosters. Your cookie has likely expired;
  grab a fresh one.
- **A credit was spent unexpectedly** — only `fetch`, `slate` (without
  `--dry-run`), and a cache miss spend credits. Run `slate --dry-run` first to
  see the worst-case cost.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how each piece works and
[DECISIONS.md](DECISIONS.md) for why.
