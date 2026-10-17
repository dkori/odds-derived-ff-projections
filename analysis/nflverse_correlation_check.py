"""Empirical correlation structure for fantasy-relevant receiving/rushing stats,
by player ROLE, from real nflverse weekly data (public, no API key).

Questions the model hinges on:
  1. Within a player, how tightly do the stat pairs move together?
       - receptions <-> receiving yards
       - receptions <-> (receiving) TDs
       - receiving yards <-> TDs
       - rushing yards <-> receiving yards   (dual-threat RBs only)
  2. Are those correlations roughly uniform across players, or do they vary by
     role? Roles: perimeter WR, slot WR, TE, pass-catching RB (proxied from
     aDOT + position + rushing volume, since nflverse has no alignment label).

TDs are near-binary per game, so we use point-biserial-ish Spearman and report
how noisy they are honestly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

SEASONS = [2019, 2020, 2021, 2022, 2023]
URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "player_stats/player_stats_{yr}.parquet")
MIN_GAMES = 32          # per player, for stable correlations
TOP_N_PER_ROLE = 15     # receivers per role bucket


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(URL.format(yr=y)) for y in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(["WR", "TE", "RB"])].copy()
    for c in ["receptions", "receiving_yards", "receiving_tds", "targets",
              "receiving_air_yards", "carries", "rushing_yards"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def assign_role(agg: pd.DataFrame) -> pd.Series:
    """Classify each player into a role from aggregate usage."""
    adot = agg["air"] / agg["targets"].clip(lower=1)
    carries_pg = agg["carries"] / agg["games"].clip(lower=1)
    role = pd.Series("other", index=agg.index)
    is_rb = agg["position"] == "RB"
    is_te = agg["position"] == "TE"
    is_wr = agg["position"] == "WR"
    role[is_rb & (agg["rec"] >= 60)] = "pass-catching RB"
    role[is_te] = "TE"
    role[is_wr & (adot < 9.5)] = "slot WR"
    role[is_wr & (adot >= 9.5)] = "perimeter WR"
    return role


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def main() -> None:
    df = load()
    agg = (df.groupby(["player_display_name", "position"])
             .agg(games=("week", "count"),
                  rec=("receptions", "sum"),
                  rec_yds=("receiving_yards", "sum"),
                  air=("receiving_air_yards", "sum"),
                  targets=("targets", "sum"),
                  carries=("carries", "sum"))
             .reset_index())
    agg = agg[agg["games"] >= MIN_GAMES].copy()
    agg["role"] = assign_role(agg)
    agg = agg[agg["role"] != "other"]

    # Top N by receptions within each role.
    picks = (agg.sort_values("rec", ascending=False)
                .groupby("role").head(TOP_N_PER_ROLE))

    records = []
    for _, r in picks.iterrows():
        g = df[df["player_display_name"] == r["player_display_name"]]
        rec = g["receptions"].to_numpy(float)
        yds = g["receiving_yards"].to_numpy(float)
        tds = g["receiving_tds"].to_numpy(float)
        rush = g["rushing_yards"].to_numpy(float)
        ypc = r["rec_yds"] / max(r["rec"], 1)
        records.append({
            "player": r["player_display_name"],
            "role": r["role"],
            "games": int(r["games"]),
            "YPC": ypc,
            "rho_rec_yds": safe_spearman(rec, yds),
            "rho_rec_td": safe_spearman(rec, tds),
            "rho_yds_td": safe_spearman(yds, tds),
            "rho_rush_recyds": safe_spearman(rush, yds),
        })
    res = pd.DataFrame(records)

    pair_cols = ["rho_rec_yds", "rho_rec_td", "rho_yds_td", "rho_rush_recyds"]
    pair_names = {
        "rho_rec_yds": "receptions <-> rec yards",
        "rho_rec_td": "receptions <-> TDs",
        "rho_yds_td": "rec yards  <-> TDs",
        "rho_rush_recyds": "rush yds   <-> rec yards",
    }

    print(f"Seasons {SEASONS[0]}-{SEASONS[-1]}  |  {len(res)} players, "
          f">= {MIN_GAMES} games each\n")

    # Per-role summary for each pair.
    for role, grp in res.groupby("role"):
        print(f"### {role}  (n={len(grp)})")
        for c in pair_cols:
            vals = grp[c].dropna().to_numpy()
            if role == "pass-catching RB" or c != "rho_rush_recyds" or True:
                if len(vals) == 0:
                    continue
                print(f"    {pair_names[c]:<26} mean {vals.mean():>6.3f}  "
                      f"sd {vals.std():>5.3f}  range [{vals.min():.2f}, {vals.max():.2f}]")
        print()

    print("### ALL players pooled")
    for c in pair_cols:
        vals = res[c].dropna().to_numpy()
        print(f"    {pair_names[c]:<26} mean {vals.mean():>6.3f}  "
              f"sd {vals.std():>5.3f}  range [{vals.min():.2f}, {vals.max():.2f}]  "
              f"(n={len(vals)})")

    # Archetype tilt: does rho_rec_yds depend on YPC?
    tilt = safe_spearman(res["YPC"].to_numpy(), res["rho_rec_yds"].to_numpy())
    print(f"\nArchetype tilt  rank-corr(YPC, rho_rec_yds) = {tilt:.3f}  "
          f"(~0 => role-independent)")


if __name__ == "__main__":
    main()
