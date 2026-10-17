"""Build density figures from a stored snapshot (no re-simulation).

Shared by ``scripts/plot_players.py`` and the Flask playground so both render
identically from the stored histogram. Plotly is imported lazily so importing
this module doesn't require the ``web`` extra unless a figure is actually built.

Follows docs/FRONTEND.md: <=3 players -> overlaid density; 4+ -> small multiples.
"""
from __future__ import annotations

import numpy as np

# Colorblind-friendly qualitative palette (Plotly default set).
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
]
# Fill is kept very translucent so heavily overlapping densities stay readable;
# the lines themselves are drawn fully opaque (rgb, no alpha) so each player's
# curve stays crisp regardless of how faint the fill is.
FILL_ALPHA = 0.06


def density_from_histogram(rec: dict, smooth: bool = True):
    """(x, density) from a stored histogram; integrates to ~1.

    The store keeps a 64-bin histogram rather than raw samples, so drawing it
    directly is step-like/jagged. To recover the smooth look of the old
    KDE-over-samples plot we (a) upsample onto a finer grid and (b) apply a small
    Gaussian blur to the density. This is a *display* smoothing of an already-
    finalized distribution -- it does not change any stored numbers.
    """
    counts = np.asarray(rec["histogram"]["counts"], dtype=float)
    edges = np.asarray(rec["histogram"]["edges"], dtype=float)
    n = float(rec["histogram"].get("n", counts.sum()) or 1.0)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    density = counts / (n * widths)

    # The stored histogram caps its range near q99 and folds ALL overflow mass
    # above the top edge into the final bin (see ScoreDistribution.histogram, so
    # counts sum to n). That makes the last bin a ~1% spike that is NOT a density
    # at that x -- it's P(>= top edge). Drawing it (and especially smoothing it)
    # produces a spurious uptick at the right tail. Drop that overflow bin for
    # display so the curve tapers naturally. The full tail is still available
    # exactly via the stored dense_quantiles; nothing persisted changes here.
    if counts.size >= 3:
        centers = centers[:-1]
        density = density[:-1]

    if not smooth:
        return centers, density
    return _smooth_density(centers, density)


def _smooth_density(centers: np.ndarray, density: np.ndarray):
    """Gaussian-smooth a binned density onto a finer grid (KDE-like curve)."""
    if centers.size < 3:
        return centers, density
    bin_w = float(np.mean(np.diff(centers)))
    # Fine grid for a smooth line. Do NOT pad past the last bin center: the
    # stored histogram is already capped near q99 (see montecarlo.histogram's
    # upper_quantile), so padding a full bin further would let the Gaussian blur
    # paint a visible hump from ~0 tail mass and stretch the x-axis. Keep the
    # grid within the data range.
    grid = np.linspace(centers[0], centers[-1], 400)
    # Kernel bandwidth ~1.5 bins -> gentle smoothing that preserves shape.
    sigma = 1.5 * bin_w
    # Vectorized Gaussian sum of the binned mass (mass = density * bin_w).
    mass = density * bin_w
    diff = grid[:, None] - centers[None, :]
    kern = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    smoothed = kern @ mass
    return grid, smoothed


def usage_tag(rec: dict) -> str:
    """Position-agnostic usage bucket from which stat lines a player has.

    No position label is used (consistent with the model's design); this is a
    convenience filter derived purely from the odds the book posted.
    """
    mp = set(rec.get("markets_present", []))
    has_rec = "receptions" in mp or "receiving_yards" in mp
    has_rush = "rushing_yards" in mp
    if has_rec and has_rush:
        return "dual"
    if has_rec:
        return "pass_catcher"
    if has_rush:
        return "rusher"
    return "other"


_STAT_LABELS = {
    "receptions": "receptions",
    "receiving_yards": "receiving yds",
    "rushing_yards": "rushing yds",
    "touchdowns": "touchdowns",
    "passing_yards": "passing yds",
    "passing_tds": "passing TDs",
}


def data_notes(rec: dict) -> dict:
    """Human-readable data-provenance notes for one player record.

    Returns:
        {"present": [labels], "missing": [labels],
         "td": {"available", "ladder", "n_anchors", "model"},
         "summary": "one-line string"}

    Reads the ``data_provenance`` block written by the slate. Falls back to the
    older ``markets_present``-only records (TD detail then reports as unknown),
    so it works against snapshots produced before provenance was added.
    """
    prov = rec.get("data_provenance")
    if prov is None:
        # Legacy record: infer from markets_present; TD richness is unknown.
        present = list(rec.get("markets_present", []))
        missing = [s for s in _STAT_LABELS if s not in present]
        td_present = "touchdowns" in present
        prov = {
            "present": present,
            "missing": missing,
            "td": {
                "available": td_present,
                "ladder": None,  # unknown for legacy snapshots
                "n_anchors": None,
                "model": None,
            },
        }

    present_labels = [_STAT_LABELS.get(s, s) for s in prov.get("present", [])]

    # Group the player (QB vs skill) from the record so the "missing" list only
    # mentions stats that are relevant to that group: a receiver isn't "missing"
    # passing yards, and a QB isn't "missing" receptions.
    from ffmc.sim.store import ProjectionsStore

    group = ProjectionsStore.record_group(rec)
    if group == "qb":
        relevant_missing = {"rushing_yards", "passing_yards"}
    else:
        relevant_missing = {"receptions", "receiving_yards", "rushing_yards"}
    missing_labels = [
        _STAT_LABELS.get(s, s)
        for s in prov.get("missing", [])
        if s in relevant_missing
    ]
    td = prov.get("td", {})
    pass_td = prov.get("pass_td", {}) if group == "qb" else {}

    # Build the TD phrase.
    if not td.get("available"):
        td_note = "no TD line"
    elif td.get("ladder") is True:
        n = td.get("n_anchors")
        td_note = f"full TD ladder ({n} anchors)" if n else "full TD ladder"
    elif td.get("ladder") is False:
        td_note = "anytime-TD only (no ladder)"
    else:  # None -> legacy/unknown
        td_note = "TD line present (ladder unknown)"

    # Build the passing-TD phrase (QBs only).
    pass_td_note = None
    if group == "qb":
        if not pass_td.get("available"):
            pass_td_note = "no passing-TD line"
        elif pass_td.get("ladder") is True:
            n = pass_td.get("n_anchors")
            pass_td_note = (
                f"full passing-TD ladder ({n} anchors)" if n else "full passing-TD ladder"
            )
        else:
            pass_td_note = "passing-TD anchor only (no ladder)"

    # One-line summary: what's there, what's not, and the TD detail(s).
    parts = []
    if present_labels:
        parts.append("has " + ", ".join(present_labels))
    if missing_labels:
        parts.append("no " + ", ".join(missing_labels))
    parts.append(td_note)
    if pass_td_note:
        parts.append(pass_td_note)

    # Low-confidence flag: a continuous stat with a single anchor (main line
    # only, no alternates) has an ASSUMED shape/tail, not a measured one.
    single = prov.get("single_anchor_stats", []) or []
    low_conf = bool(prov.get("low_confidence")) or bool(single)
    if single:
        labels = ", ".join(_STAT_LABELS.get(s, s) for s in single)
        parts.append(f"⚠ {labels} main line only (assumed shape — low confidence)")

    summary = "; ".join(parts)

    return {
        "present": present_labels,
        "missing": missing_labels,
        "group": group,
        "td": {
            "available": bool(td.get("available")),
            "ladder": td.get("ladder"),
            "n_anchors": td.get("n_anchors"),
            "model": td.get("model"),
        },
        "pass_td": {
            "available": bool(pass_td.get("available")),
            "ladder": pass_td.get("ladder"),
            "n_anchors": pass_td.get("n_anchors"),
            "model": pass_td.get("model"),
        } if group == "qb" else None,
        "anchor_counts": prov.get("anchor_counts", {}),
        "single_anchor_stats": single,
        "low_confidence": low_conf,
        "summary": summary,
    }



def title_for(store: dict) -> str:
    return (
        f"Projected fantasy points — {store.get('week_label')} "
        f"({store.get('scoring')}, "
        f"{'correlated' if store.get('correlated') else 'independent'}, "
        f"book={store.get('book')})"
    )


def build_figure(store: dict, series: list[tuple[str, dict]], mode: str = "density"):
    """Return a Plotly Figure for the requested view.

    mode="density" (default): density curves — overlay for <=3 players, small
        multiples for 4+. Good for distribution *shape*.
    mode="cdf": cumulative distribution curves (the flipped CDF / quantile
        function, x=percentile y=points), always overlaid. Good for comparing
        *floors/ceilings*: a horizontal gap between two curves is the points
        difference at a given percentile; a vertical gap is the probability
        difference at a given score. Built straight from the stored dense
        quantile grid (no re-simulation).
    mode="cdf_std": the STANDARD (textbook) CDF, x=fantasy points, y=cumulative
        probability (0-1), always overlaid. Reads "P(score <= x)": a steeper
        curve is a more predictable player; the horizontal position of a given
        cumulative level is that player's quantile at that level.
    mode="diff": TWO players only. x=percentile, y = points difference between
        the two players at that percentile, oriented so the HIGHER-MEAN player is
        the positive direction. y>0 means the higher-mean player is ahead at that
        percentile; where it crosses below zero, the lower-mean player actually
        wins (e.g. a higher floor).
    """
    if mode == "cdf":
        return _cdf_overlay(store, series)
    if mode == "cdf_std":
        return _cdf_std_overlay(store, series)
    if mode == "diff":
        return _diff_chart(store, series)
    if len(series) <= 3:
        return _overlay(store, series)
    return _small_multiples(store, series)


def percentile_at(rec: dict, x) -> np.ndarray:
    """Percentile(s) in [0,100] for score value(s) ``x`` for one player.

    Inverts the stored dense quantile grid (``rec["quantiles"]`` = {levels,
    values}) via linear interpolation: given a points value, return the share of
    outcomes at or below it. This is the exact stored CDF (the same grid used for
    any P(>x)), so the hover percentile is consistent with the model, not a
    re-estimate from the smoothed display curve.

    Clamps to [0,100] and handles the flat regions at the ends gracefully.
    """
    q = rec.get("quantiles") or {}
    levels = np.asarray(q.get("levels", []), dtype=float)
    values = np.asarray(q.get("values", []), dtype=float)
    x = np.asarray(x, dtype=float)
    if levels.size < 2 or values.size != levels.size:
        return np.full(x.shape, np.nan)
    # values[i] is the score at cumulative level[i] (non-decreasing in level by
    # construction), so interpolating value -> level inverts the CDF.
    pct = np.interp(x, values, levels, left=levels[0], right=levels[-1])
    return np.clip(pct * 100.0, 0.0, 100.0)


def _overlay(store, series):
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, rec) in enumerate(series):
        r, g, b = PALETTE[i % len(PALETTE)]
        x, y = density_from_histogram(rec)
        pct = percentile_at(rec, x)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{name}  (mean {rec['mean']:.1f})",
                line=dict(color=f"rgb({r},{g},{b})", width=2),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{FILL_ALPHA})",
                customdata=pct,
                hovertemplate=(
                    "%{x:.1f} pts<br>%{customdata:.0f}th pctile"
                    "<extra>" + name + "</extra>"
                ),
            )
        )
        fig.add_vline(x=rec["mean"], line_width=1, line_dash="dot",
                      line_color=f"rgb({r},{g},{b})", opacity=0.6)
    fig.update_layout(
        title=title_for(store), xaxis_title="fantasy points", yaxis_title="density",
        legend_title="player", template="plotly_white", hovermode="x unified",
    )
    return fig


def _small_multiples(store, series):
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(series)
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    hi = max(float(rec["histogram"]["edges"][-1]) for _, rec in series)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{name} (mean {rec['mean']:.1f})" for name, rec in series],
        shared_xaxes=True, shared_yaxes=False,  # independent y so low peaks aren't flattened
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    for i, (name, rec) in enumerate(series):
        r, g, b = PALETTE[i % len(PALETTE)]
        row, col = i // cols + 1, i % cols + 1
        x, y = density_from_histogram(rec)
        pct = percentile_at(rec, x)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", name=name, showlegend=False,
                line=dict(color=f"rgb({r},{g},{b})", width=2),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{FILL_ALPHA})",
                customdata=pct,
                hovertemplate=(
                    "%{x:.1f} pts<br>%{customdata:.0f}th pctile"
                    "<extra>" + name + "</extra>"
                ),
            ),
            row=row, col=col,
        )
        fig.add_vline(x=rec["mean"], line_width=1, line_dash="dot",
                      line_color=f"rgb({r},{g},{b})", opacity=0.6, row=row, col=col)
    fig.update_xaxes(range=[0, hi], title_text="fantasy points")
    fig.update_yaxes(title_text="density")
    fig.update_layout(title=title_for(store), template="plotly_white", showlegend=False)
    return fig


def _cdf_xy(rec: dict):
    """(x=cumulative percent, y=points) for a player's quantile curve.

    Axes are intentionally SWAPPED from a textbook CDF: we plot the quantile
    function (points as a function of percentile) rather than the CDF (percentile
    as a function of points). For the playground's job -- comparing players'
    floors/ceilings -- this is cleaner: the *vertical* gap between two curves at
    a given percentile is read directly as the points difference (the unit the
    user reasons in), and the curve sweeps floor (left) -> ceiling (right) with
    "up = more points".

    Built straight from the stored dense quantile grid (x = levels as percent,
    y = quantile values), so it's the exact model CDF -- no smoothing/binning.
    """
    q = rec.get("quantiles") or {}
    levels = np.asarray(q.get("levels", []), dtype=float)
    values = np.asarray(q.get("values", []), dtype=float)
    if levels.size < 2 or values.size != levels.size:
        return np.array([]), np.array([])
    return levels * 100.0, values


def _cdf_overlay(store, series):
    """Overlaid quantile curves (points-by-percentile) for all selected players.

    Always overlaid (not small-multiplied): the whole point is to read how the
    curves sit relative to each other. A curve that sits HIGHER at a given
    percentile is the better distribution there; where curves cross, their
    ordering flips (e.g. a higher floor but lower ceiling shows as a crossover).
    Monotone curves stay legible with several players, unlike overlapping
    densities.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, rec) in enumerate(series):
        r, g, b = PALETTE[i % len(PALETTE)]
        x, y = _cdf_xy(rec)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{name}  (mean {rec['mean']:.1f})",
                line=dict(color=f"rgb({r},{g},{b})", width=2),
                hovertemplate=(
                    "%{x:.0f}th pctile<br>%{y:.1f} pts"
                    "<extra>" + name + "</extra>"
                ),
            )
        )
        # Median marker: each player's p50 points level (horizontal now).
        fig.add_hline(
            y=rec["median"], line_width=1, line_dash="dot",
            line_color=f"rgb({r},{g},{b})", opacity=0.5,
        )
    # Vertical reference lines at the floor/ceiling percentiles people ask about,
    # so reading each player's points at p10 / p50 / p90 is a clean vertical grid.
    for p in (10, 50, 90):
        fig.add_vline(x=p, line_width=1, line_dash="dot", line_color="#cccccc", opacity=0.7)
    fig.update_layout(
        title=title_for(store) + "  —  points by percentile",
        xaxis_title="percentile",
        yaxis_title="fantasy points",
        legend_title="player", template="plotly_white", hovermode="x unified",
    )
    fig.update_xaxes(range=[0, 100])
    return fig


def _cdf_std_xy(rec: dict):
    """(x=fantasy points, y=cumulative probability in [0,1]) — the STANDARD CDF.

    This is the stored quantile grid with axes un-swapped from ``_cdf_xy``:
    x = quantile values (points), y = levels (cumulative probability). Reads
    "the probability the player scores at or below x points". Exact model CDF
    (no smoothing/binning).
    """
    q = rec.get("quantiles") or {}
    levels = np.asarray(q.get("levels", []), dtype=float)
    values = np.asarray(q.get("values", []), dtype=float)
    if levels.size < 2 or values.size != levels.size:
        return np.array([]), np.array([])
    return values, levels


def _cdf_std_overlay(store, series):
    """Overlaid standard CDFs (x=points, y=cumulative probability) for all players."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, rec) in enumerate(series):
        r, g, b = PALETTE[i % len(PALETTE)]
        x, y = _cdf_std_xy(rec)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{name}  (mean {rec['mean']:.1f})",
                line=dict(color=f"rgb({r},{g},{b})", width=2),
                hovertemplate=(
                    "%{x:.1f} pts<br>P(≤)=%{y:.2f}"
                    "<extra>" + name + "</extra>"
                ),
            )
        )
    # Horizontal reference lines at the common cumulative levels.
    for p in (0.1, 0.5, 0.9):
        fig.add_hline(y=p, line_width=1, line_dash="dot", line_color="#cccccc", opacity=0.7)
    fig.update_layout(
        title=title_for(store) + "  —  cumulative probability",
        xaxis_title="fantasy points",
        yaxis_title="P(score ≤ x)",
        legend_title="player", template="plotly_white", hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def diff_series(series: list[tuple[str, dict]]):
    """Compute the two-player percentile-difference curve.

    Returns ``(levels_pct, diff, hi_name, lo_name)`` where:
      - ``levels_pct`` is the shared percentile grid (0-100),
      - ``diff[i] = hi.points(p_i) - lo.points(p_i)``, oriented so the
        HIGHER-MEAN player ("hi") is the positive direction,
      - ``hi_name`` / ``lo_name`` label the two players.

    Both players must expose the same stored quantile ``levels`` grid (they do:
    the slate writes one shared ``default_quantile_levels`` grid for every
    player). Raises ``ValueError`` if not exactly two players or the grids differ.
    """
    if len(series) != 2:
        raise ValueError("diff chart requires exactly 2 players")
    (name_a, rec_a), (name_b, rec_b) = series
    qa = rec_a.get("quantiles") or {}
    qb = rec_b.get("quantiles") or {}
    la = np.asarray(qa.get("levels", []), dtype=float)
    lb = np.asarray(qb.get("levels", []), dtype=float)
    va = np.asarray(qa.get("values", []), dtype=float)
    vb = np.asarray(qb.get("values", []), dtype=float)
    if la.size < 2 or la.size != va.size or lb.size != vb.size:
        raise ValueError("both players need a valid quantile grid")
    if la.size != lb.size or not np.allclose(la, lb):
        # Fall back to the intersection grid by interpolating B onto A's levels.
        vb = np.interp(la, lb, vb)
        lb = la
    # Orient by mean: higher-mean player is positive.
    if rec_b["mean"] > rec_a["mean"]:
        hi_name, lo_name, hi_v, lo_v = name_b, name_a, vb, va
    else:
        hi_name, lo_name, hi_v, lo_v = name_a, name_b, va, vb
    return la * 100.0, hi_v - lo_v, hi_name, lo_name


def _diff_chart(store, series):
    """Two-player difference chart: points gap by percentile, higher-mean positive."""
    import plotly.graph_objects as go

    x, diff, hi_name, lo_name = diff_series(series)
    fig = go.Figure()
    # Shade where the higher-mean player is ahead (>0) vs behind (<0) using one
    # trace with fill to zero; color the whole line a neutral accent.
    fig.add_trace(
        go.Scatter(
            x=x, y=diff, mode="lines",
            name=f"{hi_name} − {lo_name}",
            line=dict(color="rgb(31,119,180)", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
            hovertemplate=(
                "%{x:.0f}th pctile<br>%{y:+.1f} pts"
                "<extra>" + hi_name + " − " + lo_name + "</extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1.5, line_color="#888888")
    for p in (10, 50, 90):
        fig.add_vline(x=p, line_width=1, line_dash="dot", line_color="#cccccc", opacity=0.7)
    fig.update_layout(
        title=(
            title_for(store)
            + f"  —  {hi_name} minus {lo_name} (higher mean = positive)"
        ),
        xaxis_title="percentile",
        yaxis_title=f"points: {hi_name} − {lo_name}",
        template="plotly_white", hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(range=[0, 100])
    return fig
