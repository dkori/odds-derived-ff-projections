"""Parse a Yahoo Fantasy *matchup* page into per-player projected/actual points.

The matchup table is rendered as a single mirrored table per matchup
(``statTable1`` = your matchup, further tables for other matchups you can view).
Each ``<tr>`` holds one roster slot for *both* teams at once, laid out
symmetrically around a center position column:

    col  0  left  stats breakdown  (e.g. "4 Pass TD, 279 Pass Yds")
    col  1  left  player  (<a class="name" data-ys-playerid=...>)
    col  2  left  PROJECTED points   (th title="Projected Points", class F-shade)
    col  3  left  ACTUAL points      (th title="Fantasy Points",   class Fw-b)
    col  4  (hidden) position
    col  5  center roster slot label (QB/RB/WR/TE/W/R/T/K/DEF, or TOTAL)
    col  6  (hidden) position
    col  7  right ACTUAL points
    col  8  right PROJECTED points
    col  9  right player
    col 10  right stats breakdown

The final row of the table is a ``TOTAL`` row (team totals), which we skip.

The parse is intentionally tolerant: Yahoo markup drifts, so we locate the point
columns by their header ``title`` attributes rather than hard-coded indices
where practical, and fall back to the documented indices otherwise. Empty /
not-yet-played cells (``-``) parse to ``None``.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ffmc.yahoo requires beautifulsoup4 + lxml. "
        "Install with: pip install -e '.[yahoo]'"
    ) from exc


@dataclass(frozen=True)
class PlayerLine:
    """One player's projected vs actual fantasy points for a week."""

    player_id: str
    name: str
    nfl_team: Optional[str]        # e.g. "Dal"
    position: Optional[str]        # the player's position, e.g. "QB"
    roster_slot: Optional[str]     # the slot they occupy, e.g. "W/R/T" or "BN"
    team_side: str                 # "home" (left) or "away" (right)
    projected: Optional[float]
    actual: Optional[float]
    owner: Optional[str] = None    # fantasy team (manager) name that rosters them

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchupPage:
    """All player lines parsed from a matchup page, plus any week hint found."""

    week: Optional[int]
    players: list[PlayerLine]

    def to_dict(self) -> dict:
        return {"week": self.week, "players": [p.to_dict() for p in self.players]}


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# "Dal - QB" / "GB - WR" (team may be title- or upper-case) then position.
_TEAM_POS_RE = re.compile(r"\b([A-Z][A-Za-z]{1,2})\s*-\s*(QB|RB|WR|TE|K|DEF|D/ST)\b")


def _num(text: str) -> Optional[float]:
    """Parse a point cell; '-' / '' / non-numeric -> None."""
    if text is None:
        return None
    text = text.strip()
    if not text or text in {"-", "–", "—"}:
        return None
    m = _NUM_RE.search(text)
    return float(m.group()) if m else None


_POS_TOKENS = {"QB", "RB", "WR", "TE", "K", "DEF", "D/ST", "FB"}


def _hidden_position(tds, cols, is_left: bool) -> Optional[str]:
    """Player's NFL position from the hidden columns flanking the center slot.

    Layout: ... left_actual, [hidden pos L], center slot, [hidden pos R],
    right_actual ... So the left player's position sits just after left_actual
    and the right player's just before right_actual.
    """
    idx = cols["left_actual"] + 1 if is_left else cols["right_actual"] - 1
    if 0 <= idx < len(tds):
        txt = tds[idx].get_text(strip=True)
        if txt in _POS_TOKENS:
            return txt
    return None


def _player_from_anchor(anchor) -> tuple[str, str]:
    """(player_id, display_name) from a `a.name[data-ys-playerid]` anchor."""
    return anchor.get("data-ys-playerid", ""), anchor.get_text(strip=True)


def _player_id_in(cell) -> str:
    """First data-ys-playerid found anywhere in a player cell ('' if none).

    Yahoo puts the id on the name anchor in some layouts and on a sibling
    playernote/video anchor in others, so we search the whole cell.
    """
    if cell is None:
        return ""
    el = cell.select_one("[data-ys-playerid]")
    return el.get("data-ys-playerid", "") if el else ""


def _team_pos(cell) -> tuple[Optional[str], Optional[str]]:
    """Pull 'Dal - QB' out of the player cell's detail span."""
    m = _TEAM_POS_RE.search(cell.get_text(" ", strip=True))
    if m:
        return m.group(1), m.group(2)
    return None, None


def _column_index_map(table) -> dict:
    """Map semantic columns to indices using the header `title`s when possible.

    Returns indices for: left_proj, left_actual, right_proj, right_actual.
    Falls back to the documented layout (2/3/8/7) if headers are missing.
    """
    header = table.find("thead")
    titles: list[str] = []
    if header:
        for th in header.find_all("th"):
            titles.append((th.get("title") or th.get_text(strip=True) or "").lower())

    proj_cols = [i for i, t in enumerate(titles) if "projected" in t]
    actual_cols = [i for i, t in enumerate(titles) if t == "fantasy points"]

    if len(proj_cols) >= 2 and len(actual_cols) >= 2:
        # left columns are the lower indices, right columns the higher ones
        return {
            "left_proj": proj_cols[0],
            "left_actual": actual_cols[0],
            "right_actual": actual_cols[-1],
            "right_proj": proj_cols[-1],
        }
    # documented fallback
    return {"left_proj": 2, "left_actual": 3, "right_actual": 7, "right_proj": 8}


def _slot_from_center(tds, cols) -> Optional[str]:
    """The center roster-slot label (QB/RB/.../W/R/T/BN/TOTAL).

    It sits between the left-actual and right-actual columns. The real page
    flanks it with two *hidden* position cells, so we can't just take the first
    text match. The visible center cell carries the ``Bg-shade`` class; prefer
    it, then fall back to the geometric midpoint, then any slot-shaped text.
    """
    lo = cols["left_actual"]
    hi = cols["right_actual"]
    candidates = list(range(lo + 1, hi))
    if not candidates:
        return None

    # 1) the visible center cell is Bg-shade (and not Hidden)
    for i in candidates:
        classes = tds[i].get("class") or []
        if "Bg-shade" in classes and "Hidden" not in classes:
            txt = tds[i].get_text(strip=True)
            if txt:
                return txt

    # 2) geometric midpoint
    mid = (lo + hi) // 2
    if lo < mid < hi:
        txt = tds[mid].get_text(strip=True)
        if txt:
            return txt

    # 3) any slot-shaped text (composite like W/R/T, or short upper token)
    for i in candidates:
        txt = tds[i].get_text(strip=True)
        if txt and "/" in txt:
            return txt
        if txt and txt.isupper() and txt.isalpha() and 1 <= len(txt) <= 5:
            return txt
    return None


def _parse_table(table) -> list[PlayerLine]:
    cols = _column_index_map(table)
    body = table.find("tbody")
    if body is None:
        return []

    out: list[PlayerLine] = []
    for tr in body.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) <= max(cols.values()):
            continue  # not a full player row

        slot = _slot_from_center(tds, cols)
        if slot == "TOTAL":
            continue  # team-total footer row

        # The display-name anchor carries class 'name' and a title/text. The
        # player id lives either on that same anchor (2026 layout) or on a
        # sibling 'playernote' anchor in the same cell (2025 layout), so we
        # resolve the id per-cell rather than requiring it on the name anchor.
        name_anchors = [
            a for a in tr.select("a.name")
            if (a.get("title") or a.get_text(strip=True))
        ]
        for anchor in name_anchors:
            cell = anchor.find_parent("td")
            name = anchor.get("title") or anchor.get_text(strip=True)
            pid = _player_id_in(cell) or anchor.get("data-ys-playerid", "")
            if not name:
                continue
            # Determine side by the anchor's cell index.
            try:
                cidx = tds.index(cell)
            except ValueError:
                cidx = 0
            is_left = cidx <= cols["left_actual"]
            team, pos = _team_pos(cell)

            if is_left:
                proj = _num(tds[cols["left_proj"]].get_text(strip=True))
                actual = _num(tds[cols["left_actual"]].get_text(strip=True))
                side = "home"
            else:
                proj = _num(tds[cols["right_proj"]].get_text(strip=True))
                actual = _num(tds[cols["right_actual"]].get_text(strip=True))
                side = "away"

            # Older pages don't carry "Team - Pos" in the name cell, but the
            # player's position is in the hidden flanking columns: the one just
            # inside the center for the left player, just outside for the right.
            if pos is None:
                pos = _hidden_position(tds, cols, is_left)

            out.append(
                PlayerLine(
                    player_id=pid,
                    name=name,
                    nfl_team=team,
                    position=pos,
                    roster_slot=slot,
                    team_side=side,
                    projected=proj,
                    actual=actual,
                )
            )
    return out


def parse_matchup_teams(html: str) -> tuple[Optional[int], Optional[int]]:
    """Return the (self_mid, opponent_mid) team ids this matchup page covers.

    Yahoo embeds the viewed matchup as ``mid1=<self>`` (dominant, repeated in
    many links) and ``mid2=<opponent>``. We take the most common mid1 and the
    mid2 that differs from it. Either may be ``None`` if not found. The crawler
    uses this to skip the opponent's redundant page.
    """
    mid1s = re.findall(r"mid1=(\d+)", html)
    mid2s = re.findall(r"mid2=(\d+)", html)
    self_id = Counter(mid1s).most_common(1)[0][0] if mid1s else None
    opp_id = next((m for m in mid2s if m != self_id), None)
    if opp_id is None and mid2s:
        opp_id = Counter(mid2s).most_common(1)[0][0]
    return (
        int(self_id) if self_id is not None else None,
        int(opp_id) if opp_id is not None else None,
    )


# A matchup header links each team as:
#   <a class="F-link" href="https://football.fantasysports.yahoo.com/f1/<lg>/<id>">Team Name
# They render home (left) first, then away (right) — the same order as the two
# roster stat tables, so index 0 -> home side, index 1 -> away side.
_TEAM_NAME_RE = re.compile(
    r'href="https?://[^"]*?/f1/\d+/\d+"\s*>\s*([^<]{1,60}?)\s*<',
)


def parse_matchup_team_names(html: str) -> tuple[Optional[str], Optional[str]]:
    """Return the (home_team_name, away_team_name) fantasy team names.

    The matchup header links both teams in home-then-away order (the same order
    as the two roster tables). Returns ``(None, None)`` if the names can't be
    found (so callers degrade gracefully — e.g. a layout change just drops the
    owner enrichment rather than crashing the crawl).
    """
    names = _TEAM_NAME_RE.findall(html)
    # De-duplicate consecutive repeats while preserving order (Yahoo sometimes
    # repeats the same team link in a sub-nav); we only need the first two
    # distinct names, home then away.
    seen: list[str] = []
    for n in names:
        n = n.strip()
        if n and (not seen or seen[-1] != n) and n not in seen:
            seen.append(n)
        if len(seen) == 2:
            break
    home = seen[0] if len(seen) >= 1 else None
    away = seen[1] if len(seen) >= 2 else None
    return home, away


def _detect_week(soup) -> Optional[int]:
    """Best-effort week number from the page (selected week dropdown / URL)."""
    # The week selector marks the current week; also appears in canonical links.
    sel = soup.find("option", selected=True)
    if sel and sel.get("value", "").isdigit():
        return int(sel["value"])
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        m = re.search(r"[?&]week=(\d+)", link["href"])
        if m:
            return int(m.group(1))
    m = re.search(r"[?&]week=(\d+)", soup.decode()[:5000])
    return int(m.group(1)) if m else None


def parse_matchup_html(html: str, *, week: Optional[int] = None) -> MatchupPage:
    """Parse matchup-page HTML into a :class:`MatchupPage`.

    ``week`` overrides the auto-detected week when provided (recommended, since
    you know which week you fetched).
    """
    soup = BeautifulSoup(html, "lxml")
    players: list[PlayerLine] = []

    # statTable1 is your matchup; additional matchups render as statTable2, ...
    # We parse every stat table present so a league-wide page also works.
    tables = soup.find_all("table", id=re.compile(r"^statTable\d+$"))
    for table in tables:
        players.extend(_parse_table(table))

    # de-dupe on (player_id, team_side) in case of repeated tables
    seen = set()
    deduped = []
    for p in players:
        key = (p.player_id, p.team_side)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return MatchupPage(week=week if week is not None else _detect_week(soup),
                       players=deduped)
