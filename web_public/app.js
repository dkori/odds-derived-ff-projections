/* Public odds-derived projections app (vanilla JS + Plotly).
 * Loads one enriched JSON (named by data/latest.json), lets a user pick EXACTLY 2
 * players, and renders 3 charts with explainers + our-mean / Yahoo-proj / position
 * labels. The only facet is POSITION (QB/RB/WR/TE) — no league/owner data.
 */

let DATA = null;        // the loaded payload (current week)
let NAMES = [];         // sorted selectable player names
let selected = [];      // up to 2 names
let colorOf = {};       // name -> stable palette slot (0/1) while selected
let activeScoring = 'half_ppr';  // scoring toggle; default 0.5 PPR (this league)
let WEEKS = [];         // weeks manifest entries [{week, file}], if present

// Hover stays on (the unified spike line + value tooltip come from the per-chart
// hovermode:'x unified'); zoom/pan is disabled so a stray drag or scroll on
// mobile can't zoom the plot.
const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
  doubleClick: false,
};

async function boot() {
  // Prior-week serving: a weeks manifest (weeks.json) lists every shipped week's
  // asset and which is latest. Each week is its own file loaded on demand, so
  // offering prior weeks costs nothing to the default page load. Falls back to
  // the single latest.json pointer, then to a hardcoded file, if the manifest
  // is missing (older deploys).
  let latestFile = 'data/projections_week6.json';
  try {
    const man = await (await fetch('data/weeks.json')).json();
    if (man && Array.isArray(man.weeks) && man.weeks.length) {
      WEEKS = man.weeks.slice().sort((a, b) => b.week - a.week); // newest first
      latestFile = 'data/' + (man.latest || WEEKS[0].file);
    }
  } catch (e) {
    try {
      const ptr = await (await fetch('data/latest.json')).json();
      if (ptr && ptr.file) latestFile = 'data/' + ptr.file;
    } catch (e2) {
      console.warn('no weeks.json/latest.json; using fallback', latestFile);
    }
  }
  populateWeekPicker(latestFile);
  await loadWeek(latestFile);
  wireControls();
  wireViewControls();
  render();
}

// Populate the week dropdown from the manifest (newest first). If there's no
// manifest (one-week deploy) the picker is hidden.
function populateWeekPicker(latestFile) {
  const sel = document.getElementById('week-picker');
  if (!sel) return;
  if (!WEEKS.length) { sel.style.display = 'none'; return; }
  const latestName = latestFile.replace(/^data\//, '');
  sel.innerHTML = WEEKS.map((w, i) => {
    const sl = ('data/' + w.file === latestFile) || (i === 0 && !latestName);
    return `<option value="data/${escapeHtml(w.file)}"${sl ? ' selected' : ''}>` +
      `Week ${w.week}${i === 0 ? ' (latest)' : ''}</option>`;
  }).join('');
}

// Load one week's asset and refresh the selectable-name list. Selections are
// kept across a week change; a player absent from the new week is dropped.
async function loadWeek(dataFile) {
  const res = await fetch(dataFile);
  DATA = await res.json();
  // Default the scoring toggle to the store's declared default when present.
  // (Only on first load — respect a user's explicit choice afterward via the
  // select's own value in wireViewControls.)
  NAMES = Object.keys(DATA.players)
    .filter((n) => playerHasCurve(DATA.players[n]))
    .sort();
  // Drop any selected players not present in the newly loaded week.
  selected = selected.filter((n) => DATA.players[n]);
  Object.keys(colorOf).forEach((n) => { if (!DATA.players[n]) delete colorOf[n]; });
  renderMeta();
}

// A player is chartable under the ACTIVE scoring if that scoring's quantile grid
// (or the flat grid, for pre-multi-scoring weeks) has more than one level.
function playerHasCurve(rec) {
  const q = scoringField(rec, 'quantiles');
  return q && q.levels && q.levels.length > 1;
}

// Return a distribution field (mean/median/std/quantiles/histogram/…) for the
// ACTIVE scoring. New weeks carry a `scorings` block keyed standard/half_ppr/ppr;
// older weeks only have the flat 0.5-PPR fields, so we fall back to those. This
// is what makes the toggle degrade gracefully on prior weeks.
function scoringField(rec, key) {
  if (rec && rec.scorings && rec.scorings[activeScoring]
      && rec.scorings[activeScoring][key] != null) {
    return rec.scorings[activeScoring][key];
  }
  return rec ? rec[key] : undefined;  // flat fallback (store's declared scoring)
}

// A record "view" for charting under the active scoring: the scoring-specific
// distribution fields spread over the static per-player fields (positions,
// yahoo_proj, nfl_team, name). Charts read this instead of the raw record.
function recView(name) {
  const r = DATA.players[name];
  if (!r) return null;
  return {
    name,
    positions: r.positions,
    yahoo_proj: r.yahoo_proj,
    nfl_team: r.nfl_team,
    mean: scoringField(r, 'mean'),
    median: scoringField(r, 'median'),
    std: scoringField(r, 'std'),
    floor_p10: scoringField(r, 'floor_p10'),
    ceiling_p90: scoringField(r, 'ceiling_p90'),
    quantiles: scoringField(r, 'quantiles'),
    histogram: scoringField(r, 'histogram'),
  };
}

// Wire the week picker + scoring toggle (the "view" controls, distinct from the
// player-picker controls in wireControls).
function wireViewControls() {
  const weekSel = document.getElementById('week-picker');
  const scoreSel = document.getElementById('scoring-mode');
  if (scoreSel) {
    scoreSel.value = activeScoring;
    scoreSel.addEventListener('change', () => {
      activeScoring = scoreSel.value;
      // Re-filter chartable names (a player could lack a curve under a scoring,
      // though in practice all three are present together) and re-render.
      NAMES = Object.keys(DATA.players).filter((n) => playerHasCurve(DATA.players[n])).sort();
      render();
    });
  }
  if (weekSel) {
    weekSel.addEventListener('change', async () => {
      await loadWeek(weekSel.value);
      // Rebuild the position filter counts etc. for the new week, then render.
      render();
    });
  }
}

function renderMeta() {
  const total = DATA.n_players;
  const pc = DATA.position_counts || {};
  const posSummary = (DATA.positions || [])
    .map((p) => `${p} ${pc[p] != null ? pc[p] : 0}`)
    .join(' · ');
  const scoreLabel = { standard: 'Standard', half_ppr: '0.5 PPR', ppr: 'Full PPR' }[activeScoring]
    || DATA.scoring;
  document.getElementById('meta').textContent =
    `${DATA.week_label} · ${DATA.season} week ${DATA.week} · book ${DATA.book} · ` +
    `${scoreLabel} · ${total} priced players (${posSummary})`;
}

function wireControls() {
  const input = document.getElementById('search');
  const list = document.getElementById('matches');
  const posSel = document.getElementById('pos-filter');
  const sortSel = document.getElementById('sort-mode');
  const trigger = document.getElementById('player-trigger');
  const panel = document.getElementById('player-panel');

  // Populate the position dropdown: "All positions" then each position in the
  // dataset's declared order (QB/RB/WR/TE).
  const opts = ['<option value="__all__">All positions</option>'];
  (DATA.positions || []).forEach((p) =>
    opts.push(`<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`));
  posSel.innerHTML = opts.join('');

  // Order a name list by the current sort mode. "mean" sorts by our projected
  // mean descending (highest first) — the "who are the best options here"
  // ordering; ties and missing means fall back to alphabetical for stability.
  function applySort(names) {
    if (sortSel && sortSel.value === 'mean') {
      return names.slice().sort((a, b) => {
        const ma = scoringField(DATA.players[a], 'mean');
        const mb = scoringField(DATA.players[b], 'mean');
        if (mb !== ma) return mb - ma;
        return a.localeCompare(b);
      });
    }
    return names; // NAMES is already alphabetical
  }

  // Candidate names under the current position filter, then ordered by the sort
  // mode. A player is a candidate if their `positions` array includes the
  // selected position (multi-eligible players show under each of their positions).
  function candidates() {
    const v = posSel.value;
    const base = (v === '__all__')
      ? NAMES
      : NAMES.filter((n) => (DATA.players[n].positions || []).includes(v));
    return applySort(base);
  }

  // Render the full candidate list (filtered by the in-panel search text).
  function refreshMatches() {
    const pool = candidates();
    const q = input.value.trim().toLowerCase();
    let hits;
    if (!q) {
      hits = pool.slice(0, 300); // whole (position-filtered) list, tap to pick
    } else {
      const starts = pool.filter((n) => n.toLowerCase().startsWith(q));
      const contains = pool.filter((n) => n.toLowerCase().includes(q) && !starts.includes(n));
      hits = starts.concat(contains).slice(0, 60);
    }
    list.innerHTML = hits.length
      ? hits.map((n) => `<li data-name="${escapeHtml(n)}">${escapeHtml(n)}${badge(n)}</li>`).join('')
      : '<li class="empty">no players</li>';
  }

  function openPanel() {
    panel.classList.add('open');
    trigger.setAttribute('aria-expanded', 'true');
    refreshMatches();
    // Intentionally do NOT focus the search input — on mobile that would pop the
    // keyboard the moment the dropdown opens. The user taps the field to type.
  }
  function closePanel() {
    panel.classList.remove('open');
    trigger.setAttribute('aria-expanded', 'false');
  }
  function togglePanel() {
    if (panel.classList.contains('open')) closePanel(); else openPanel();
  }

  trigger.addEventListener('click', togglePanel);
  posSel.addEventListener('change', () => { input.value = ''; if (panel.classList.contains('open')) refreshMatches(); });
  if (sortSel) sortSel.addEventListener('change', () => { if (panel.classList.contains('open')) refreshMatches(); });
  input.addEventListener('input', refreshMatches);

  list.addEventListener('click', (e) => {
    const li = e.target.closest('li[data-name]');
    if (!li) return;
    addPlayer(li.dataset.name);
    input.value = '';
    closePanel();
  });

  // Close on outside click / Escape.
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.player-select')) closePanel();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePanel(); });
}

function posLabel(name) {
  const r = DATA.players[name];
  if (!r) return '';
  const pos = (r.positions || []).join('/');
  const team = r.nfl_team ? ` · ${r.nfl_team}` : '';
  return pos ? pos + team : (team ? team.replace(' · ', '') : '');
}

function badge(name) {
  const r = DATA.players[name];
  const label = posLabel(name);
  const parts = [];
  // When sorting by our mean, surface the mean inline so the descending order
  // is readable at a glance (e.g. scanning for the best RB).
  const sortSel = document.getElementById('sort-mode');
  const meanV = scoringField(r, 'mean');
  if (sortSel && sortSel.value === 'mean' && r && typeof meanV === 'number') {
    parts.push(`<span class="mini-mean">${meanV.toFixed(1)}</span>`);
  }
  if (label) parts.push(`<span class="mini-pos">${escapeHtml(label)}</span>`);
  return parts.length ? ' ' + parts.join(' ') : '';
}

function addPlayer(name) {
  if (!DATA.players[name]) return;
  if (selected.includes(name)) return;
  if (selected.length >= 2) {
    // Hard cap at 2: replace the oldest so the UX stays a head-to-head.
    const dropped = selected.shift();
    delete colorOf[dropped];
  }
  // Assign the lowest free palette slot (0 or 1) so a player keeps the SAME
  // color for as long as they're selected.
  const used = new Set(Object.values(colorOf));
  const slot = used.has(0) ? 1 : 0;
  colorOf[name] = slot;
  selected.push(name);
  render();
}

function removePlayer(name) {
  selected = selected.filter((n) => n !== name);
  delete colorOf[name];
  render();
}

function render() {
  renderChips();
  const label = document.getElementById('player-trigger-label');
  if (label) {
    label.textContent = selected.length === 0
      ? 'Select a player…'
      : (selected.length === 1 ? 'Add a second player…' : 'Replace a player…');
  }
  const charts = document.getElementById('charts');
  const hint = document.getElementById('hint');
  if (selected.length === 0) {
    charts.style.display = 'none';
    hint.style.display = 'block';
    hint.textContent = 'Search and pick two players to compare head-to-head.';
    return;
  }
  hint.style.display = selected.length === 1 ? 'block' : 'none';
  if (selected.length === 1) hint.textContent = 'Pick a second player to unlock the difference chart.';
  charts.style.display = 'block';

  const series = selected.map((n) => ({ name: n, rec: recView(n), i: colorOf[n] }));
  renderCdfStd(series);
  renderDiff(series);
  renderDensity(series);
}

function renderChips() {
  const box = document.getElementById('chips');
  box.innerHTML = selected
    .map((n) => {
      const r = recView(n);
      const label = posLabel(n);
      const posChip = label ? `<span class="chip-pos">${escapeHtml(label)}</span>` : '';
      // Yahoo's projection is scraped under THIS league's scoring (0.5 PPR), so
      // it's only comparable when the toggle is on 0.5 PPR. Under Standard/Full
      // PPR we drop it rather than show a mismatched number.
      const yp = activeScoring === 'half_ppr'
        ? (r.yahoo_proj != null ? ` · Yahoo ${r.yahoo_proj.toFixed(1)}` : ' · no Yahoo proj')
        : '';
      return `<div class="chip">
        <span class="chip-name">${escapeHtml(n)}</span>
        <span class="chip-stat">ours: mean ${r.mean.toFixed(1)} · med ${r.median.toFixed(1)}${yp}</span>
        ${posChip}
        <button data-name="${escapeHtml(n)}" class="chip-x">×</button>
      </div>`;
    })
    .join('');
  box.querySelectorAll('.chip-x').forEach((b) =>
    b.addEventListener('click', () => removePlayer(b.dataset.name)));
}

/* --- the three charts ------------------------------------------------- */

function renderDensity(series) {
  const traces = series.map((s) => {
    const { x, y } = densityFromHistogram(s.rec);
    const pct = percentileAt(s.rec, x);
    return {
      x, y, mode: 'lines', name: `${s.name} (mean ${s.rec.mean.toFixed(1)})`,
      line: { color: rgb(s.i), width: 2 },
      fill: 'tozeroy', fillcolor: rgba(s.i, FILL_ALPHA),
      customdata: pct,
      hovertemplate: '%{x:.1f} pts<br>%{customdata:.0f}th pctile<extra>' + s.name + '</extra>',
    };
  });
  Plotly.newPlot('chart-density', traces, {
    margin: { t: 10, r: 10, b: 72, l: 50 },
    xaxis: { title: 'fantasy points', fixedrange: true },
    yaxis: { title: 'density', fixedrange: true },
    template: 'plotly_white', hovermode: 'x unified', dragmode: false,
    legend: { orientation: 'h', y: -0.3, yanchor: 'top' },
  }, PLOTLY_CONFIG);
}

function renderCdfStd(series) {
  const traces = series.map((s) => {
    const { x, y } = survivalXY(s.rec);
    return {
      x, y, mode: 'lines', name: `${s.name} (mean ${s.rec.mean.toFixed(1)})`,
      line: { color: rgb(s.i), width: 2 },
      // On a unified-x hover the two players sit on DIFFERENT point grids, so
      // printing each trace's own %{x} shows two slightly different "pts" at the
      // same cursor. Show only the probability per player here; the shared point
      // total comes from the x-axis spike label below.
      hovertemplate: 'P(≥) = %{y:.2f}<extra>' + s.name + '</extra>',
    };
  });
  const shapes = [];
  [0.1, 0.5, 0.9].forEach((p) => shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1,
    y0: p, y1: p, line: { color: '#ccc', width: 1, dash: 'dot' } }));
  series.forEach((s) => {
    shapes.push({ type: 'line', x0: s.rec.median, x1: s.rec.median, yref: 'paper', y0: 0, y1: 1,
      line: { color: rgb(s.i), width: 1.2, dash: 'dot' } });
  });
  const annotations = series.map((s) => ({
    x: s.rec.median, xref: 'x', y: 1, yref: 'paper',
    text: `${s.name.split(' ').slice(-1)} med`, showarrow: false,
    font: { size: 10, color: rgb(s.i) }, bgcolor: 'rgba(255,255,255,0.6)', yanchor: 'bottom',
  }));
  Plotly.newPlot('chart-cdf-std', traces, {
    margin: { t: 18, r: 10, b: 72, l: 50 },
    xaxis: {
      title: 'fantasy points', fixedrange: true,
      // A unified spike labels the single shared point total on the axis, so the
      // tooltip rows only need to carry each player's probability.
      showspikes: true, spikemode: 'across+marker', spikesnap: 'cursor',
      spikethickness: 1, spikedash: 'solid', spikecolor: '#888',
    },
    yaxis: { title: 'P(score ≥ x)', range: [0, 1], fixedrange: true },
    template: 'plotly_white', hovermode: 'x unified', dragmode: false, shapes, annotations,
    legend: { orientation: 'h', y: -0.3, yanchor: 'top' },
  }, PLOTLY_CONFIG);
}

function renderDiff(series) {
  const panel = document.getElementById('diff-panel');
  if (series.length !== 2) {
    panel.classList.add('disabled');
    Plotly.purge('chart-diff');
    document.getElementById('diff-note').textContent =
      'The difference chart needs exactly two players.';
    return;
  }
  panel.classList.remove('disabled');
  const { x, diff, hiName, loName, hiV, loV } = diffSeries(series);
  const hiIdx = series.find((s) => s.name === hiName).i;
  const loIdx = series.find((s) => s.name === loName).i;
  const hiColor = rgb(hiIdx);
  const loColor = rgb(loIdx);

  const posY = diff.map((d) => (d >= 0 ? d : 0));
  const negY = diff.map((d) => (d <= 0 ? d : 0));

  const posTrace = {
    x, y: posY, mode: 'lines', name: `${hiName} ahead`,
    line: { color: hiColor, width: 2 },
    fill: 'tozeroy', fillcolor: rgba(hiIdx, 0.15),
    hoverinfo: 'skip',
  };
  const negTrace = {
    x, y: negY, mode: 'lines', name: `${loName} ahead`,
    line: { color: loColor, width: 2 },
    fill: 'tozeroy', fillcolor: rgba(loIdx, 0.15),
    hoverinfo: 'skip',
  };
  const hoverTrace = {
    x: x.map((v) => Math.round(v)),
    y: diff.map((v) => Math.round(v * 10) / 10),
    customdata: hiV.map((v, i) => [
      Math.round(v * 10) / 10,       // higher-mean player's points
      Math.round(loV[i] * 10) / 10,  // lower-mean player's points
    ]),
    mode: 'lines', line: { color: 'rgba(0,0,0,0)', width: 0 },
    name: `${hiName} − ${loName}`, showlegend: false,
    hovertemplate:
      '%{x}th percentile<br>' +
      hiName + ': %{customdata[0]:.1f} pts<br>' +
      loName + ': %{customdata[1]:.1f} pts<br>' +
      'gap: %{y:+.1f} pts (' + hiName + ' − ' + loName + ')' +
      '<extra></extra>',
  };

  const maxAbs = Math.max(1, ...diff.map((d) => Math.abs(d)));
  const M = maxAbs * 1.18;

  const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0,
    line: { color: '#888', width: 1.5 } }];
  [10, 50, 90].forEach((p) => shapes.push({ type: 'line', yref: 'paper', x0: p, x1: p,
    y0: 0, y1: 1, line: { color: '#ccc', width: 1, dash: 'dot' } }));

  const annotations = [
    { x: 50, xref: 'x', y: M * 0.72, yref: 'y', text: `▲ ${hiName} ahead`,
      showarrow: false, font: { size: 13, color: hiColor },
      bgcolor: 'rgba(255,255,255,0.65)' },
    { x: 50, xref: 'x', y: -M * 0.72, yref: 'y', text: `▼ ${loName} ahead`,
      showarrow: false, font: { size: 13, color: loColor },
      bgcolor: 'rgba(255,255,255,0.65)' },
  ];

  Plotly.newPlot('chart-diff', [posTrace, negTrace, hoverTrace], {
    margin: { t: 10, r: 10, b: 72, l: 50 },
    xaxis: { title: 'percentile', range: [0, 100], fixedrange: true },
    yaxis: { title: `points difference`, range: [-M, M], fixedrange: true },
    template: 'plotly_white', hovermode: 'x unified', dragmode: false,
    showlegend: false, shapes, annotations,
  }, PLOTLY_CONFIG);

  const pctAt = (vals, p) => {
    const idx = x.reduce((best, xv, i) => (Math.abs(xv - p) < Math.abs(x[best] - p) ? i : best), 0);
    return Math.round(vals[idx]);
  };
  const anchors = (vals) => `${pctAt(vals, 10)} / ${pctAt(vals, 50)} / ${pctAt(vals, 90)}`;
  document.getElementById('diff-note').innerHTML =
    `Above the line (${escapeHtml(hiName)}'s color) = ${escapeHtml(hiName)} ahead at that ` +
    `percentile; below = ${escapeHtml(loName)} ahead (e.g. a higher floor). ` +
    `<br><b>Floor / median / ceiling (10th / 50th / 90th pctile):</b> ` +
    `${escapeHtml(hiName)} ${anchors(hiV)} pts · ${escapeHtml(loName)} ${anchors(loV)} pts.`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
