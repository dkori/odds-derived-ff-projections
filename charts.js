/* Chart transforms ported 1:1 from src/ffmc/sim/viz.py.
 *
 * The Python tests in tests/test_viz_charts.py lock this exact math, so these
 * JS functions must reproduce it:
 *   - densityFromHistogram  <- density_from_histogram (drop overflow bin + smooth)
 *   - cdfFlippedXY          <- _cdf_xy      (x=percentile, y=points)
 *   - cdfStdXY              <- _cdf_std_xy  (x=points, y=cumulative prob)
 *   - diffSeries            <- diff_series  (2-player, higher-mean positive)
 *
 * No simulation here — everything reads the stored histogram / quantile grid.
 */

// Colorblind-friendly palette (matches viz.PALETTE).
const PALETTE = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
];
const FILL_ALPHA = 0.06;

function rgb(i) {
  const [r, g, b] = PALETTE[i % PALETTE.length];
  return `rgb(${r},${g},${b})`;
}
function rgba(i, a) {
  const [r, g, b] = PALETTE[i % PALETTE.length];
  return `rgba(${r},${g},${b},${a})`;
}

/* --- density from the stored 64-bin histogram ------------------------- */
// Port of density_from_histogram + _smooth_density.
function densityFromHistogram(rec, smooth = true) {
  const counts = rec.histogram.counts.map(Number);
  const edges = rec.histogram.edges.map(Number);
  const n = Number(rec.histogram.n) || counts.reduce((a, b) => a + b, 0) || 1;

  let centers = [];
  let density = [];
  for (let i = 0; i < counts.length; i++) {
    const w = edges[i + 1] - edges[i];
    centers.push(edges[i] + w / 2);
    density.push(counts[i] / (n * w));
  }
  // Drop the overflow bin (folds all mass >= top edge; a spurious right-tail
  // spike). Same guard as the Python (counts.size >= 3).
  if (counts.length >= 3) {
    centers = centers.slice(0, -1);
    density = density.slice(0, -1);
  }
  if (!smooth || centers.length < 3) return { x: centers, y: density };
  return smoothDensity(centers, density);
}

// Port of _smooth_density: Gaussian blur of binned mass onto a 400-pt grid.
function smoothDensity(centers, density) {
  const diffs = [];
  for (let i = 1; i < centers.length; i++) diffs.push(centers[i] - centers[i - 1]);
  const binW = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const sigma = 1.5 * binW;
  const lo = centers[0];
  const hi = centers[centers.length - 1];
  const G = 400;
  const mass = density.map((d) => d * binW);
  const x = [];
  const y = [];
  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  for (let gi = 0; gi < G; gi++) {
    const gx = lo + ((hi - lo) * gi) / (G - 1);
    let s = 0;
    for (let j = 0; j < centers.length; j++) {
      const d = gx - centers[j];
      s += Math.exp(-0.5 * (d / sigma) * (d / sigma)) * norm * mass[j];
    }
    x.push(gx);
    y.push(s);
  }
  return { x, y };
}

/* --- percentile at a score (invert the stored quantile grid) ---------- */
// Port of percentile_at: interp value -> level, return 0..100.
function percentileAt(rec, xs) {
  const levels = rec.quantiles.levels.map(Number);
  const values = rec.quantiles.values.map(Number);
  return xs.map((x) => {
    if (x <= values[0]) return levels[0] * 100;
    if (x >= values[values.length - 1]) return levels[levels.length - 1] * 100;
    let i = 1;
    while (i < values.length && values[i] < x) i++;
    const t = (x - values[i - 1]) / (values[i] - values[i - 1] || 1);
    const lvl = levels[i - 1] + t * (levels[i] - levels[i - 1]);
    return Math.max(0, Math.min(100, lvl * 100));
  });
}

/* --- flipped CDF / quantile function (x=percentile, y=points) --------- */
function cdfFlippedXY(rec) {
  const levels = rec.quantiles.levels.map(Number);
  const values = rec.quantiles.values.map(Number);
  return { x: levels.map((l) => l * 100), y: values };
}

/* --- standard CDF (x=points, y=cumulative probability) ---------------- */
function cdfStdXY(rec) {
  const levels = rec.quantiles.levels.map(Number);
  const values = rec.quantiles.values.map(Number);
  return { x: values, y: levels };
}

/* --- survival / "at least" (x=points, y=P(score >= x)) ---------------- */
// Complement of the standard CDF: y = 1 - level. Downward-sloping. Reads
// "at this score, the probability of scoring AT LEAST that many points."
function survivalXY(rec) {
  const levels = rec.quantiles.levels.map(Number);
  const values = rec.quantiles.values.map(Number);
  return { x: values, y: levels.map((l) => 1 - l) };
}

/* --- flipped survival (x=P(>=) percent, y=points) --------------------- */
// The flipped (quantile-function) chart re-expressed in "at least" terms:
// x = P(score >= y) as a percent (= (1 - level)*100), y = points. Reading
// across: "the points this player hits with at least X% probability." x runs
// high-prob/low-score (left) to low-prob/high-score (right) once reversed.
function flippedSurvivalXY(rec) {
  const levels = rec.quantiles.levels.map(Number);
  const values = rec.quantiles.values.map(Number);
  return { x: levels.map((l) => (1 - l) * 100), y: values };
}

/* --- two-player difference (higher-mean positive) --------------------- */
// Port of diff_series. series = [{name, rec}, {name, rec}].
function diffSeries(series) {
  if (series.length !== 2) throw new Error('diff requires exactly 2 players');
  const a = series[0];
  const b = series[1];
  const la = a.rec.quantiles.levels.map(Number);
  const va = a.rec.quantiles.values.map(Number);
  let lb = b.rec.quantiles.levels.map(Number);
  let vb = b.rec.quantiles.values.map(Number);

  // Interpolate B onto A's grid if the grids differ.
  const sameGrid =
    la.length === lb.length && la.every((v, i) => Math.abs(v - lb[i]) < 1e-9);
  if (!sameGrid) {
    vb = la.map((lvl) => interp(lvl, lb, vb));
    lb = la;
  }
  // Orient by mean: higher-mean player is positive.
  let hiName, loName, hiV, loV;
  if (b.rec.mean > a.rec.mean) {
    hiName = b.name; loName = a.name; hiV = vb; loV = va;
  } else {
    hiName = a.name; loName = b.name; hiV = va; loV = vb;
  }
  const x = la.map((l) => l * 100);
  const diff = hiV.map((v, i) => v - loV[i]);
  // Expose each player's own point total at every percentile so the hover can
  // show absolute scores (floor/ceiling context), not just the difference.
  return { x, diff, hiName, loName, hiV, loV };
}

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (i < xs.length && xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}
