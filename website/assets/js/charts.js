// Chart.js loader + a thin wrapper that themes charts from the CSS tokens, rebuilds them on
// theme change, and always renders a data-table twin (which is also the fallback when the
// CDN cannot be reached).

import { h, cssVar, isDark, onThemeChange, prefersReducedMotion } from './core.js?v=team2';

const CHART_URL = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js';
let chartPromise = null;

export function loadChart() {
  if (window.Chart) return Promise.resolve(window.Chart);
  if (!chartPromise) {
    chartPromise = new Promise((resolve) => {
      const sc = document.createElement('script');
      sc.src = CHART_URL;
      sc.async = true;
      sc.crossOrigin = 'anonymous';
      const timer = setTimeout(() => resolve(window.Chart || null), 12000);
      sc.onload = () => { clearTimeout(timer); resolve(window.Chart || null); };
      sc.onerror = () => { clearTimeout(timer); resolve(null); };
      document.head.append(sc);
    });
  }
  return chartPromise;
}

/** Resolved theme values for chart configs. */
export function theme() {
  return {
    dark: isDark(),
    text: cssVar('--text'),
    text2: cssVar('--text-2'),
    muted: cssVar('--axis-ink'),
    grid: cssVar('--grid'),
    axis: cssVar('--axis'),
    surface: cssVar('--chart-surface'),
    tipBg: cssVar('--surface'),
    border: cssVar('--border-strong'),
    accent: cssVar('--accent'),
    critical: cssVar('--critical'),
    font: cssVar('--font') || 'Inter, system-ui, sans-serif',
  };
}

let defaultsApplied = false;
function applyDefaults(Chart) {
  const t = theme();
  Chart.defaults.locale = 'en-US'; // the page is English; do not format numbers in the OS locale
  Chart.defaults.font.family = t.font;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = t.muted;
  Chart.defaults.borderColor = t.grid;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.responsive = true;
  Chart.defaults.animation.duration = prefersReducedMotion() ? 0 : 450;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.boxHeight = 12;
  Chart.defaults.plugins.legend.labels.color = t.text2;
  Chart.defaults.plugins.legend.labels.useBorderRadius = true;
  Chart.defaults.plugins.legend.labels.borderRadius = 3;
  Chart.defaults.plugins.tooltip.backgroundColor = t.tipBg;
  Chart.defaults.plugins.tooltip.titleColor = t.text2;
  Chart.defaults.plugins.tooltip.bodyColor = t.text;
  Chart.defaults.plugins.tooltip.borderColor = t.border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 9;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.boxPadding = 4;
  Chart.defaults.plugins.tooltip.usePointStyle = true;
  Chart.defaults.plugins.tooltip.titleFont = { weight: '500' };
  Chart.defaults.plugins.tooltip.bodyFont = { weight: '600' };
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.line.borderJoinStyle = 'round';
  Chart.defaults.elements.line.borderCapStyle = 'round';
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 5;
  Chart.defaults.elements.point.hoverBorderWidth = 2;
  Chart.defaults.elements.bar.borderRadius = 4;
  defaultsApplied = true;
}

/** Standard axis styling. */
export function axis(t, extra = {}) {
  const { title, ...rest } = extra;
  return {
    grid: { color: t.grid, drawTicks: false, ...(rest.grid || {}) },
    border: { color: t.axis },
    ticks: { color: t.muted, padding: 6, ...(rest.ticks || {}) },
    title: title ? { display: true, text: title, color: t.muted, font: { size: 11.5 } } : { display: false },
    ...Object.fromEntries(Object.entries(rest).filter(([k]) => k !== 'grid' && k !== 'ticks')),
  };
}

/** Build a data-table twin. table = {caption, columns: [...], rows: [[...]]} */
export function tableView(table, open = false) {
  const d = h('details', { class: 'table-view' });
  if (open) d.open = true;
  d.append(h('summary', null, table.summary || 'Show the data as a table'));
  const t = h('table', { class: 'data-table' });
  if (table.caption) t.append(h('caption', null, table.caption));
  t.append(h('thead', null, h('tr', null, table.columns.map((c, i) => h('th', { scope: 'col', class: i ? 'r' : null }, c)))));
  t.append(h('tbody', null, table.rows.map((r) => h('tr', null, r.map((c, i) => h('td', { class: i ? 'r' : null }, c == null ? '–' : c))))));
  d.append(h('div', { class: 'table-scroll' }, t));
  return d;
}

const mounted = new Set();
onThemeChange(() => {
  if (window.Chart) applyDefaults(window.Chart);
  for (const m of mounted) m.rebuild();
});

/**
 * Mount a chart into `host`.
 * build(t) -> Chart.js config, called again after theme changes.
 * opts: {height: 'sm'|'lg'|number, label (aria), table: {...} | () => {...}}
 */
export function mountChart(host, build, opts = {}) {
  const box = h('div', { class: 'chart-box' + (opts.height === 'sm' ? ' h-sm' : opts.height === 'lg' ? ' h-lg' : '') });
  if (typeof opts.height === 'number') box.style.height = opts.height + 'px';
  const canvas = h('canvas', { role: 'img', 'aria-label': opts.label || 'Chart' });
  box.append(canvas);
  host.append(box);
  const table = typeof opts.table === 'function' ? opts.table() : opts.table;
  let tv = null;
  if (table) { tv = tableView(table); host.append(tv); }

  const m = {
    chart: null,
    build,
    rebuild() {
      if (!window.Chart || !box.isConnected) return;
      if (m.chart) { m.chart.destroy(); m.chart = null; }
      try { m.chart = new window.Chart(canvas, m.build(theme())); }
      catch (e) { console.error('chart failed', e); }
    },
    update(newBuild, newTable, newLabel) {
      if (newBuild) m.build = newBuild;
      if (newLabel) canvas.setAttribute('aria-label', newLabel);
      if (newTable && tv) { const fresh = tableView(newTable, tv.open); tv.replaceWith(fresh); tv = fresh; }
      m.rebuild();
    },
    destroy() { if (m.chart) m.chart.destroy(); mounted.delete(m); box.remove(); if (tv) tv.remove(); },
  };
  loadChart().then((Chart) => {
    if (!Chart) {
      box.replaceWith(h('p', { class: 'chart-fallback' }, 'The interactive chart could not load (Chart.js unavailable). The same numbers are in the table below.'));
      if (tv) tv.open = true;
      return;
    }
    if (!defaultsApplied) applyDefaults(Chart);
    mounted.add(m);
    m.rebuild();
  });
  return m;
}

/** Plugin: vertical reference lines at x values (linear x scale). */
export const vlinePlugin = {
  id: 'vlines',
  afterDatasetsDraw(chart, _args, o) {
    const lines = (o && o.lines) || [];
    if (!lines.length) return;
    const { ctx, chartArea } = chart;
    const x = chart.scales.x;
    ctx.save();
    for (const ln of lines) {
      const px = x.getPixelForValue(ln.x);
      if (!Number.isFinite(px) || px < chartArea.left - 1 || px > chartArea.right + 1) continue;
      ctx.strokeStyle = ln.color || o.color || '#888';
      ctx.lineWidth = ln.width || 1.5;
      ctx.setLineDash(ln.dash || []);
      ctx.beginPath(); ctx.moveTo(px, chartArea.top); ctx.lineTo(px, chartArea.bottom); ctx.stroke();
      if (ln.label) {
        ctx.setLineDash([]);
        ctx.fillStyle = ln.textColor || o.textColor || '#666';
        ctx.font = `600 11px ${o.font || 'sans-serif'}`;
        ctx.textAlign = ln.align || 'center';
        ctx.fillText(ln.label, px + (ln.dx || 0), chartArea.top + 11 + (ln.dy || 0));
      }
    }
    ctx.restore();
  },
};
