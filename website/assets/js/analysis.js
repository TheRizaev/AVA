// Ablations (detector, input size, sampling rate vs dev Score A and speed) and the class confusion
// on our dev labels. Data: data/ablations.json (scripts/ablations.py), data/confusion.json
// (scripts/confusion.py).

import { h, fmtInt } from './core.js?v=team2';
import { classInfo, sortClasses, slot } from './palette.js?v=team2';
import { mountChart, axis } from './charts.js?v=team2';

/** Horizontal-bar value labels at the bar tip (text in ink colour, never the series colour). */
const tipLabels = {
  id: 'tipLabels',
  afterDatasetsDraw(chart, _a, o) {
    if (!o || !o.format) return;
    const { ctx } = chart;
    ctx.save();
    ctx.font = `600 12px ${o.font || 'sans-serif'}`;
    ctx.fillStyle = o.color || '#333';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      meta.data.forEach((bar, i) => { const txt = o.format(ds.data[i]); if (txt) ctx.fillText(txt, bar.x + 6, bar.y); });
    });
    ctx.restore();
  },
};

function card(title, sub, ...body) {
  return h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, title), sub ? h('p', { class: 'muted small' }, sub) : null), ...body);
}

function barChart(host, rows, value, fmt, label, axisTitle) {
  return mountChart(host, (t) => ({
    type: 'bar',
    data: {
      labels: rows.map((r) => r.short),
      datasets: [{ data: rows.map(value), backgroundColor: slot(0), borderColor: t.surface, borderWidth: 2, borderSkipped: false, borderRadius: 4, maxBarThickness: 22 }],
    },
    options: {
      indexAxis: 'y',
      layout: { padding: { right: 48 } },
      plugins: {
        legend: { display: false },
        tipLabels: { format: fmt, color: t.text, font: t.font },
        tooltip: { callbacks: { title: (it) => rows[it[0].dataIndex].label, label: (it) => ` ${fmt(it.raw)}` } },
      },
      scales: { x: axis(t, { title: axisTitle, beginAtZero: true }), y: axis(t, { grid: { display: false }, ticks: { color: t.text2 } }) },
    },
    plugins: [tipLabels],
  }), { height: 40 + rows.length * 34, label });
}

function renderAblations(root, abl) {
  const rows = (abl && Array.isArray(abl.variants) ? abl.variants : []).map((v) => ({
    ...v, short: v.label.replace(' (submitted)', '').replace(/ · /g, ' · ') + (v.key === 'baseline' ? ' ★' : ''),
  }));
  if (!rows.length) {
    root.append(card('Ablations', null, h('p', { class: 'note' }, 'The ablation runs are not available yet.')));
    return;
  }
  const base = rows.find((r) => r.key === 'baseline');
  const scoreHost = h('div');
  const speedHost = h('div');
  const classes = sortClasses([...new Set(rows.flatMap((r) => Object.keys(r.per_class || {})))]);
  const table = h('table', { class: 'data-table' },
    h('caption', null, 'Per-class F1 (mean over tIoU 0.3 / 0.5 / 0.7) on our dev labels'),
    h('thead', null, h('tr', null, ['Variant', 'Score A', 'Δ vs submitted', ...classes.map((c) => classInfo(c).label), 'Analysis time'].map((c, i) => h('th', { scope: 'col', class: i ? 'r' : null }, c)))),
    h('tbody', null, rows.map((r) => h('tr', null,
      h('td', { style: { whiteSpace: 'nowrap', fontWeight: r.key === 'baseline' ? '600' : null } }, r.label),
      h('td', { class: 'r' }, r.score_a.toFixed(3)),
      h('td', { class: 'r' }, base && r.key !== 'baseline' ? (r.score_a - base.score_a >= 0 ? '+' : '−') + Math.abs(r.score_a - base.score_a).toFixed(3) : '–'),
      ...classes.map((c) => h('td', { class: 'r' }, r.per_class && c in r.per_class ? r.per_class[c].toFixed(2) : '–')),
      h('td', { class: 'r' }, r.analysis_x_realtime != null ? `${r.analysis_x_realtime.toFixed(2)}× real time` : '–')))));
  root.append(h('div', { class: 'grid grid-2' },
    card('Dev Score A by variant', '★ = submitted configuration', scoreHost),
    card('Analysis time', 'decode + detection + tracking + hazards, × video duration', speedHost)));
  root.append(card('Per-class detail', null, h('div', { class: 'table-scroll' }, table), h('p', { class: 'note' }, abl.note || '')));
  barChart(scoreHost, rows, (r) => r.score_a, (v) => v.toFixed(3), 'Bar chart of dev Score A for each ablation variant.', 'Score A (macro F1)');
  barChart(speedHost, rows, (r) => r.analysis_x_realtime, (v) => `${v.toFixed(2)}×`, 'Bar chart of analysis time relative to video duration for each ablation variant.', '× video duration');
}

function renderConfusion(root, conf) {
  if (!conf || !Array.isArray(conf.classes)) return;
  const cls = conf.classes;
  const max = Math.max(1, ...conf.matrix.flat(), ...conf.missed, ...conf.false_alarm);
  const shade = (n) => (n ? { background: `color-mix(in srgb, var(--accent) ${Math.round(12 + 60 * n / max)}%, transparent)` } : null);
  const cell = (n, extra = {}) => h('td', { class: 'r', style: { ...(shade(n) || {}), ...extra } }, n ? String(n) : '·');
  const SHORT = { congestion: 'Cong.', failure_to_yield: 'Yield', illegal_turn: 'Turn', jaywalking: 'Jaywalk', red_light: 'Red light',
    solid_line_crossing: 'Solid line', stop_line: 'Stop line', stopped_vehicle: 'Stopped', wrong_way: 'Wrong way', accident: 'Accident',
    road_obstacle: 'Obstacle', fire_smoke: 'Fire', near_miss: 'Near miss', illegal_u_turn: 'U-turn' };
  const head = h('tr', null, h('th', { scope: 'col' }, 'Labelled ↓ · predicted →'),
    ...cls.map((c) => h('th', { scope: 'col', class: 'r', title: classInfo(c).label }, SHORT[c] || classInfo(c).label)), h('th', { scope: 'col', class: 'r' }, 'Missed'));
  const body = cls.map((c, i) => h('tr', null, h('th', { scope: 'row' }, classInfo(c).label),
    ...conf.matrix[i].map((n, j) => cell(n, i === j ? { fontWeight: '600' } : {})), cell(conf.missed[i], { color: 'var(--text-2)' })));
  const fa = h('tr', null, h('th', { scope: 'row' }, 'False alarm'), ...conf.false_alarm.map((n) => cell(n, { color: 'var(--text-2)' })), h('td', null, ''));
  const matched = conf.matrix.flat().reduce((a, b) => a + b, 0);
  const cross = conf.matrix.reduce((a, row, i) => a + row.reduce((s, n, j) => s + (i !== j ? n : 0), 0), 0);
  const missed = conf.missed.reduce((a, b) => a + b, 0);
  const falseAlarms = conf.false_alarm.reduce((a, b) => a + b, 0);
  root.append(card('Where the errors are: class confusion on our dev labels',
    `segments matched one-to-one regardless of class at tIoU ≥ ${conf.iou}`,
    h('div', { class: 'table-scroll' }, h('table', { class: 'data-table confusion' }, h('thead', null, head), h('tbody', null, ...body, fa))),
    h('p', { class: 'note' }, `${fmtInt(matched)} labelled events are matched, ${fmtInt(cross)} of them with the wrong class: the rules do not confuse classes. `
      + `The errors are events missed (${missed}) and false alarms (${falseAlarms}), most of them failure_to_yield, whose boundary between `
      + '"drove through while a pedestrian was on the crossing" and "passed behind a pedestrian who had cleared the lane" is also where our two labelling passes disagreed most.')));
}

export function renderAnalysis(root, { ablations, confusion }) {
  root.replaceChildren();
  renderAblations(root, ablations);
  renderConfusion(root, confusion);
}
