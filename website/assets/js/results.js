// Results section: video tabs + EventViewer, a dashboard across videos, dev-set metrics
// and the failure cases from report.json.

import { h, fmtTime, fmtInt, isNum, todoText } from './core.js?v=team2';
import { EventViewer, normaliseEvents, normaliseRisk, riskStats, ALARM_THRESHOLD } from './viewer.js?v=team2';
import { mountChart, axis } from './charts.js?v=team2';
import { classInfo, classColor, sortClasses } from './palette.js?v=team2';

export function renderResults(root, results, { eda, metrics, report } = {}) {
  const vids = (results && Array.isArray(results.videos) ? results.videos : []).filter((v) => v && v.name);
  if (!vids.length) { root.replaceChildren(h('p', { class: 'muted' }, 'No results yet.')); return { show() {} }; }
  const edaBy = Object.fromEntries(((eda && eda.videos) || []).map((v) => [v.name, v]));
  const prepared = vids.map((v) => {
    const events = normaliseEvents(v.events, v.duration);
    const risk = normaliseRisk(v.risk);
    return { ...v, ev: events, rk: risk, rs: riskStats(risk), classes: sortClasses([...new Set(events.map((e) => e.label))]) };
  });

  // ---------- tabs + viewer
  const tablist = h('div', { class: 'vtabs', role: 'tablist', 'aria-label': 'Sample videos' });
  const panel = h('div', { class: 'card', role: 'tabpanel', id: 'results-panel', tabindex: '-1' });
  const viewerRoot = h('div');
  panel.append(viewerRoot);
  let viewer = null, current = null;

  const tabs = prepared.map((v, i) => {
    const e = edaBy[v.name];
    const tab = h('button', { type: 'button', class: 'vtab', role: 'tab', id: `vtab-${v.name}`, 'aria-selected': 'false', 'aria-controls': 'results-panel', tabindex: i ? '-1' : '0',
      onclick: () => show(v.name) },
      h('img', { src: `media/${v.name}_thumb.jpg`, alt: '', loading: 'lazy', width: 64, height: 36 }),
      h('span', null, h('span', { class: 'vt-name' }, v.name),
        h('span', { class: 'vt-meta' }, [e && e.lighting, fmtTime(v.duration), `${v.ev.length} events`].filter(Boolean).join(' · '))));
    tab.addEventListener('keydown', (ev) => {
      const k = ev.key;
      if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(k)) return;
      ev.preventDefault();
      let j = k === 'Home' ? 0 : k === 'End' ? prepared.length - 1 : (i + (k === 'ArrowRight' ? 1 : -1) + prepared.length) % prepared.length;
      tabs[j].focus();
      show(prepared[j].name);
    });
    tablist.append(tab);
    return tab;
  });

  function show(name, t, { scroll = false, updateHash = true } = {}) {
    const v = prepared.find((x) => x.name === name) || prepared[0];
    if (!current || current.name !== v.name) {
      if (viewer) viewer.destroy();
      current = v;
      tabs.forEach((tb, i) => { const sel = prepared[i].name === v.name; tb.setAttribute('aria-selected', String(sel)); tb.tabIndex = sel ? 0 : -1; });
      panel.setAttribute('aria-labelledby', `vtab-${v.name}`);
      viewer = new EventViewer(viewerRoot, {
        duration: v.duration, events: v.events, risk: v.risk,
        signal: edaBy[v.name] && edaBy[v.name].signal,
        videoSrc: v.video || `media/${v.name}_annotated.mp4`,
        poster: `media/${v.name}_thumb.jpg`,
        videoLabel: v.name,
        noVideoTitle: 'Annotated video not published yet',
        noVideoText: `${v.video || `media/${v.name}_annotated.mp4`} is not on the server yet. The timeline, risk curve and table below still work: click them to move the playhead.`,
      });
      if (updateHash && history.replaceState) { try { history.replaceState(null, '', `#results/${v.name}`); } catch { /* ignore */ } }
    }
    if (isNum(t)) viewer.seek(t);
    if (scroll) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------- dashboard
  const dash = h('article', { class: 'card' });
  const allClasses = sortClasses([...new Set(prepared.flatMap((v) => v.classes))]);
  const chartBox = h('div');
  dash.append(h('div', { class: 'card-head' }, h('h3', null, 'Dashboard: events per class, per video'), h('p', { class: 'muted small' }, 'Hover a segment for its count · click a legend entry to hide a class')), chartBox);
  mountChart(chartBox, (t) => ({
    type: 'bar',
    data: {
      labels: prepared.map((v) => v.name),
      datasets: allClasses.map((c) => ({ label: classInfo(c).label, data: prepared.map((v) => v.ev.filter((e) => e.label === c).length), backgroundColor: classColor(c), borderColor: t.surface, borderWidth: { right: 2 }, borderSkipped: false, borderRadius: 3, maxBarThickness: 24 })),
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { position: 'bottom', labels: { padding: 12 } }, tooltip: { mode: 'nearest', intersect: true, callbacks: { label: (it) => ` ${it.dataset.label}: ${it.raw}` } } },
      scales: { x: axis(t, { stacked: true, title: 'events' }), y: axis(t, { stacked: true, grid: { display: false }, ticks: { color: t.text2, font: { weight: '600' } } }) },
    },
  }), { height: 300, label: 'Stacked bar chart of detected events per class for each sample video.',
    table: { summary: 'Show the counts as a table', columns: ['Video', ...allClasses.map((c) => classInfo(c).label), 'Total'],
      rows: prepared.map((v) => [v.name, ...allClasses.map((c) => v.ev.filter((e) => e.label === c).length), v.ev.length]) } });

  const sumTable = h('table', { class: 'data-table' },
    h('thead', null, h('tr', null, ['Video', 'Duration', 'Events', 'Events / min', 'Classes', 'Peak risk', `Alarms (≥ ${ALARM_THRESHOLD})`].map((c, i) => h('th', { scope: 'col', class: i ? 'r' : null }, c)))),
    h('tbody', null, prepared.map((v) => h('tr', null,
      h('td', null, h('button', { type: 'button', class: 'seek-btn', style: { all: 'unset', cursor: 'pointer', color: 'var(--accent)', fontWeight: '600' }, onclick: () => show(v.name, undefined, { scroll: true }) }, v.name)),
      h('td', { class: 'r' }, fmtTime(v.duration)),
      h('td', { class: 'r' }, fmtInt(v.ev.length)),
      h('td', { class: 'r' }, (v.ev.length / (v.duration / 60)).toFixed(1)),
      h('td', { class: 'r' }, String(v.classes.length)),
      h('td', { class: 'r', style: { whiteSpace: 'nowrap' } }, v.rs.peak ? `${v.rs.peak[1].toFixed(2)} at ${fmtTime(v.rs.peak[0])}` : '–'),
      h('td', { class: 'r' }, String(v.rs.alarms.length))))));
  dash.append(h('div', { class: 'table-scroll', style: { marginTop: '16px' } }, sumTable),
    h('p', { class: 'note' }, 'As far as we know the sample videos contain no accident, so alarms and accident segments on them are false positives. The risk scale was calibrated on these clips for about 0.2 false alarms per minute.'));

  // ---------- metrics
  const met = renderMetrics(metrics);

  // ---------- failure cases
  const fails = (report && Array.isArray(report.failures)) ? report.failures.filter((f) => f && (f.title || f.text)) : [];
  const failCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, 'Failure cases'), h('p', { class: 'muted small' }, 'Click to watch the moment')));
  if (fails.length) {
    failCard.append(h('div', { class: 'fail-grid' }, fails.map((f) => {
      const known = prepared.find((v) => v.name === f.video);
      return h('div', { class: 'fail-card' }, h('h4', null, ...todoText(f.title || 'Untitled')), h('p', null, ...todoText(f.text || '')),
        known ? h('button', { type: 'button', class: 'btn btn-ghost btn-sm', onclick: () => show(known.name, Number(f.t) || 0, { scroll: true }) },
          `Watch ${known.name} at ${fmtTime(Number(f.t) || 0)}`) : null);
    })));
  } else failCard.append(h('p', { class: 'muted' }, 'No failure cases listed yet (data/report.json → failures).'));

  root.replaceChildren(tablist, panel, dash, met, failCard);

  // initial video: from the hash (#results/C3897 or #results/C3897/86) or the first one
  const m = /^#results\/([^/]+)(?:\/([\d.]+))?$/.exec(location.hash || '');
  if (m && prepared.some((v) => v.name === decodeURIComponent(m[1]))) {
    show(decodeURIComponent(m[1]), m[2] ? Number(m[2]) : undefined, { updateHash: false });
    requestAnimationFrame(() => panel.scrollIntoView({ block: 'start' }));
  } else show(prepared[0].name, undefined, { updateHash: false });

  return { show, videos: prepared };
}

function f1Of(x) {
  if (isNum(x)) return x;
  if (x && isNum(x.f1)) return x.f1;
  return null;
}
function pick(pc, keys) { for (const k of keys) if (pc && pc[k] != null) return pc[k]; return null; }

function renderMetrics(metrics) {
  const card = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, 'Dev-set metrics'), h('p', { class: 'muted small' }, 'official evaluate.py')));
  const m = metrics || {};
  const fmt = (v) => (isNum(v) ? v.toFixed(3) : '–');
  if (!isNum(m.score_a)) {
    card.append(h('div', { class: 'metric-pending' }, h('div', null, h('b', null, 'Dev-set evaluation pending. '),
      'Per-class F1 at tIoU 0.3 / 0.5 / 0.7, their mean and Score A will appear here as soon as data/metrics.json is filled.')));
    if (m.dataset && !/^TODO/i.test(m.dataset)) card.append(h('p', { class: 'note' }, m.dataset));
    return card;
  }
  card.append(h('dl', { class: 'score-row' },
    h('div', { class: 'stat' }, h('dt', null, 'Score A'), h('dd', null, fmt(m.score_a))),
    isNum(m.score_b) ? h('div', { class: 'stat' }, h('dt', null, 'Score B'), h('dd', null, fmt(m.score_b))) : null,
    isNum(m.model_score) ? h('div', { class: 'stat' }, h('dt', null, 'Model score M'), h('dd', null, fmt(m.model_score))) : null));
  const pcs = m.per_class && typeof m.per_class === 'object' ? m.per_class : {};
  const classes = sortClasses(Object.keys(pcs));
  if (classes.length) {
    const rows = classes.map((c) => {
      const pc = pcs[c] || {};
      const f3 = f1Of(pick(pc, ['0.3', 'f1_0.3', 'f1@0.3']));
      const f5 = f1Of(pick(pc, ['0.5', 'f1_0.5', 'f1@0.5']));
      const f7 = f1Of(pick(pc, ['0.7', 'f1_0.7', 'f1@0.7']));
      let mean = pick(pc, ['f1_mean', 'mean']);
      if (!isNum(mean) && [f3, f5, f7].every(isNum)) mean = (f3 + f5 + f7) / 3;
      const at5 = pick(pc, ['0.5']);
      const tpfpfn = at5 && isNum(at5.tp) ? `${at5.tp}/${at5.fp}/${at5.fn}` : '–';
      return { c, f3, f5, f7, mean, tpfpfn };
    });
    card.append(h('div', { class: 'table-scroll' }, h('table', { class: 'data-table' },
      h('thead', null, h('tr', null, ['Class', 'F1@0.3', 'F1@0.5', 'F1@0.7', 'Mean', 'TP/FP/FN @0.5'].map((c, i) => h('th', { scope: 'col', class: i ? 'r' : null }, c)))),
      h('tbody', null, rows.map((r) => h('tr', null,
        h('td', null, h('span', { class: 'cls-cell', style: { display: 'inline-flex', gap: '8px', alignItems: 'center' } }, h('span', { class: 'swatch', style: { background: `var(--cls-${r.c}, var(--muted))` } }), classInfo(r.c).label)),
        h('td', { class: 'r' }, fmt(r.f3)), h('td', { class: 'r' }, fmt(r.f5)), h('td', { class: 'r' }, fmt(r.f7)),
        h('td', { class: 'r' }, h('strong', null, fmt(r.mean))), h('td', { class: 'r' }, r.tpfpfn)))))));
  }
  if (m.note && !/pending/i.test(m.note)) card.append(h('p', { class: 'note' }, ...todoText(m.note)));
  if (m.dataset && !/^TODO/i.test(m.dataset)) card.append(h('p', { class: 'note' }, m.dataset));
  return card;
}
