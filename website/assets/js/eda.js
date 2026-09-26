// EDA section: footage facts, findings that shaped the solution, and a per-video explorer.

import { h, s, fmtTime, fmtInt, isNum, median, sum } from './core.js?v=team2';
import { mountChart, axis, vlinePlugin } from './charts.js?v=team2';
import { SIGNAL, OBJECTS, KINDS, slot } from './palette.js?v=team2';
import { normaliseSignal } from './viewer.js?v=team2';

const DECODE = { full: 0.60, ref: 0.33 }; // CPU decode wall-time per second of video (scripts/bench_decode.py)

export function alpha(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
const r1 = (v) => (isNum(v) ? Math.round(v * 10) / 10 : v);
const fmtT = (v) => fmtTime(Number(v));

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
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      meta.data.forEach((bar, i) => {
        const v = ds.data[i];
        const txt = o.format(v, i, di);
        if (!txt) return;
        ctx.textAlign = 'left';
        ctx.fillText(txt, bar.x + 6, bar.y);
      });
    });
    ctx.restore();
  },
};

// ---------------------------------------------------------------- derived numbers
function signalPlan(v) {
  const ph = normaliseSignal(v.signal);
  const red = [], green = [], yellow = [], onsets = [];
  for (let i = 0; i < ph.length; i++) {
    const [a, b, st] = ph[i];
    const d = b - a + 0.1;
    const inner = i > 0 && i < ph.length - 1;
    if (inner && st === 'red') red.push(d);
    if (inner && st === 'yellow') yellow.push(d);
    if (inner && st === 'green' && ph[i - 1][2] === 'red' && ph[i + 1][2] === 'yellow') green.push(d);
    if (i > 0 && st === 'green' && ph[i - 1][2] === 'red') onsets.push(a);
  }
  const cycles = onsets.slice(1).map((t, i) => t - onsets[i]);
  const flashing = ph.some((p, i) => p[2] === 'unknown' && i > 0 && ph[i - 1][2] === 'green');
  return { red: median(red), green: median(green), yellow: median(yellow), cycle: median(cycles), flashing };
}

function regStats(v) {
  const R = v.registration || {};
  const t = R.t || [], dx = R.dx || [], dy = R.dy || [];
  const mdx = median(dx), mdy = median(dy);
  let drift = 0;
  for (let i = 0; i < t.length; i++) {
    if (t[i] > 30) break;
    drift = Math.max(drift, Math.hypot(dx[i] - mdx, dy[i] - mdy));
  }
  return { dx: mdx, dy: mdy, shift: Math.hypot(mdx, mdy), drift };
}

function brightStats(v) {
  const m = (v.brightness && v.brightness.mean || []).filter(isNum);
  if (!m.length) return { min: NaN, max: NaN, mean: NaN };
  return { min: Math.min(...m), max: Math.max(...m), mean: sum(m) / m.length };
}

const tracksTotal = (v) => sum(Object.values(v.tracks_by_kind || {}));

// ---------------------------------------------------------------- building blocks
function finding(num, title, { span2 = false } = {}, ...body) {
  return h('article', { class: 'card finding' + (span2 ? ' span-2' : ''), 'data-order': num },
    h('p', { class: 'finding-num' }, `Finding ${num}`), h('h3', null, title), ...body);
}
function soWhat(saw, did) {
  return h('div', { class: 'so' },
    h('div', null, h('b', null, 'What we saw'), h('p', null, saw)),
    h('div', { class: 'did' }, h('b', null, 'What we did'), h('p', null, did)));
}
function chartHost(title, sub) {
  const host = h('div', { class: 'chart-host' });
  return { el: h('div', null, title ? h('p', { class: 'chart-title' }, title) : null, sub ? h('p', { class: 'chart-sub' }, sub) : null, host), host };
}
export function imageFigure(src, alt, caption, lightbox) {
  const img = h('img', { src, alt, loading: 'lazy', decoding: 'async', width: 1920, height: 1080 });
  const btn = h('button', { type: 'button', 'aria-label': `Enlarge: ${alt}`, onclick: () => lightbox && lightbox(src, alt, caption) }, img);
  img.addEventListener('error', () => { btn.replaceWith(h('div', { class: 'chart-fallback' }, `Image not found: ${src}`)); }, { once: true });
  return h('figure', { class: 'img-fig' }, btn, h('figcaption', null, caption));
}
function observeWidth(el, cb) {
  let last = 0;
  const run = () => { const w = Math.floor(el.clientWidth); if (w && Math.abs(w - last) > 2) { last = w; cb(w); } };
  if ('ResizeObserver' in window) new ResizeObserver(() => requestAnimationFrame(run)).observe(el);
  else window.addEventListener('resize', run);
  requestAnimationFrame(run);
}

// ---------------------------------------------------------------- signal chart (SVG)
function signalChart(vids) {
  const host = h('div', { class: 'signal-host' });
  const maxDur = Math.max(...vids.map((v) => v.duration || 0), 1);
  const draw = (W) => {
    const narrow = W < 520;
    const L = narrow ? 52 : 70, R = 10, rowH = 26, top = 4;
    const H = top + vids.length * rowH + 22;
    const x = (t) => L + (t / maxDur) * (W - L - R);
    const svg = s('svg', { class: 'signal-svg', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': 'Main-road signal phase over time for each sample video: alternating red, green and short yellow phases.' });
    const step = maxDur > 240 ? 60 : 30;
    const ax = s('g', { class: 'sig-axis' });
    for (let t = 0; t <= maxDur + 0.01; t += step) {
      ax.append(s('line', { class: 'sig-grid', x1: x(t), x2: x(t), y1: top, y2: top + vids.length * rowH }));
      ax.append(s('text', { x: x(t), y: H - 6, 'text-anchor': t === 0 ? 'start' : 'middle' }, fmtTime(t)));
    }
    svg.append(ax);
    vids.forEach((v, i) => {
      const y = top + i * rowH;
      svg.append(s('text', { x: L - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end' }, v.name));
      for (const [a, b, st] of normaliseSignal(v.signal)) {
        const rect = s('rect', { class: SIGNAL[st].cls, x: x(a), y: y + 5, width: Math.max(1, x(b + 0.1) - x(a)), height: rowH - 10, rx: 2 });
        rect.append(s('title', null, `${v.name}: ${SIGNAL[st].label.toLowerCase()} ${fmtTime(a)}–${fmtTime(b)}`));
        svg.append(rect);
      }
    });
    host.replaceChildren(svg);
  };
  observeWidth(host, draw);
  const legend = h('ul', { class: 'legend', 'aria-label': 'Signal colours' },
    ...Object.values(SIGNAL).map((sg) => h('li', null, h('span', { class: 'swatch', style: { background: `var(--${sg.cls})` } }), sg.label)));
  return h('div', null, host, legend);
}

// ---------------------------------------------------------------- main
export function renderEDA(root, eda, lightbox) {
  const vids = (eda && Array.isArray(eda.videos)) ? eda.videos : [];
  if (!vids.length) { root.replaceChildren(h('p', { class: 'muted' }, 'No EDA data yet.')); return; }
  const out = [];

  // ---------- tiles
  const totalDur = sum(vids.map((v) => v.duration));
  const bitrate = median(vids.map((v) => v.bitrate_mbps));
  const kinds = {};
  for (const v of vids) for (const [k, n] of Object.entries(v.tracks_by_kind || {})) kinds[k] = (kinds[k] || 0) + n;
  const plans = vids.map((v) => ({ name: v.name, ...signalPlan(v) }));
  const cycles = plans.map((p) => p.cycle).filter(isNum);
  const cycleTxt = cycles.length ? (Math.round(Math.min(...cycles)) === Math.round(Math.max(...cycles)) ? `${Math.round(cycles[0])} s` : `${Math.round(Math.min(...cycles))}–${Math.round(Math.max(...cycles))} s`) : '–';
  const v0 = vids[0];
  out.push(h('dl', { class: 'eda-tiles' },
    tile('Footage analysed', `${(totalDur / 60).toFixed(1)}`, 'min', `${vids.length} clips · ${v0.width}×${v0.height} @ ${v0.fps} fps`),
    tile('Bitrate', `${Math.round(bitrate)}`, 'Mb/s', `${v0.codec}, ${/10/.test(v0.pix_fmt) ? '10-bit' : ''} ${/422/.test(v0.pix_fmt) ? '4:2:2' : ''}`),
    tile('Road-user tracks', fmtInt(sum(Object.values(kinds))), '', `${fmtInt(kinds.vehicle || 0)} vehicles · ${fmtInt(kinds.person || 0)} pedestrians`),
    tile('Signal cycle', cycleTxt, '', 'fixed-time plan, read from the lamps')));

  // ---------- footage table
  const fmtOf = (v) => `${v.width}×${v.height} · ${v.fps} fps · ${v.codec} · ${v.pix_fmt}`;
  const sameFormat = vids.every((v) => fmtOf(v) === fmtOf(v0));
  const cols = ['Video', 'Lighting', 'Duration', 'Frames', ...(sameFormat ? [] : ['Format']), 'Bitrate', 'Vehicle / person tracks', 'Vehicles through stop line'];
  const tbl = h('table', { class: 'data-table' },
    h('caption', null, (sameFormat ? `All four clips: ${fmtOf(v0)} (ffprobe). ` : 'The sample clips (ffprobe). ') + 'Tracks are tracker IDs, so one person can count twice after an occlusion.'),
    h('thead', null, h('tr', null, cols.map((c, i) => h('th', { scope: 'col', class: i > 1 ? 'r' : null }, c)))),
    h('tbody', null, vids.map((v) => h('tr', null,
      h('td', null, h('strong', null, v.name)), h('td', null, v.lighting || '–'),
      h('td', { class: 'r' }, fmtTime(v.duration)), h('td', { class: 'r' }, fmtInt(v.frames)),
      sameFormat ? null : h('td', { class: 'r' }, fmtOf(v)),
      h('td', { class: 'r' }, `${v.bitrate_mbps} Mb/s`),
      h('td', { class: 'r' }, `${fmtInt((v.tracks_by_kind || {}).vehicle)} / ${fmtInt((v.tracks_by_kind || {}).person)}`),
      h('td', { class: 'r' }, v.throughput ? `${v.throughput.total} (${(v.throughput.total / v.duration * 60).toFixed(0)} / min)` : '–')))));
  out.push(h('div', { class: 'card' }, h('div', { class: 'table-scroll' }, tbl)));

  // ---------- findings
  const grid = h('div', { class: 'finding-grid' });
  out.push(grid);

  // 1. decode
  {
    const c = chartHost('CPU decode time per second of video', 'lower is better · measured with scripts/bench_decode.py');
    grid.append(finding(1, 'Decoding 4K 10-bit 4:2:2 is the bottleneck', {}, c.el,
      soWhat(`The clips are ${v0.codec}, ${v0.pix_fmt}, ~${Math.round(bitrate)} Mb/s. NVDEC cannot decode 4:2:2 10-bit, so decoding runs on the CPU, and a full-resolution decode alone costs about ${DECODE.full} s per second of video.`,
        `Decode only the reference frames of the IBBP GOP (every 3rd frame ≈ 10 fps, ≈${DECODE.ref} s per second), convert straight to 1920×1080, and decode in a background thread that overlaps GPU inference.`)));
    mountChart(c.host, (t) => ({
      type: 'bar',
      data: { labels: ['Every frame, full res', 'Reference frames only'], datasets: [{ label: 'seconds per video-second', data: [DECODE.full, DECODE.ref], backgroundColor: [t.dark ? '#5a5953' : '#c3c2b7', t.accent], maxBarThickness: 24, borderRadius: 4, borderSkipped: 'start' }] },
      options: { indexAxis: 'y', layout: { padding: { right: 44 } }, plugins: { legend: { display: false }, tipLabels: { format: (v) => `${v.toFixed(2)} s`, color: t.text, font: t.font }, tooltip: { callbacks: { label: (it) => ` ${it.raw.toFixed(2)} s of CPU time per second of video` } } },
        scales: { x: axis(t, { min: 0, max: 0.8, title: 'seconds of decode per second of video' }), y: axis(t, { grid: { display: false }, ticks: { color: t.text2 } }) } },
      plugins: [tipLabels],
    }), { height: 150, label: 'Bar chart: full decode 0.60 s per second of video, reference frames only 0.33 s.',
      table: { columns: ['Decode mode', 's per video-second'], rows: [['Every frame, full resolution', DECODE.full.toFixed(2)], ['Reference frames only', DECODE.ref.toFixed(2)]] } });
  }

  // 2. camera shift and drift
  {
    const regs = vids.map((v) => ({ name: v.name, ...regStats(v) }));
    const moved = regs.filter((r) => r.shift > 5).sort((a, b) => b.shift - a.shift);
    const drifter = regs.slice().sort((a, b) => b.drift - a.drift)[0];
    const c1 = chartHost('Median offset of each clip vs the reference frame', 'px in the 1920×1080 frame');
    const c2 = chartHost(`${drifter.name}: offset during the first two minutes`, 'registration every 5 s');
    const saw = (moved.length
      ? `The camera was re-mounted between recordings: ${moved.map((r) => `${r.name} sits (${r.dx >= 0 ? '+' : ''}${r.dx.toFixed(0)}, ${r.dy >= 0 ? '+' : ''}${r.dy.toFixed(0)}) px`).join(', ')} away from the reference. `
      : '') + `Inside a clip it also drifts: ${drifter.name} moves ${drifter.drift.toFixed(1)} px during its first 30 s before settling.`;
    grid.append(finding(3, 'The camera moves between and within clips', { span2: true },
      h('div', { class: 'two-charts' }, c1.el, c2.el),
      soWhat(saw, 'Register every 5 s with SIFT + RANSAC to one reference frame and keep all geometry (layout, tracks, flow field) there. A hand-drawn layout would otherwise be off by a whole lane.')));
    mountChart(c1.host, (t) => ({
      type: 'bar',
      data: { labels: regs.map((r) => r.name), datasets: [
        { label: 'dx (horizontal)', data: regs.map((r) => r1(r.dx)), backgroundColor: slot(0), maxBarThickness: 24, borderRadius: 4 },
        { label: 'dy (vertical)', data: regs.map((r) => r1(r.dy)), backgroundColor: slot(1), maxBarThickness: 24, borderRadius: 4 }] },
      options: { plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${it.raw} px` } } },
        scales: { x: axis(t, { grid: { display: false } }), y: axis(t, { title: 'px vs reference', grid: { color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? t.axis : t.grid) } }) } },
    }), { label: 'Grouped bar chart of the median registration offset per clip.',
      table: { columns: ['Video', 'dx (px)', 'dy (px)', 'drift in first 30 s (px)'], rows: regs.map((r) => [r.name, r.dx.toFixed(1), r.dy.toFixed(1), r.drift.toFixed(1)]) } });
    const dv = vids.find((v) => v.name === drifter.name);
    const R = dv.registration;
    const idx = R.t.map((tt, i) => i).filter((i) => R.t[i] <= 120);
    mountChart(c2.host, (t) => ({
      type: 'line',
      data: { datasets: [
        { label: 'dx', data: idx.map((i) => ({ x: R.t[i], y: R.dx[i] })), borderColor: slot(0), backgroundColor: slot(0), pointRadius: 2.5 },
        { label: 'dy', data: idx.map((i) => ({ x: R.t[i], y: R.dy[i] })), borderColor: slot(1), backgroundColor: slot(1), pointRadius: 2.5 }] },
      options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { title: (it) => fmtT(it[0].parsed.x), label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(2)} px` } } },
        scales: { x: axis(t, { type: 'linear', min: 0, max: 120, ticks: { callback: fmtT, stepSize: 30 }, title: 'time in clip' }), y: axis(t, { title: 'px' }) } },
    }), { label: `Line chart of ${drifter.name} registration offset over its first two minutes.`,
      table: { columns: ['t', 'dx', 'dy'], rows: idx.map((i) => [fmtTime(R.t[i]), R.dx[i].toFixed(2), R.dy[i].toFixed(2)]) } });
  }

  // 3. lighting
  {
    const bs = vids.map((v) => ({ name: v.name, lighting: v.lighting, ...brightStats(v) }));
    const c = chartHost('Mean frame brightness during each clip', 'bar = min to max over the clip · grey level 0–255');
    const lo = bs.slice().sort((a, b) => a.mean - b.mean)[0], hi = bs.slice().sort((a, b) => b.mean - a.mean)[0];
    grid.append(finding(2, 'Lighting runs from noon to dusk', {}, c.el,
      soWhat(`Mean brightness falls from about ${Math.round(hi.mean)} (${hi.name}, ${hi.lighting}) to ${Math.round(lo.min)}–${Math.round(lo.max)} (${lo.name}, ${lo.lighting}). Lamp colours, shadows and SIFT features all change with it.`,
        'Register against a bank of three lighting references (noon, evening sun, dusk) and normalise each signal lamp separately instead of using fixed colour thresholds.')));
    mountChart(c.host, (t) => ({
      type: 'bar',
      data: { labels: bs.map((b) => `${b.name} · ${b.lighting}`), datasets: [{ label: 'brightness range', data: bs.map((b) => [r1(b.min), r1(b.max)]), backgroundColor: slot(0), maxBarThickness: 22, borderRadius: 4, borderSkipped: false }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: (it) => { const b = bs[it.dataIndex]; return ` ${b.min.toFixed(0)}–${b.max.toFixed(0)}, mean ${b.mean.toFixed(0)}`; } } } },
        scales: { x: axis(t, { min: 0, max: 120, title: 'mean grey level' }), y: axis(t, { grid: { display: false }, ticks: { color: t.text2 } }) } },
    }), { height: 200, label: 'Floating bar chart of the brightness range of each clip.',
      table: { columns: ['Video', 'Lighting', 'Min', 'Mean', 'Max'], rows: bs.map((b) => [b.name, b.lighting, b.min.toFixed(1), b.mean.toFixed(1), b.max.toFixed(1)]) } });
  }

  // 4. lanes
  {
    const lh = eda.lane_hist;
    if (lh && Array.isArray(lh.angle)) {
      const peaks = countLanes(lh.angle, lh.count, lh.lines);
      const c = chartHost('Approach vehicles by lane angle', 'angle of the ground point seen from the vanishing point · red lines: solid dividers');
      grid.append(finding(5, 'Lanes become clean peaks in vanishing-point angle', { span2: true }, h('div', { class: 'two-charts' }, c.el,
        imageFigure('media/lane_groups.jpg', 'Approach lanes coloured by lane-angle group, with the mean angle of each group.', 'Lane groups on the reference frame. Lane lines are rays from the vanishing point, so one angle describes one lane at any distance.', lightbox)),
        soWhat(`In pixel coordinates the lanes converge; as angles from the vanishing point the approach traffic forms ${peaks} clear peaks, one per lane, and the ${(lh.lines || []).length} solid dividers fall in the valleys between them.`,
          'solid_line_crossing is a 1-D test: the vehicle\'s smoothed angle crosses a divider angle and settles on the other side (hysteresis). The same angle gives lane identity for rear-end risk in Part B.')));
      mountChart(c.host, (t) => ({
        type: 'bar',
        data: { datasets: [{ label: 'track points', data: lh.angle.map((a, i) => ({ x: a, y: lh.count[i] })), backgroundColor: slot(0), barPercentage: 0.9, categoryPercentage: 1, borderRadius: 2 }] },
        options: { plugins: { legend: { display: false }, vlines: { lines: (lh.lines || []).map((a) => ({ x: a, color: t.critical, dash: [4, 3] })), font: t.font },
          tooltip: { callbacks: { title: (it) => `${it[0].parsed.x.toFixed(2)}°`, label: (it) => ` ${fmtInt(it.parsed.y)} track points` } } },
          scales: { x: axis(t, { type: 'linear', min: 20, max: 36, offset: false, ticks: { stepSize: 2, callback: (v) => `${v}°` }, title: 'lane angle', grid: { display: false } }), y: axis(t, { title: 'track points' }) } },
        plugins: [vlinePlugin],
      }), { label: `Histogram of approach-vehicle lane angles with ${peaks} peaks and the solid dividers marked.`,
        table: { columns: ['Angle (°)', 'Track points'], rows: lh.angle.map((a, i) => [a.toFixed(2), fmtInt(lh.count[i])]).concat((lh.lines || []).map((a) => [`${a} (solid divider)`, '–'])) } });
    }
  }

  // 5. signal
  {
    const planTxt = plans.map((p) => `${p.name} ${isNum(p.cycle) ? Math.round(p.cycle) + ' s' : '–'}`).join(', ');
    const med = (k) => median(plans.map((p) => p[k]));
    const flashing = plans.filter((p) => p.flashing).map((p) => p.name);
    grid.append(finding(4, 'A fixed-time signal plan, readable from the lamps', { span2: true },
      h('p', { class: 'chart-title' }, 'Main-road signal phase, per clip'),
      signalChart(vids),
      h('div', { class: 'table-scroll' }, h('table', { class: 'data-table', style: { marginTop: '12px' } },
        h('thead', null, h('tr', null, ['Video', 'Cycle', 'Red', 'Green', 'Yellow'].map((c, i) => h('th', { scope: 'col', class: i ? 'r' : null }, c)))),
        h('tbody', null, plans.map((p) => h('tr', null, h('td', null, p.name), ...['cycle', 'red', 'green', 'yellow'].map((k) => h('td', { class: 'r' }, isNum(p[k]) ? `${p[k].toFixed(0)} s` : '–'))))))),
      soWhat(`Cycle per clip: ${planTxt} (median red ${med('red').toFixed(0)} s, green ${med('green').toFixed(0)} s, yellow ${med('yellow').toFixed(0)} s). ${flashing.length ? `In ${flashing.join(', ')} the green flashes before yellow, which the reader sees as green/unknown flicker. ` : ''}Approach vehicles start crossing the stop line about 2 s after green.`,
        'Read the phase from the three lamps of the vehicle head on the median tip (a pedestrian head in phase is the fallback). red_light and stop_line use the actual phase; approach queues that release on green are never stopped vehicles or congestion.')));
  }

  // 6. composition
  {
    const kindsPresent = KINDS.filter((k) => vids.some((v) => (v.tracks_by_kind || {})[k.id]));
    const c = chartHost('Tracks per clip by road-user type');
    const ratio = vids.map((v) => ((v.tracks_by_kind || {}).person || 0) / Math.max(1, (v.tracks_by_kind || {}).vehicle || 0));
    grid.append(finding(6, 'Pedestrians outnumber vehicles', {}, c.el,
      h('ul', { class: 'kv' }, vids.map((v) => h('li', null, h('b', null, v.name), ` ${v.throughput ? v.throughput.total : '–'} vehicles through the stop line`))),
      soWhat(`Every clip has ${Math.min(...ratio).toFixed(1)}–${Math.max(...ratio).toFixed(1)}× more pedestrian tracks than vehicle tracks. Cyclists are rare (${fmtInt(kinds.bicycle || 0)} tracks in total), and riders are detected as a person on top of a two-wheeler.`,
        'Pedestrian rules (jaywalking, failure to yield) get the most material. A person whose box overlaps a bicycle or motorcycle for a large part of the track is treated as a rider, not a pedestrian.')));
    mountChart(c.host, (t) => ({
      type: 'bar',
      data: { labels: vids.map((v) => v.name), datasets: kindsPresent.map((k) => ({ label: k.label, data: vids.map((v) => (v.tracks_by_kind || {})[k.id] || 0), backgroundColor: slot(k.slot), maxBarThickness: 22, borderColor: t.surface, borderWidth: { right: 2 }, borderSkipped: false, borderRadius: 3 })) },
      options: { indexAxis: 'y', plugins: { legend: { position: 'bottom' }, tooltip: { mode: 'index', callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtInt(it.raw)}` } } },
        scales: { x: axis(t, { stacked: true, title: 'tracks' }), y: axis(t, { stacked: true, grid: { display: false }, ticks: { color: t.text2 } }) } },
    }), { height: 220, label: 'Stacked bar chart of track counts per clip by road-user type.',
      table: { columns: ['Video', ...kindsPresent.map((k) => k.label)], rows: vids.map((v) => [v.name, ...kindsPresent.map((k) => fmtInt((v.tracks_by_kind || {})[k.id] || 0))]) } });
  }

  // 7. speeds
  {
    const bins = (vids[0].speeds && vids[0].speeds.bin_px_s) || [];
    const all = bins.map((_, i) => sum(vids.map((v) => (v.speeds && v.speeds.count ? v.speeds.count[i] : 0))));
    const medians = vids.map((v) => v.speeds && v.speeds.median).filter(isNum);
    const c = chartHost('Vehicle speed, all clips', 'px/s in the 1080p reference frame, 0.5-s smoothing, moving vehicles only');
    grid.append(finding(7, 'Speeds are only known in pixels', {}, c.el,
      soWhat(`Median vehicle speed per clip is ${Math.min(...medians).toFixed(0)}–${Math.max(...medians).toFixed(0)} px/s, with a tail to ~${bins[bins.length - 1] || 650} px/s. There is no metric calibration, and perspective makes a pixel near the camera worth less road than one far away.`,
        'All thresholds are in px/s of the reference frame (moving > 25–60 px/s, abrupt stop from ≥ 150 px/s to < 25 px/s in 0.7 s) and were tuned on the samples; a new camera would need re-tuning.')));
    const binW = bins.length > 1 ? bins[1] - bins[0] : 25;
    mountChart(c.host, (t) => ({
      type: 'bar',
      data: { datasets: [{ label: 'observations', data: bins.map((b, i) => ({ x: b + binW / 2, y: all[i] })), backgroundColor: slot(0), barPercentage: 0.9, categoryPercentage: 1, borderRadius: 2 }] },
      options: { plugins: { legend: { display: false }, vlines: { lines: [{ x: median(medians), color: t.text2, dash: [4, 3], label: `median of clips ≈ ${median(medians).toFixed(0)}`, align: 'left', dx: 6 }], textColor: t.text2, font: t.font },
        tooltip: { callbacks: { title: (it) => `${(it[0].parsed.x - binW / 2).toFixed(0)}–${(it[0].parsed.x + binW / 2).toFixed(0)} px/s`, label: (it) => ` ${fmtInt(it.parsed.y)} observations` } } },
        scales: { x: axis(t, { type: 'linear', min: 0, max: (bins[bins.length - 1] || 650) + binW, title: 'px/s', grid: { display: false } }), y: axis(t, { title: 'observations' }) } },
      plugins: [vlinePlugin],
    }), { label: 'Histogram of vehicle speeds in pixels per second.',
      table: { columns: ['Speed bin (px/s)', 'Observations'], rows: bins.map((b, i) => [`${b}–${b + binW}`, fmtInt(all[i])]) } });
  }

  // 8. pedestrians
  grid.append(finding(8, 'Where people walk', { span2: true },
    h('div', { class: 'img-grid' },
      imageFigure('media/occupancy_person.jpg', 'Heat map of pedestrian positions over all sample clips.', 'Pedestrian occupancy, all clips (log scale). Sidewalks and the three zebra crossings dominate.', lightbox),
      imageFigure('media/C3905_person_tracks.jpg', 'Every pedestrian track of clip C3905 drawn on the background, coloured by heading.', 'C3905 pedestrian tracks, colour = heading. Note the long diagonal line straight across the junction.', lightbox)),
    soWhat('Pedestrians stay mostly on the sidewalks and the three crossings, but some cross the junction diagonally, far from any crossing.',
      'jaywalking = a pedestrian on the carriageway outside the (slightly dilated) crossings for at least 1 s. failure_to_yield only looks at the crossings themselves.')));

  // 9. vehicles
  grid.append(finding(9, 'Traffic follows a few stable corridors', { span2: true },
    h('div', { class: 'img-grid' },
      imageFigure('media/flow_field.jpg', 'Arrows showing the dominant motion direction in each cell of the image.', 'Learned flow field: dominant direction per 16-px cell. Green = one direction dominates (> 50 %), amber = mixed (turning areas).', lightbox),
      imageFigure('media/occupancy_vehicle.jpg', 'Heat map of vehicle positions over all sample clips.', 'Vehicle occupancy, all clips (log scale): queues at the stop line are the hottest area.', lightbox),
      imageFigure('media/C3905_vehicle_tracks.jpg', 'Every vehicle track of clip C3905 drawn on the background, coloured by heading.', 'C3905 vehicle tracks, colour = heading: straight-through, left turns and U-turns around the median tip.', lightbox),
      imageFigure('media/layout_C3896.jpg', 'The scene layout polygons drawn over clip C3896.', 'Scene layout drawn once on the reference: approach, outbound, median, stop zone, three crossings (cw1–cw3), islands, junction.', lightbox)),
    soWhat('Vehicle motion is highly regular: each cell of the image has one dominant direction except in the turning areas, and U-turns loop around the median tip.',
      'wrong_way = moving against a strongly one-way cell of the flow field; illegal_u_turn = origin on the approach, destination on the outbound carriageway, passing the turning area beyond the median tip.')));

  // cards are built in code order; show them in finding order so single-width cards pair up
  [...grid.children].sort((a, b) => Number(a.dataset.order) - Number(b.dataset.order)).forEach((el) => grid.append(el));

  // ---------- explorer
  out.push(explorer(vids));
  root.replaceChildren(...out);
}

function tile(label, value, unit, sub) {
  return h('div', { class: 'stat' }, h('dt', null, label), h('dd', null, value, unit ? h('small', null, ' ' + unit) : null),
    sub ? h('dd', { class: 'stat-sub' }, sub) : null);
}

/** Lanes = segments between the solid dividers that carry a real share of the traffic. */
function countLanes(angles, counts, lines) {
  const cuts = [-Infinity, ...[...(lines || [])].sort((a, b) => a - b), Infinity];
  const total = sum(counts) || 1;
  let n = 0;
  for (let k = 0; k < cuts.length - 1; k++) {
    const mass = sum(angles.map((a, i) => (a > cuts[k] && a <= cuts[k + 1] ? counts[i] : 0)));
    if (mass / total > 0.02) n++;
  }
  return n;
}

// ---------------------------------------------------------------- per-video explorer
function explorer(vids) {
  let cur = vids[0];
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Choose a video' });
  const c = {
    counts: chartHost('Objects in view over time', 'mean detections per analysed frame, 5-s bins, stacked'),
    thr: chartHost('Approach vehicles crossing the stop line', 'per minute'),
    speed: chartHost('Vehicle speed distribution', 'px/s'),
    bright: chartHost('Frame brightness', 'mean grey level of 1-s thumbnails'),
    reg: chartHost('Registration offset vs reference', 'px, every 5 s'),
  };
  const card = (x, span2) => h('article', { class: 'card' + (span2 ? ' span-2' : '') }, x.el);
  const wrapEl = h('div', null,
    h('div', { class: 'explorer-bar' }, h('h3', null, 'Explore one video'), seg),
    h('div', { class: 'explorer-grid' }, card(c.counts, true), card(c.thr), card(c.speed), card(c.bright), card(c.reg)));

  const builds = {
    counts: (v) => [(t) => {
      const C = v.counts || { t: [] };
      const present = OBJECTS.filter((o) => Array.isArray(C[o.id]) && C[o.id].some((x) => x > 0));
      return {
        type: 'line',
        data: { datasets: present.map((o, i) => ({ label: o.label, data: C.t.map((tt, j) => ({ x: tt + 2.5, y: C[o.id][j] })), borderColor: slot(o.slot), backgroundColor: alpha(slot(o.slot), t.dark ? 0.35 : 0.28), fill: i === 0 ? 'origin' : '-1', tension: 0.25, borderWidth: 1.5, pointRadius: 0 })) },
        options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'bottom' }, tooltip: { itemSort: (a, b) => b.datasetIndex - a.datasetIndex, callbacks: { title: (it) => `${fmtT(it[0].parsed.x - 2.5)}–${fmtT(it[0].parsed.x + 2.5)}`, label: (it) => ` ${it.dataset.label}: ${it.raw.y.toFixed(1)}` } } },
          scales: { x: axis(t, { type: 'linear', min: 0, max: v.duration, ticks: { callback: fmtT, maxTicksLimit: 8 }, title: 'time in clip' }), y: axis(t, { stacked: true, min: 0, title: 'objects per frame' }) } },
      };
    }, { columns: ['Bin start', ...OBJECTS.map((o) => o.label)], rows: (v.counts && v.counts.t || []).map((tt, j) => [fmtTime(tt), ...OBJECTS.map((o) => (v.counts[o.id] ? v.counts[o.id][j] : '–'))]) }],
    thr: (v) => [(t) => ({
      type: 'bar',
      data: { labels: (v.throughput && v.throughput.minute || []).map((m) => `${m}–${m + 1}`), datasets: [{ label: 'vehicles', data: (v.throughput && v.throughput.vehicles) || [], backgroundColor: slot(0), maxBarThickness: 24, borderRadius: 4, borderSkipped: 'start' }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => `minute ${it[0].label}`, label: (it) => ` ${it.raw} vehicles` } } },
        scales: { x: axis(t, { grid: { display: false }, title: 'minute of clip' }), y: axis(t, { min: 0, title: 'vehicles', ticks: { precision: 0 } }) } },
    }), { columns: ['Minute', 'Vehicles'], rows: (v.throughput && v.throughput.minute || []).map((m, i) => [`${m}–${m + 1}`, v.throughput.vehicles[i]]) }],
    speed: (v) => {
      const S = v.speeds || { bin_px_s: [], count: [] };
      const bw = S.bin_px_s.length > 1 ? S.bin_px_s[1] - S.bin_px_s[0] : 25;
      return [(t) => ({
        type: 'bar',
        data: { datasets: [{ label: 'observations', data: S.bin_px_s.map((b, i) => ({ x: b + bw / 2, y: S.count[i] })), backgroundColor: slot(0), barPercentage: 0.9, categoryPercentage: 1, borderRadius: 2 }] },
        options: { plugins: { legend: { display: false }, vlines: { lines: isNum(S.median) ? [{ x: S.median, color: t.text2, dash: [4, 3], label: `median ${S.median.toFixed(0)}`, align: 'left', dx: 6 }] : [], textColor: t.text2, font: t.font },
          tooltip: { callbacks: { title: (it) => `${(it[0].parsed.x - bw / 2).toFixed(0)}–${(it[0].parsed.x + bw / 2).toFixed(0)} px/s`, label: (it) => ` ${fmtInt(it.parsed.y)} observations` } } },
          scales: { x: axis(t, { type: 'linear', min: 0, max: (S.bin_px_s[S.bin_px_s.length - 1] || 650) + bw, title: 'px/s', grid: { display: false } }), y: axis(t, { title: 'observations' }) } },
        plugins: [vlinePlugin],
      }), { columns: ['Speed bin (px/s)', 'Observations'], rows: S.bin_px_s.map((b, i) => [`${b}–${b + bw}`, fmtInt(S.count[i])]) }];
    },
    bright: (v) => [(t) => ({
      type: 'line',
      data: { datasets: [{ label: 'brightness', data: (v.brightness && v.brightness.t || []).map((tt, i) => ({ x: tt, y: v.brightness.mean[i] })), borderColor: slot(0), backgroundColor: alpha(slot(0), 0.1), fill: 'origin', tension: 0.2 }] },
      options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => fmtT(it[0].parsed.x), label: (it) => ` grey level ${it.parsed.y.toFixed(1)}` } } },
        scales: { x: axis(t, { type: 'linear', min: 0, max: v.duration, ticks: { callback: fmtT, maxTicksLimit: 7 } }), y: axis(t, { min: 0, max: 128, title: 'grey level' }) } },
    }), { columns: ['t', 'Mean grey level'], rows: (v.brightness && v.brightness.t || []).filter((_, i) => i % 5 === 0).map((tt) => [fmtTime(tt), v.brightness.mean[v.brightness.t.indexOf(tt)]]) }],
    reg: (v) => {
      const R = v.registration || { t: [], dx: [], dy: [] };
      return [(t) => ({
        type: 'line',
        data: { datasets: [
          { label: 'dx', data: R.t.map((tt, i) => ({ x: tt, y: R.dx[i] })), borderColor: slot(0), backgroundColor: slot(0), pointRadius: 1.5 },
          { label: 'dy', data: R.t.map((tt, i) => ({ x: tt, y: R.dy[i] })), borderColor: slot(1), backgroundColor: slot(1), pointRadius: 1.5 }] },
        options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { title: (it) => fmtT(it[0].parsed.x), label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(2)} px` } } },
          scales: { x: axis(t, { type: 'linear', min: 0, max: v.duration, ticks: { callback: fmtT, maxTicksLimit: 7 } }), y: axis(t, { title: 'px' }) } },
      }), { columns: ['t', 'dx (px)', 'dy (px)'], rows: R.t.map((tt, i) => [fmtTime(tt), R.dx[i], R.dy[i]]) }];
    },
  };

  const mounted = {};
  const labels = {
    counts: (v) => `Stacked area chart of detected objects over time in ${v.name}.`,
    thr: (v) => `Bar chart of vehicles crossing the stop line per minute in ${v.name}.`,
    speed: (v) => `Histogram of vehicle speeds in ${v.name}.`,
    bright: (v) => `Line chart of frame brightness over time in ${v.name}.`,
    reg: (v) => `Line chart of registration offset over time in ${v.name}.`,
  };
  const show = (v) => {
    cur = v;
    for (const b of seg.children) b.setAttribute('aria-pressed', String(b.dataset.name === v.name));
    for (const key of Object.keys(builds)) {
      const [build, table] = builds[key](v);
      if (mounted[key]) mounted[key].update(build, table, labels[key](v));
      else mounted[key] = mountChart(c[key].host, build, { table, label: labels[key](v), height: key === 'counts' ? 'lg' : undefined });
    }
  };
  for (const v of vids) seg.append(h('button', { type: 'button', 'data-name': v.name, 'aria-pressed': 'false', onclick: () => show(v) }, `${v.name}`, h('span', { class: 'muted' }, ` · ${v.lighting || ''}`)));
  show(cur);
  wrapEl.className = 'explorer';
  return wrapEl;
}
