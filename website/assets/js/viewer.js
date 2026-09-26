// EventViewer: annotated video + clickable event timeline + causal risk curve + events table,
// all synced to one playhead. Used by the Results section and by the live demo.

import { h, s, clamp, isNum, fmtTime, fmtDur, timeStep, prefersReducedMotion } from './core.js?v=team2';
import { classInfo, classVar, classOrder, sortClasses, SIGNAL } from './palette.js?v=team2';

export const ALARM_THRESHOLD = 0.5;
const ALARM_MERGE_GAP = 2.0;

// ---------------------------------------------------------------- data hygiene
export function normaliseEvents(events, duration) {
  const out = [];
  if (!Array.isArray(events)) return out;
  for (const e of events) {
    let st, en, label;
    if (Array.isArray(e)) [st, en, label] = e;
    else if (e && typeof e === 'object') ({ start: st, end: en, label } = e);
    st = Number(st); en = Number(en);
    if (!isNum(st) || !isNum(en) || typeof label !== 'string' || !label) continue;
    if (en < st) [st, en] = [en, st];
    if (isNum(duration) && duration > 0) { st = clamp(st, 0, duration); en = clamp(en, 0, duration); }
    if (en <= st) continue;
    out.push({ s: st, e: en, label });
  }
  out.sort((a, b) => a.s - b.s || classOrder(a.label) - classOrder(b.label));
  out.forEach((ev, i) => { ev.i = i; });
  return out;
}

export function normaliseRisk(risk) {
  const out = [];
  if (!Array.isArray(risk)) return out;
  for (const p of risk) {
    if (!Array.isArray(p)) continue;
    const t = Number(p[0]), v = Number(p[1]);
    if (!isNum(t) || !isNum(v)) continue;
    out.push([t, clamp(v, 0, 1)]);
  }
  out.sort((a, b) => a[0] - b[0]);
  return out;
}

export function normaliseSignal(signal) {
  if (!Array.isArray(signal)) return [];
  return signal.filter((p) => Array.isArray(p) && isNum(p[0]) && isNum(p[1]) && p[1] >= p[0])
    .map((p) => [p[0], p[1], SIGNAL[p[2]] ? p[2] : 'unknown']);
}

/** Peak, seconds above the alarm threshold and alarm runs (merged when closer than 2 s, like the metric). */
export function riskStats(risk) {
  let peak = null, above = 0;
  const alarms = [], bands = [];
  let band = null;
  for (let i = 0; i < risk.length; i++) {
    const [t, v] = risk[i];
    if (!peak || v > peak[1]) peak = [t, v];
    const dt = i + 1 < risk.length ? risk[i + 1][0] - t : (i > 0 ? t - risk[i - 1][0] : 0);
    if (v >= ALARM_THRESHOLD) {
      above += Math.max(0, dt);
      const last = alarms[alarms.length - 1];
      if (last && t - last[1] <= ALARM_MERGE_GAP) last[1] = t; else alarms.push([t, t]);
      if (band) band[1] = t + Math.max(0, dt); else band = [t, t + Math.max(0, dt)];
    } else if (band) { bands.push(band); band = null; }
  }
  if (band) bands.push(band);
  return { peak, above, alarms, bands };
}

// ---------------------------------------------------------------- component
export class EventViewer {
  /**
   * opts: {duration, events, risk, signal?, videoSrc?, poster?, videoLabel?, noVideoTitle?, noVideoText?, compact?}
   */
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = opts;
    this.risk = normaliseRisk(opts.risk);
    let dur = Number(opts.duration);
    if (!isNum(dur) || dur <= 0) {
      const ends = (Array.isArray(opts.events) ? opts.events : []).map((e) => Number(Array.isArray(e) ? e[1] : e && e.end)).filter(isNum);
      dur = Math.max(1, ...ends, this.risk.length ? this.risk[this.risk.length - 1][0] : 0);
    }
    this.duration = dur;
    this.events = normaliseEvents(opts.events, dur);
    this.signal = normaliseSignal(opts.signal);
    this.classes = sortClasses([...new Set(this.events.map((e) => e.label))]);
    this.byClass = new Map(this.classes.map((c) => [c, this.events.filter((e) => e.label === c)]));
    this.counts = new Map(this.classes.map((c) => [c, this.byClass.get(c).length]));
    this.stats = riskStats(this.risk);
    this.hidden = new Set();
    this.time = 0;
    this.sort = { key: 's', dir: 1 };
    this.barEls = {};
    this.rowEls = {};
    this._active = [];
    this._activeKey = null;
    this.videoState = opts.videoSrc ? 'loading' : 'none';
    this.pendingSeek = null;
    this.raf = 0;
    this.lastW = 0;
    this.build();
  }

  // ---------- DOM ----------
  build() {
    const o = this.opts;
    this.el = h('div', { class: 'ev' });
    this.tip = h('div', { class: 'ev-tip', role: 'tooltip', hidden: true });

    // media
    this.media = h('div', { class: 'ev-media' });
    this.noVideo = h('div', { class: 'ev-novideo', hidden: true },
      o.poster ? h('img', { src: o.poster, alt: o.videoLabel ? `Still frame of ${o.videoLabel}` : 'Still frame', loading: 'lazy' }) : null,
      h('div', { class: 'msg' },
        h('b', null, o.noVideoTitle || 'Annotated video not available yet'),
        o.noVideoText || 'The timeline, the risk curve and the table below still work: click them to move the playhead.'));
    if (o.videoSrc) {
      this.video = h('video', { controls: true, preload: 'metadata', playsinline: true, poster: o.poster || null,
        'aria-label': o.videoLabel ? `Annotated video, ${o.videoLabel}` : 'Annotated video' });
      this.source = h('source', { src: o.videoSrc, type: 'video/mp4' });
      this.video.append(this.source, 'Your browser cannot play this video.');
      this.media.append(this.video, this.noVideo);
      this.bindVideo();
    } else {
      this.media.append(this.noVideo);
      this.noVideo.hidden = false;
    }

    // side panel
    this.clockCur = h('span', null, fmtTime(0, true));
    const clock = h('div', { class: 'ev-clock' }, this.clockCur, h('small', null, ' / ', fmtTime(this.duration, true)));
    this.nowList = h('div', { class: 'ev-now-list', 'aria-live': 'polite' });
    const now = h('div', { class: 'ev-now' }, h('h4', null, 'Happening now'), this.nowList);
    this.statsEl = h('dl', { class: 'ev-stats' });
    this.legendHead = h('div', { class: 'ev-legend-head' }, h('span', null, 'Classes · click to hide or show'));
    this.legend = h('div', { class: 'ev-legend', role: 'group', 'aria-label': 'Show or hide event classes' });
    const side = h('aside', { class: 'ev-side', 'aria-label': 'Playback summary' }, clock, now, this.statsEl,
      this.classes.length ? h('div', null, this.legendHead, this.legend) : null);

    // charts
    this.tlHost = h('div', { class: 'ev-timeline' });
    this.rkHost = h('div', { class: 'ev-risk' });
    const charts = h('div', { class: 'ev-charts' },
      h('div', { class: 'ev-chart-head' }, h('h4', null, 'Event timeline'), h('span', { class: 'hint' }, 'Click a bar to jump to the event · click anywhere to seek')),
      this.tlHost,
      h('div', { class: 'ev-chart-head' }, h('h4', null, 'Accident risk (causal, per frame)'), h('span', { class: 'hint' }, `Dashed line: alarm threshold ${ALARM_THRESHOLD}`)),
      this.rkHost);

    // table
    this.tbody = h('tbody');
    this.tableSummary = h('summary', null, `All events (${this.events.length})`);
    this.thead = h('thead');
    const table = h('table', { class: 'data-table ev-table' }, this.thead, this.tbody);
    this.tableWrap = h('details', { class: 'ev-table-wrap', open: !this.opts.collapseTable },
      this.tableSummary,
      this.events.length ? h('div', { class: 'ev-table-scroll' }, table) : h('p', { class: 'ev-empty' }, 'No events were detected in this clip.'));
    this.tbody.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-i]');
      if (!tr) return;
      const ev = this.events[Number(tr.getAttribute('data-i'))];
      if (ev) this.seek(ev.s, { scroll: true });
    });

    this.el.append(h('div', { class: 'ev-top' }, this.media, side), charts, this.tableWrap, this.tip);
    this.root.replaceChildren(this.el);

    this.renderStats();
    this.renderLegend();
    this.renderHead();
    this.renderTable();
    this.renderCharts();
    this.setTime(0);

    if ('ResizeObserver' in window) {
      this.ro = new ResizeObserver(() => {
        const w = Math.round(this.tlHost.clientWidth);
        if (w && Math.abs(w - this.lastW) >= 2) this.renderCharts();
      });
      this.ro.observe(this.tlHost);
    } else {
      this._onResize = () => this.renderCharts();
      window.addEventListener('resize', this._onResize);
    }
  }

  bindVideo() {
    const v = this.video;
    const fail = () => this.showNoVideo();
    this.source.addEventListener('error', fail);
    v.addEventListener('error', fail);
    v.addEventListener('loadedmetadata', () => {
      this.videoState = 'ok';
      if (this.pendingSeek != null) { try { v.currentTime = this.pendingSeek; } catch { /* ignore */ } this.pendingSeek = null; }
    });
    v.addEventListener('timeupdate', () => { if (!this.raf) this.setTime(v.currentTime); });
    v.addEventListener('seeked', () => this.setTime(v.currentTime));
    v.addEventListener('play', () => this.startLoop());
    v.addEventListener('pause', () => this.stopLoop());
    v.addEventListener('ended', () => this.stopLoop());
  }

  showNoVideo() {
    if (this.videoState === 'failed') return;
    this.videoState = 'failed';
    this.stopLoop();
    if (this.video) this.video.hidden = true;
    this.noVideo.hidden = false;
  }

  startLoop() {
    if (this.raf) return;
    const loop = () => {
      if (!this.video || this.video.paused || this.video.ended) { this.raf = 0; return; }
      this.setTime(this.video.currentTime);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }
  stopLoop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; if (this.video) this.setTime(this.video.currentTime); }

  // ---------- time ----------
  seek(t, { scroll = false } = {}) {
    t = clamp(Number(t) || 0, 0, this.duration);
    if (this.video && this.videoState !== 'failed') {
      if (this.video.readyState >= 1) { try { this.video.currentTime = t; } catch { /* ignore */ } }
      else this.pendingSeek = t;
    }
    this.setTime(t);
    if (scroll) this.ensureMediaVisible();
  }

  ensureMediaVisible() {
    const r = this.media.getBoundingClientRect();
    const header = 70;
    if (r.bottom < header + 40 || r.top > window.innerHeight - 40) {
      this.media.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'center' });
    }
  }

  setTime(t) {
    if (!isNum(t)) return;
    this.time = clamp(t, 0, this.duration);
    this.clockCur.textContent = fmtTime(this.time, true);
    this.updatePlayhead();
    this.updateActive();
  }

  updatePlayhead() {
    if (!this.W) return;
    const x = this.x(this.time);
    for (const ln of [this.tlPlay, this.rkPlay]) {
      if (!ln) continue;
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
    }
    if (this.tlCap) this.tlCap.setAttribute('transform', `translate(${x},0)`);
  }

  updateActive(force = false) {
    const t = this.time;
    const act = this.events.filter((ev) => ev.s <= t && t < ev.e && !this.hidden.has(ev.label));
    const key = act.map((e) => e.i).join(',');
    if (!force && key === this._activeKey) return;
    for (const i of this._active) { this.barEls[i]?.classList.remove('is-active'); this.rowEls[i]?.classList.remove('is-active'); }
    for (const ev of act) { this.barEls[ev.i]?.classList.add('is-active'); this.rowEls[ev.i]?.classList.add('is-active'); }
    this._active = act.map((e) => e.i);
    this._activeKey = key;
    if (!act.length) {
      this.nowList.replaceChildren(h('span', null, this.events.length ? 'No event at this moment' : 'No events in this clip'));
      return;
    }
    const shown = act.slice(0, 6).map((ev) => h('span', { class: 'now-chip' },
      h('span', { class: 'swatch', style: { background: classVar(ev.label) } }), classInfo(ev.label).label,
      h('span', { class: 'muted num' }, ` until ${fmtTime(ev.e)}`)));
    if (act.length > 6) shown.push(h('span', { class: 'now-chip' }, `+${act.length - 6} more`));
    this.nowList.replaceChildren(...shown);
  }

  // ---------- panels ----------
  renderStats() {
    const st = this.stats;
    const items = [
      ['Events', String(this.events.length)],
      ['Classes', String(this.classes.length)],
    ];
    const peakDd = h('dd', null);
    if (st.peak) {
      peakDd.append(h('button', { type: 'button', title: 'Jump to the peak', onclick: () => this.seek(st.peak[0], { scroll: true }) },
        st.peak[1].toFixed(2)), h('span', { class: 'muted small num' }, ` at ${fmtTime(st.peak[0])}`));
    } else peakDd.append('–');
    const alarmDd = h('dd', null, this.risk.length ? `${st.alarms.length}` : '–',
      this.risk.length ? h('span', { class: 'muted small' }, ` · ${st.above.toFixed(1)} s ≥ ${ALARM_THRESHOLD}`) : null);
    this.statsEl.replaceChildren(
      ...items.map(([k, v]) => h('div', null, h('dt', null, k), h('dd', null, v))),
      h('div', null, h('dt', null, 'Peak risk'), peakDd),
      h('div', null, h('dt', null, 'Alarms'), alarmDd));
  }

  renderLegend() {
    if (!this.classes.length) return;
    const chips = this.classes.map((c) => h('button', {
      type: 'button', class: 'chip', 'aria-pressed': String(!this.hidden.has(c)),
      title: classInfo(c).desc || classInfo(c).label,
      onclick: () => this.toggleClass(c),
    }, h('span', { class: 'swatch', style: { background: classVar(c) } }), classInfo(c).label, h('span', { class: 'count' }, String(this.counts.get(c)))));
    this.legend.replaceChildren(...chips);
    const head = this.legendHead;
    head.replaceChildren(h('span', null, 'Classes · click to hide or show'));
    if (this.hidden.size) head.append(h('button', { type: 'button', onclick: () => { this.hidden.clear(); this.refreshFilter(); } }, 'Show all'));
  }

  toggleClass(c) {
    if (this.hidden.has(c)) this.hidden.delete(c); else this.hidden.add(c);
    this.refreshFilter();
  }
  refreshFilter() {
    this.renderLegend();
    this.renderTable();
    this.renderTimeline();
    this.updatePlayhead();
    this.updateActive(true);
  }

  renderHead() {
    const cols = [['label', 'Class', ''], ['s', 'Start', 'r'], ['e', 'End', 'r'], ['d', 'Duration', 'r']];
    this.thead.replaceChildren(h('tr', null, cols.map(([key, name, cls]) => {
      const active = this.sort.key === key;
      const th = h('th', { scope: 'col', class: cls || null, 'aria-sort': active ? (this.sort.dir > 0 ? 'ascending' : 'descending') : 'none' },
        h('button', { type: 'button', onclick: () => { this.sort = { key, dir: active ? -this.sort.dir : 1 }; this.renderHead(); this.renderTable(); } },
          name, active ? (this.sort.dir > 0 ? ' ▲' : ' ▼') : ''));
      return th;
    })));
  }

  renderTable() {
    if (!this.events.length) return;
    const { key, dir } = this.sort;
    const val = (ev) => key === 'label' ? classOrder(ev.label) : key === 'd' ? ev.e - ev.s : ev[key];
    const rows = this.events.filter((ev) => !this.hidden.has(ev.label)).sort((a, b) => (val(a) - val(b)) * dir || a.s - b.s);
    this.rowEls = {};
    this.tbody.replaceChildren(...rows.map((ev) => {
      const info = classInfo(ev.label);
      const tr = h('tr', { 'data-i': ev.i },
        h('td', null, h('span', { class: 'cls-cell' }, h('span', { class: 'swatch', style: { background: classVar(ev.label) } }), info.label)),
        h('td', { class: 'r' }, h('button', { type: 'button', class: 'seek-btn', 'aria-label': `Jump to ${info.label} at ${fmtTime(ev.s, true)}` }, fmtTime(ev.s, true))),
        h('td', { class: 'r' }, fmtTime(ev.e, true)),
        h('td', { class: 'r' }, fmtDur(ev.e - ev.s)));
      this.rowEls[ev.i] = tr;
      return tr;
    }));
    const shown = rows.length, total = this.events.length;
    this.tableSummary.textContent = shown === total ? `All events (${total})` : `Events (${shown} of ${total}, filtered)`;
    this.updateActive(true);
  }

  // ---------- charts ----------
  x(t) { return this.L + (t / this.duration) * (this.W - this.L - this.R); }
  tAt(x) { return clamp(((x - this.L) / (this.W - this.L - this.R)) * this.duration, 0, this.duration); }
  svgX(svg, e) {
    const r = svg.getBoundingClientRect();
    return (e.clientX - r.left) * (this.W / (r.width || this.W));
  }

  renderCharts() {
    const W = Math.round(this.tlHost.clientWidth || this.el.clientWidth || 0);
    if (W < 60) return; // not laid out yet (hidden tab)
    this.W = W; this.lastW = W;
    this.narrow = W < 560;
    this.L = this.narrow ? 6 : 150;
    this.R = 14;
    this.renderTimeline();
    this.renderRisk();
    this.updatePlayhead();
  }

  ticks(svg, yTop, yBottom, labelY) {
    const step = timeStep(this.duration, this.narrow ? 4 : 8);
    for (let t = 0; t <= this.duration + 1e-6; t += step) {
      const x = this.x(t);
      svg.append(s('line', { class: 'tl-grid', x1: x, x2: x, y1: yTop, y2: yBottom }));
      const anchor = t === 0 ? 'start' : (x > this.W - 24 ? 'end' : 'middle');
      svg.append(s('text', { class: 'tl-tick', x, y: labelY, 'text-anchor': anchor }, fmtTime(t)));
    }
  }

  showTip(e, content) {
    this.tip.replaceChildren(...content);
    this.tip.hidden = false;
    const root = this.el.getBoundingClientRect();
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let left = e.clientX - root.left + 14;
    let top = e.clientY - root.top - th - 10;
    if (left + tw > root.width - 4) left = e.clientX - root.left - tw - 14;
    if (left < 4) left = 4;
    if (top < 4) top = e.clientY - root.top + 18;
    this.tip.style.left = left + 'px';
    this.tip.style.top = top + 'px';
  }
  hideTip() { this.tip.hidden = true; }

  renderTimeline() {
    if (!this.W) return;
    const { W, L, narrow } = this;
    const rowH = narrow ? 30 : 24, barH = narrow ? 11 : 14, top = 4, axisH = 22;
    const rows = [];
    if (this.signal.length) rows.push({ kind: 'signal', label: 'Main-road signal' });
    for (const c of this.classes) rows.push({ kind: 'class', id: c, label: classInfo(c).label });
    const n = Math.max(rows.length, 1);
    const plotBottom = top + n * rowH;
    const H = plotBottom + axisH;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img',
      'aria-label': `Event timeline: ${this.events.length} events in ${this.classes.length} classes over ${fmtTime(this.duration)}. The table below lists every event.` });
    rows.forEach((row, i) => {
      const y = top + i * rowH;
      if (i % 2 === 0) svg.append(s('rect', { class: 'tl-band', x: 0, y, width: W, height: rowH }));
      const dim = row.kind === 'class' && this.hidden.has(row.id);
      if (narrow) svg.append(s('text', { class: 'tl-label' + (dim ? ' dim' : ''), x: L + 2, y: y + 11 }, row.label));
      else svg.append(s('text', { class: 'tl-label' + (dim ? ' dim' : ''), x: L - 10, y: y + rowH / 2 + 4, 'text-anchor': 'end' }, row.label));
    });
    this.ticks(svg, top, plotBottom, plotBottom + 15);
    const barY = (i) => top + i * rowH + (narrow ? 15 : (rowH - barH) / 2);
    this.barEls = {};
    rows.forEach((row, i) => {
      if (row.kind === 'signal') {
        for (const [a, b, st] of this.signal) {
          const x1 = this.x(a), x2 = this.x(Math.min(this.duration, b + 0.1));
          const r = s('rect', { class: SIGNAL[st].cls, x: x1, y: barY(i) + 2, width: Math.max(1, x2 - x1), height: barH - 4, 'data-sig': `${st}|${a}|${b}` });
          svg.append(r);
        }
      } else {
        const dim = this.hidden.has(row.id);
        for (const ev of this.byClass.get(row.id)) {
          const x1 = this.x(ev.s), x2 = this.x(ev.e);
          const r = s('rect', { class: 'tl-bar' + (dim ? ' dim' : ''), x: x1, y: barY(i), width: Math.max(3, x2 - x1), height: barH, rx: 3, style: `fill:${classVar(row.id)}`, 'data-i': ev.i });
          this.barEls[ev.i] = r;
          svg.append(r);
        }
      }
    });
    if (!rows.length) {
      svg.append(s('text', { class: 'rk-empty', x: W / 2, y: top + rowH / 2 + 4, 'text-anchor': 'middle' }, 'No events detected in this clip'));
    }
    this.tlHover = s('line', { class: 'tl-hover', x1: 0, x2: 0, y1: top, y2: plotBottom, visibility: 'hidden' });
    this.tlPlay = s('line', { class: 'tl-playhead', x1: L, x2: L, y1: top, y2: plotBottom });
    this.tlCap = s('path', { class: 'tl-playhead-cap', d: 'M-5,0 L5,0 L0,6 Z' });
    svg.append(this.tlHover, this.tlPlay, this.tlCap);

    svg.addEventListener('pointermove', (e) => {
      const x = this.svgX(svg, e);
      const tgt = e.target;
      const idx = tgt && tgt.getAttribute ? tgt.getAttribute('data-i') : null;
      const sig = tgt && tgt.getAttribute ? tgt.getAttribute('data-sig') : null;
      if (x < L - 4) { this.tlHover.setAttribute('visibility', 'hidden'); this.hideTip(); return; }
      this.tlHover.setAttribute('x1', x); this.tlHover.setAttribute('x2', x);
      this.tlHover.setAttribute('visibility', 'visible');
      if (idx != null && this.events[+idx]) {
        const ev = this.events[+idx];
        this.showTip(e, [
          h('div', { class: 'tv' }, `${fmtTime(ev.s, true)} – ${fmtTime(ev.e, true)}`, h('span', { class: 'muted' }, ` · ${fmtDur(ev.e - ev.s)}`)),
          h('div', { class: 'tk' }, h('span', { class: 'swatch', style: { background: classVar(ev.label) } }), classInfo(ev.label).label, h('span', { class: 'muted' }, ' · click to jump')),
        ]);
      } else if (sig) {
        const [st, a, b] = sig.split('|');
        this.showTip(e, [h('div', { class: 'tv' }, `${fmtTime(+a)} – ${fmtTime(+b)}`), h('div', { class: 'tk' }, `Signal: ${SIGNAL[st] ? SIGNAL[st].label.toLowerCase() : st}`)]);
      } else {
        this.showTip(e, [h('div', { class: 'tv' }, fmtTime(this.tAt(x), true)), h('div', { class: 'tk' }, 'click to seek')]);
      }
    });
    svg.addEventListener('pointerleave', () => { this.tlHover.setAttribute('visibility', 'hidden'); this.hideTip(); });
    svg.addEventListener('click', (e) => {
      const idx = e.target && e.target.getAttribute ? e.target.getAttribute('data-i') : null;
      if (idx != null && this.events[+idx]) { this.seek(this.events[+idx].s); return; }
      const x = this.svgX(svg, e);
      if (x >= L - 4) this.seek(this.tAt(x));
    });
    this.tlHost.replaceChildren(svg);
  }

  renderRisk() {
    if (!this.W) return;
    const { W, L, R, narrow } = this;
    const H = narrow ? 124 : 150, top = 12, bottom = 22;
    const ph = H - top - bottom;
    const y = (v) => top + (1 - v) * ph;
    const st = this.stats;
    const label = this.risk.length
      ? `Risk curve. Peak ${st.peak[1].toFixed(2)} at ${fmtTime(st.peak[0])}; ${st.alarms.length} alarm${st.alarms.length === 1 ? '' : 's'} above ${ALARM_THRESHOLD}.`
      : 'No risk curve for this clip.';
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': label });
    // horizontal grid
    for (const v of [0, 0.25, 0.75, 1]) {
      svg.append(s('line', { class: 'tl-grid', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    }
    for (const v of [0, 0.5, 1]) {
      if (narrow) svg.append(s('text', { class: 'tl-tick', x: L + 2, y: v === 0 ? y(v) - 3 : y(v) + (v === 1 ? 11 : -3) }, v.toFixed(1)));
      else svg.append(s('text', { class: 'tl-tick', x: L - 10, y: y(v) + 4, 'text-anchor': 'end' }, v.toFixed(1)));
    }
    if (!narrow) svg.append(s('text', { class: 'tl-label', x: L - 10, y: top + ph / 2 - 12, 'text-anchor': 'end' }, 'risk'));
    this.ticks(svg, top, top + ph, H - 6);
    // alarm bands
    for (const [a, b] of st.bands) {
      const x1 = this.x(a), x2 = this.x(Math.min(this.duration, b));
      svg.append(s('rect', { class: 'rk-alarm', x: x1, y: top, width: Math.max(2, x2 - x1), height: ph }));
    }
    if (this.risk.length) {
      const pts = this.decimate(this.risk);
      let d = '';
      pts.forEach(([t, v], i) => { d += (i ? 'L' : 'M') + this.x(t).toFixed(1) + ',' + y(v).toFixed(1); });
      const x0 = this.x(pts[0][0]).toFixed(1), x1 = this.x(pts[pts.length - 1][0]).toFixed(1);
      svg.append(s('path', { class: 'rk-area', d: `${d}L${x1},${y(0)}L${x0},${y(0)}Z` }));
      svg.append(s('path', { class: 'rk-line', d }));
    } else {
      svg.append(s('text', { class: 'rk-empty', x: (L + W - R) / 2, y: top + ph / 2, 'text-anchor': 'middle' }, 'No risk curve for this clip'));
    }
    // threshold
    svg.append(s('line', { class: 'rk-thr', x1: L, x2: W - R, y1: y(ALARM_THRESHOLD), y2: y(ALARM_THRESHOLD) }));
    svg.append(s('text', { class: 'rk-thr-label', x: W - R - 2, y: y(ALARM_THRESHOLD) - 5, 'text-anchor': 'end' }, `alarm ${ALARM_THRESHOLD}`));
    // hover + playhead
    const hover = s('line', { class: 'tl-hover', x1: 0, x2: 0, y1: top, y2: top + ph, visibility: 'hidden' });
    const dot = s('circle', { class: 'rk-dot', r: 4.5, cx: -20, cy: -20, visibility: 'hidden' });
    this.rkPlay = s('line', { class: 'tl-playhead', x1: L, x2: L, y1: top, y2: top + ph });
    svg.append(hover, this.rkPlay, dot);

    const nearest = (t) => {
      const r = this.risk;
      let lo = 0, hi = r.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (r[mid][0] < t) lo = mid; else hi = mid; }
      return Math.abs(r[lo][0] - t) <= Math.abs(r[hi][0] - t) ? r[lo] : r[hi];
    };
    svg.addEventListener('pointermove', (e) => {
      const x = this.svgX(svg, e);
      if (x < L - 4) { hover.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); this.hideTip(); return; }
      const t = this.tAt(x);
      hover.setAttribute('x1', x); hover.setAttribute('x2', x); hover.setAttribute('visibility', 'visible');
      if (this.risk.length) {
        const p = nearest(t);
        dot.setAttribute('cx', this.x(p[0])); dot.setAttribute('cy', y(p[1])); dot.setAttribute('visibility', 'visible');
        this.showTip(e, [h('div', { class: 'tv' }, `risk ${p[1].toFixed(2)}`), h('div', { class: 'tk' }, `${fmtTime(p[0], true)} · click to seek`)]);
      } else {
        this.showTip(e, [h('div', { class: 'tv' }, fmtTime(t, true)), h('div', { class: 'tk' }, 'click to seek')]);
      }
    });
    svg.addEventListener('pointerleave', () => { hover.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); this.hideTip(); });
    svg.addEventListener('click', (e) => { const x = this.svgX(svg, e); if (x >= L - 4) this.seek(this.tAt(x)); });
    this.rkHost.replaceChildren(svg);
  }

  /** Keep min and max per pixel column so peaks survive when there are more points than pixels. */
  decimate(pts) {
    const cols = Math.max(50, Math.round(this.W - this.L - this.R));
    if (pts.length <= cols * 2) return pts;
    const out = [];
    let col = -1, mn = null, mx = null;
    const flush = () => { if (!mn) return; if (mn[0] <= mx[0]) out.push(mn, mx); else out.push(mx, mn); };
    for (const p of pts) {
      const c = Math.floor(((p[0]) / this.duration) * cols);
      if (c !== col) { flush(); col = c; mn = p; mx = p; }
      else { if (p[1] < mn[1]) mn = p; if (p[1] > mx[1]) mx = p; }
    }
    flush();
    return out;
  }

  destroy() {
    if (this.ro) this.ro.disconnect();
    if (this._onResize) window.removeEventListener('resize', this._onResize);
    this.stopLoop();
    if (this.video) {
      try { this.video.pause(); } catch { /* ignore */ }
      if (this.source) this.source.remove();
      this.video.removeAttribute('src');
      try { this.video.load(); } catch { /* ignore */ }
    }
    this.root.replaceChildren();
  }
}
