// Cinematic hero. Plays 12 s of junction footage annotated by our own pipeline and keeps a
// CCTV-style HUD in sync with it: a running timecode, an event log that fires at the real
// detection times (with a live "evidence crop" of the flagged vehicle), the causal
// accident-risk trace drawn up to the playhead, a lock-on reticle that follows the vehicle
// but never crosses the copy, a timeline playhead and a count-up of the stats.
//
// First play: the raw camera frame is "read" into the annotated footage by one sweep.
// One requestAnimationFrame loop, alive only while the footage plays on screen (or while the
// stats count up). Under prefers-reduced-motion nothing autoplays: the poster stays, the HUD
// shows the poster's moment (6 s) and the visitor can press Play. If the video fails to load,
// the poster stays and the play control is removed.
//
// Events and risk come from data/results.json (refined on idle); the built-in copy below is
// the fallback. Only TRACK is measured by hand from the rendered clip.

const CLIP = { id: 'C3896', start: 45, fps: 15, poster: 6.0, first: 3.2, duration: 11.945 };
const FRAME = { w: 1280, h: 634 };
const RISK_MAX = 0.6;  // sparkline range: the official alarm threshold (0.5) sits near its top
const ALARM = 0.5;
const SCAN_S = 0.9;    // vertical scan sweep that re-acquires the frame at every loop start
const TYPE_CPS = 32;   // event names type into the log at this many characters per second

// Pipeline output for C3896 (fallback copy of data/results.json): events [start, end, class]
// in source seconds, and the causal risk series at 10 Hz from 45.045 s (x 1000).
const FALLBACK_EVENTS = [
  [0.07, 340.34, 'stopped_vehicle'], [43.5, 107.0, 'congestion'],
  [49.62, 52.72, 'illegal_turn'], [53.32, 55.12, 'failure_to_yield'],
];
const FALLBACK_RISK = '114,102,93,238,214,193,175,159,146,135,126,119,115,165,169,186,183,189,166,266,234,206,182,171,221,197,174,153,134,118,104,92,81,71,63,55,49,43,189,167,147,129,114,100,114,247,303,267,235,207,182,160,141,125,110,97,85,75,66,58,51,45,40,35,31,28,24,21,19,17,15,13,12,10,9,8,7,6,6,5,4,4,4,3,3,3,65,106,93,82,195,172,152,134,118,105,94,84,75,66,58,52,45,40,35,31,28,25,22,20,18,16,14,13,12,11,10,9,8,7'
  .split(',').map((v, i) => [45.045 + i * 0.1001, Number(v) / 1000]);

const LABELS = {
  failure_to_yield: 'Failure to yield', near_miss: 'Near miss', jaywalking: 'Jaywalking',
  illegal_u_turn: 'Illegal U-turn', stopped_vehicle: 'Stopped vehicle', solid_line_crossing: 'Solid-line crossing',
  red_light: 'Red-light running', accident: 'Accident', stop_line: 'Stop-line violation', illegal_turn: 'Illegal turn',
  wrong_way: 'Wrong way', congestion: 'Congestion', road_obstacle: 'Road obstacle', fire_smoke: 'Fire / smoke',
};
const label = (id) => LABELS[id] || String(id).replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
const color = (id) => `var(--cls-${String(id).replace(/[^\w-]/g, '')}, var(--hc-paint))`;

// The flagged black SUV, read off the red event boxes our pipeline drew in the clip (measured by
// hand, so it stays tied to this rendered clip): [clip time s, x0, y0, x1, y1] in frame pixels.
// The same track commits both violations (illegal turn, then failure to yield at the lower-left
// zebra); from 6.8 s the box is clipped by the bottom of the frame.
const TRACK = [
  [4.52, 688, 438, 871, 580], [5.00, 694, 446, 871, 591], [5.40, 698, 453, 867, 603],
  [5.81, 701, 458, 861, 611], [6.21, 705, 466, 853, 620], [6.54, 698, 470, 837, 629],
  [6.81, 692, 474, 829, 634], [7.07, 682, 478, 821, 634], [7.34, 668, 484, 811, 634],
  [7.61, 657, 488, 806, 634], [7.87, 636, 494, 795, 634], [8.14, 611, 502, 781, 634],
  [8.41, 592, 506, 771, 634], [8.68, 558, 516, 747, 634], [8.88, 528, 520, 727, 634],
  [9.08, 492, 528, 695, 634], [9.34, 426, 538, 635, 634], [9.54, 372, 545, 587, 634],
  [9.74, 309, 551, 530, 634], [9.94, 242, 558, 463, 634], [10.14, 162, 563, 391, 634],
  [10.34, 76, 572, 305, 634], [10.54, 8, 578, 233, 634],
];
const TRACK_S = TRACK[0][0], TRACK_E = TRACK[TRACK.length - 1][0];

const root = document.querySelector('.hc');
if (root) {
  try { init(root); } catch (err) { console.warn('hero:', err); }
}

function init(root) {
  const $ = (sel) => root.querySelector(sel);
  const video = $('.hc-video');
  const fx = $('.hc-fx-top') || $('.hc-fx');
  if (!video || !fx) return;

  const still = $('.hc-still');
  const tc = $('.hc-tc');
  const lock = $('.hc-lock');
  const corners = Array.from(root.querySelectorAll('.hc-lk'));
  const tag = $('.hc-lock-tag');
  const tagKick = $('.hc-lock-kick');
  const tagLabel = $('.hc-lock-label');
  const scan = $('.hc-scan');
  const track = $('.hc-track');
  const head = $('.hc-head');
  const fill = $('.hc-fill');
  const btn = $('.hc-toggle');
  const btnLabel = $('.hc-toggle-label');
  const recLabel = $('.hc-rec-label');
  const hudLive = $('.hc-hud-live');
  const log = $('.hc-log');
  const ongoingList = $('.hc-ongoing');
  const chipList = $('.hc-chips');
  const riskVal = $('.hc-risk-val');
  const sparkClip = $('.hc-spark-clip');
  const sparkLine = $('.hc-spark-line');
  const sparkArea = $('.hc-spark-area');
  const sparkHead = $('.hc-spark-head');
  const sparkDot = $('.hc-spark-dot');
  const peakNote = $('.hc-peak-note');
  const peakDot = $('.hc-peak-dot');
  const hud = $('.hc-hud');
  const h1 = $('.hero-statement');

  const reducedMq = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  let reduced = !!(reducedMq && reducedMq.matches);

  // ------------------------------------------------------------ state
  let started = false;      // a real video frame is playing
  let posterMode = reduced; // the poster (6 s) is what the visitor sees while nothing plays
  let userPaused = reduced;
  let onScreen = true;
  let failed = false;
  let raf = 0;
  const last = { frame: -1, lockOn: null, key: '', idle: null, side: null, noTag: null, scan: false, flip: false, risk: '', peak: null, tagW: 0, tagH: 0 };
  let geo = { s: 1, ox: 0, oy: 0, w: 0, h: 0 };
  let copyRects = [], uiRects = [], band = false;

  const duration = () => (video.duration > 0 && isFinite(video.duration) ? video.duration : CLIP.duration);
  const now = () => (started ? video.currentTime : (posterMode ? CLIP.poster : CLIP.first));

  // ------------------------------------------------------------ model (events + risk)
  let model = null, chips = [], tracked = [];
  function buildModel(events, riskPts) {
    const dur = duration(), w0 = CLIP.start, w1 = CLIP.start + dur;
    const ongoing = [], discrete = [];
    for (const ev of events) {
      if (!Array.isArray(ev) || ev.length < 3) continue;
      const a = Number(ev[0]), b = Number(ev[1]), id = String(ev[2]);
      if (!(b > w0 && a < w1)) continue;
      const s = Math.max(0, a - w0), e = Math.min(dur, b - w0);
      if (s <= 0.05 && e >= dur - 0.05) ongoing.push({ id, a, b });
      else if (e - s > 0.05) discrete.push({ id, a, b, s, e });
    }
    discrete.sort((x, y) => x.s - y.s);
    const risk = [];
    for (const p of riskPts) {
      if (Array.isArray(p) && p[0] >= w0 - 0.01 && p[0] <= w1 + 0.01) risk.push([p[0] - w0, Math.max(0, Number(p[1]) || 0)]);
    }
    return { ongoing, events: discrete, risk };
  }
  const srcTime = (sec) => {
    const m = Math.floor(sec / 60), s = sec - m * 60;
    return `${String(m).padStart(2, '0')}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
  };
  const el = (tagName, cls, text) => { const n = document.createElement(tagName); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

  function buildLog() {
    // ongoing classes (they cover the whole clip)
    if (ongoingList) {
      ongoingList.replaceChildren(...model.ongoing.map((ev) => {
        const li = el('li'); li.style.setProperty('--c', color(ev.id));
        li.append(el('span', 'hc-sw'), el('span', 'hc-oname', label(ev.id)),
          el('span', 'hc-otime', ev.a < 1 ? 'whole video' : `since ${srcTime(ev.a)}`), el('span', 'hc-live', 'Live'));
        return li;
      }));
    }
    if (hudLive) { // phones: the most recent ongoing class as a live dot in the HUD
      const first = model.ongoing.reduce((m, ev) => (!m || ev.a > m.a ? ev : m), null);
      hudLive.textContent = first ? label(first.id) : '';
      if (first) hudLive.style.setProperty('--c', color(first.id));
    }
    // discrete events: one chip each, with an evidence crop when our tracked vehicle is involved
    const n = model.events.length;
    let prevTracked = false;
    chips = model.events.map((ev, i) => {
      const isTracked = ev.e > TRACK_S && ev.s < TRACK_E;
      const li = el('li', 'hc-chip' + (isTracked ? '' : ' is-nocrop')); li.style.setProperty('--c', color(ev.id));
      const crop = el('span', 'hc-crop'); const canvas = el('canvas'); canvas.width = 176; canvas.height = 110; crop.append(canvas);
      const name = el('span', 'hc-name'); const typed = el('span', 'hc-typed', label(ev.id)); name.append(typed, el('span', 'hc-caret'));
      const kick = `Event ${i + 1}/${n} · ${srcTime(ev.a)}`;
      const meta = `${(ev.b - ev.a).toFixed(1)} s` + (isTracked && prevTracked ? ' · same vehicle' : '');
      const stateEl = el('span', 'hc-state');
      const prog = el('span', 'hc-prog');
      const row = el('span', 'hc-row'); row.append(el('span', 'hc-kick', kick), stateEl);
      const body = el('span', 'hc-body'); body.append(row, name, el('span', 'hc-meta', meta));
      li.append(crop, body, prog);
      prevTracked = prevTracked || isTracked;
      return { ...ev, i, n, kick, name: label(ev.id), li, stateEl, prog, typed, canvas, ctx: null, isTracked, state: -1, shown: -1, drawn: -1 };
    });
    if (chipList) chipList.replaceChildren(...chips.map((c) => c.li));
    tracked = chips.filter((c) => c.isTracked);

    // timeline segments along the top of the stats bar
    if (track) {
      track.querySelectorAll('.hc-seg').forEach((s) => s.remove());
      const segs = [...model.ongoing.slice(0, 1).map((ev) => ({ id: ev.id, s: 0, e: duration(), on: true })), ...model.events];
      const frag = document.createDocumentFragment();
      for (const ev of segs) {
        const seg = el('span', 'hc-seg' + (ev.on ? ' is-ongoing' : ''));
        seg.style.setProperty('--c', color(ev.id));
        seg.dataset.s = ev.s; seg.dataset.e = ev.e;
        frag.append(seg);
      }
      track.prepend(frag);
    }
    layoutTimeline();
    last.key = ''; last.peak = null; last.risk = '';
  }

  // positions that depend on the real clip length (known after loadedmetadata)
  function layoutTimeline() {
    const dur = duration();
    if (track) {
      for (const seg of track.querySelectorAll('.hc-seg')) {
        const s = Math.max(0, +seg.dataset.s), e = Math.min(dur, +seg.dataset.e);
        seg.style.left = `${(s / dur * 100).toFixed(3)}%`;
        seg.style.width = `${(Math.max(0, e - s) / dur * 100).toFixed(3)}%`;
      }
    }
    if (sparkLine && model.risk.length) {
      const X = (t) => (t / dur * 120).toFixed(2), Y = (v) => (34 * (1 - Math.min(v, RISK_MAX) / RISK_MAX)).toFixed(2);
      const pts = model.risk.map(([t, v]) => `${X(t)} ${Y(v)}`);
      const d = 'M' + pts.join('L');
      sparkLine.setAttribute('d', d);
      sparkArea.setAttribute('d', `${d}L${X(model.risk[model.risk.length - 1][0])} 34L${X(model.risk[0][0])} 34Z`);
      const pk = model.risk.reduce((m, p) => (p[1] > m[1] ? p : m), model.risk[0]);
      model.peak = pk;
      if (peakNote) peakNote.textContent = `Peak ${pk[1].toFixed(2)} · ${srcTime(CLIP.start + pk[0])}`;
      if (peakDot) {
        peakDot.style.left = `${(pk[0] / dur * 100).toFixed(2)}%`;
        peakDot.style.top = `${(100 * (1 - Math.min(pk[1], RISK_MAX) / RISK_MAX)).toFixed(1)}%`;
      }
    }
  }

  function riskAt(t) {
    const r = model.risk;
    if (!r.length || t < r[0][0]) return null;
    let lo = 0, hi = r.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (r[mid][0] <= t) lo = mid; else hi = mid - 1; }
    return r[lo][1]; // causal: the latest estimate made at or before t
  }

  // ------------------------------------------------------------ geometry (object-fit: cover)
  function measure() {
    const w = fx.clientWidth, h = fx.clientHeight;
    if (!w || !h) return;
    const s = Math.max(w / FRAME.w, h / FRAME.h);
    const pos = getComputedStyle(video).objectPosition.split(/\s+/);
    const frac = (v) => (v && v.endsWith('%') ? parseFloat(v) / 100 : 0.5);
    geo = { s, ox: (w - FRAME.w * s) * frac(pos[0]), oy: (h - FRAME.h * s) * frac(pos[1]), w, h };
    // phones: the footage is a band above the copy that melts into the page (CSS fades the reticle there)
    band = getComputedStyle(root).getPropertyValue('--hc-band').trim() !== '';
    measureCopy();
  }
  // The copy and the HUD are obstacles for the reticle: measured on resize, never per frame.
  function measureCopy() {
    const fr = fx.getBoundingClientRect();
    const rel = (r, m) => ({ l: r.left - fr.left - m, t: r.top - fr.top - m, r: r.right - fr.left + m, b: r.bottom - fr.top + m });
    const shown = (n) => n && n.offsetParent !== null && n.getClientRects().length > 0;
    copyRects = [];
    if (h1) {
      const range = document.createRange();
      for (const line of h1.querySelectorAll('.hero-line')) { range.selectNodeContents(line); copyRects.push(rel(range.getBoundingClientRect(), 14)); }
    }
    for (const n of root.querySelectorAll('.hc-lede, .hc-text, .hero-actions > *')) if (shown(n)) copyRects.push(rel(n.getBoundingClientRect(), 10));
    uiRects = [];
    for (const n of [log, btn, hud]) if (shown(n)) uiRects.push(rel(n.getBoundingClientRect(), 8));
  }
  const hits = (l, t, r, b, rects) => rects.some((q) => l < q.r && r > q.l && t < q.b && b > q.t);

  function boxAt(t) {
    if (t < TRACK_S || t > TRACK_E) return null;
    let i = 1;
    while (i < TRACK.length - 1 && TRACK[i][0] < t) i++;
    const a = TRACK[i - 1], b = TRACK[i];
    const k = Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0])));
    return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, a[3] + (b[3] - a[3]) * k, a[4] + (b[4] - a[4]) * k];
  }

  const pad = (n) => String(n).padStart(2, '0');
  function timecode(sec) {
    const whole = Math.floor(sec);
    const ff = Math.min(CLIP.fps - 1, Math.floor((sec - whole) * CLIP.fps + 1e-6));
    return `${pad(Math.floor(whole / 3600))}:${pad(Math.floor(whole / 60) % 60)}:${pad(whole % 60)}:${pad(ff)}`;
  }

  // ------------------------------------------------------------ evidence crop
  const posterImg = new Image();
  posterImg.decoding = 'async';
  posterImg.onload = () => { for (const c of chips) c.drawn = -1; kick(); };
  if (video.poster) posterImg.src = video.poster;
  function drawCrop(c, t, frame) {
    if (c.drawn === frame) return;
    let src = null;
    if (started && video.readyState >= 2) src = video;
    else if (Math.abs(t - CLIP.poster) < 0.25 && posterImg.complete && posterImg.naturalWidth) src = posterImg;
    const box = boxAt(Math.min(Math.max(t, c.s), c.e));
    if (!src || !box) return;
    if (!c.ctx) c.ctx = c.canvas.getContext('2d', { alpha: false });
    if (!c.ctx) return;
    const cw = c.canvas.width, ch = c.canvas.height, aspect = cw / ch;
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    let w = (box[2] - box[0]) * 1.45, h = w / aspect;
    if (h < (box[3] - box[1]) * 1.3) { h = (box[3] - box[1]) * 1.3; w = h * aspect; }
    w = Math.min(w, FRAME.w); h = Math.min(h, FRAME.h);
    const sx = Math.min(Math.max(0, cx - w / 2), FRAME.w - w), sy = Math.min(Math.max(0, cy - h / 2), FRAME.h - h);
    try { c.ctx.drawImage(src, sx, sy, w, h, 0, 0, cw, ch); c.drawn = frame; } catch (e) { /* frame not ready */ }
  }

  // ------------------------------------------------------------ one frame of HUD
  function render(t) {
    const dur = duration();
    const frame = Math.floor(t * CLIP.fps + 1e-6);
    const newFrame = frame !== last.frame;
    if (newFrame) { last.frame = frame; if (tc) tc.textContent = timecode(CLIP.start + t); }

    const p = Math.min(1, Math.max(0, t / dur));
    if (head) head.style.transform = `translate3d(${(p * 100).toFixed(3)}%,0,0)`;
    if (fill) fill.style.transform = `scaleX(${p.toFixed(4)})`;

    // causal risk, drawn up to the playhead
    if (sparkClip) sparkClip.setAttribute('width', (p * 120).toFixed(2));
    if (sparkHead) sparkHead.style.transform = `translate3d(${(p * 100).toFixed(3)}%,0,0)`;
    const r = riskAt(t);
    const rTxt = r == null ? '–' : r.toFixed(2);
    if (rTxt !== last.risk) {
      last.risk = rTxt;
      if (riskVal) { riskVal.textContent = rTxt; riskVal.style.color = r != null && r >= ALARM ? 'var(--hc-rec)' : ''; }
      if (sparkDot) sparkDot.style.transform = `translate3d(0,${(34 * (1 - Math.min(r || 0, RISK_MAX) / RISK_MAX)).toFixed(1)}px,0)`;
    }
    const peakOn = !!(model.peak && t >= model.peak[0]);
    if (peakOn !== last.peak) {
      last.peak = peakOn;
      if (peakNote) peakNote.classList.toggle('is-shown', peakOn);
      if (peakDot) peakDot.classList.toggle('is-shown', peakOn);
    }

    // event log
    let active = null, latest = null;
    const live = started && !reduced;
    for (const c of chips) {
      const state = t < c.s ? 0 : t < c.e ? 1 : 2;
      if (state === 1 && c.isTracked) active = c;
      if (state > 0) latest = c;
      if (state !== c.state) {
        const was = c.state;
        c.state = state;
        c.li.classList.toggle('is-on', state > 0);
        c.li.classList.toggle('is-live', state === 1);
        c.li.classList.toggle('is-done', state === 2);
        c.stateEl.textContent = state === 1 ? 'Live' : state === 2 ? 'Logged' : '';
        if (state !== 1) c.prog.style.transform = `scaleX(${state === 2 ? 1 : 0})`;
        if (state === 0) { c.drawn = -1; if (c.ctx) { c.ctx.fillStyle = '#0f130f'; c.ctx.fillRect(0, 0, c.canvas.width, c.canvas.height); } }
        if (state === 2 && was === 1 && c.isTracked) drawCrop(c, t, -2); // the frame the event ended on
      }
      if (state === 1) {
        c.prog.style.transform = `scaleX(${((t - c.s) / (c.e - c.s)).toFixed(4)})`;
        // the name types in once, the first time the event goes live on screen
        const n = live ? Math.min(c.name.length, 1 + Math.floor((t - c.s) * TYPE_CPS)) : c.name.length;
        if (n !== c.shown) { c.shown = n; c.typed.textContent = c.name.slice(0, n); c.li.classList.toggle('is-typing', n < c.name.length); }
        if (c.isTracked && (newFrame || c.drawn < 0)) drawCrop(c, t, frame);
      } else if (c.shown !== c.name.length) {
        c.shown = c.name.length; c.typed.textContent = c.name; c.li.classList.remove('is-typing');
      }
    }
    for (const c of chips) c.li.classList.toggle('is-latest', c === latest);

    // reticle: from the first tracked violation until the track leaves the frame, never over the copy
    const box = tracked.length && t >= tracked[0].s ? boxAt(t) : null;
    let on = false;
    if (box && geo.h > 0) {
      const g = 8;
      const x0 = geo.ox + box[0] * geo.s - g, y0 = geo.oy + box[1] * geo.s - g;
      const x1 = geo.ox + box[2] * geo.s + g, y1 = geo.oy + box[3] * geo.s + g;
      const vx0 = Math.max(x0, 0), vx1 = Math.min(x1, geo.w), vy0 = Math.max(y0, 0), vy1 = Math.min(y1, geo.h);
      on = vx1 - vx0 > 48 && vy1 - vy0 > 36 && (band || !hits(vx0, vy0, vx1, vy1, copyRects));
      if (on) {
        const cs = 18;
        const pos = [[x0, y0], [x1 - cs, y0], [x0, y1 - cs], [x1 - cs, y1 - cs]];
        for (let i = 0; i < 4; i++) corners[i].style.transform = `translate3d(${pos[i][0].toFixed(1)}px,${pos[i][1].toFixed(1)}px,0)`;
        const key = active ? `e${active.i}` : 'idle';
        if (key !== last.key) {
          // a new violation re-runs the lock-on (two identical keyframes, so no reflow is needed)
          if (active && last.key) { last.flip = !last.flip; lock.classList.toggle('is-relock', last.flip); }
          last.key = key;
          tagKick.textContent = active ? active.kick : 'Tracking';
          tagLabel.textContent = active ? active.name : 'Same vehicle';
          last.tagW = tag.offsetWidth; last.tagH = tag.offsetHeight; // one layout read per label change
        }
        if (!active !== last.idle) { last.idle = !active; lock.classList.toggle('is-idle', !active); }
        // tag: above the box, else beside it, else hidden; never over the copy, the log or the HUD
        const tw = last.tagW, th = last.tagH, top = 8, avoid = copyRects.concat(uiRects);
        const cands = [[x1 - tw, y0 - 10 - th, false], [x1 + 10, y0, true], [x0 - 10 - tw, y0, true]];
        let pick = null;
        for (const cnd of cands) {
          const [tx, ty] = cnd;
          if (tx < 8 || tx + tw > geo.w - 8 || ty < top || ty + th > geo.h - 8) continue;
          if (hits(tx, ty, tx + tw, ty + th, avoid)) continue;
          pick = cnd; break;
        }
        if (pick) tag.style.transform = `translate3d(${pick[0].toFixed(1)}px,${pick[1].toFixed(1)}px,0)`;
        const side = !!(pick && pick[2]), noTag = !pick;
        if (side !== last.side) { last.side = side; lock.classList.toggle('is-side', side); }
        if (noTag !== last.noTag) { last.noTag = noTag; tag.style.visibility = noTag ? 'hidden' : ''; }
      }
    }
    if (on !== last.lockOn) { last.lockOn = on; lock.classList.toggle('is-on', on); }

    // scan sweep while a new pass of the loop starts (only on live playback)
    const sweeping = started && !reduced && !video.paused && t < SCAN_S && readDone;
    if (sweeping) {
      const k = t / SCAN_S;
      scan.style.transform = `translate3d(0,${(k * (geo.h + 120)).toFixed(1)}px,0)`;
      scan.style.opacity = String(1 - k * k);
    } else if (last.scan) {
      scan.style.opacity = '0';
    }
    last.scan = sweeping;
  }

  // ------------------------------------------------------------ first-play "read" sweep
  let intro = !!still && !reduced && getComputedStyle(still).display !== 'none';
  let readStarted = false, readDone = !intro;
  const t0 = performance.now();
  function endIntro() {
    if (readDone) return;
    readDone = true; intro = false;
    root.classList.remove('is-intro', 'is-reading');
    root.classList.add('is-read');
    kick();
  }
  function read() {
    if (!intro || readStarted) return;
    readStarted = true;
    // let the raw frame register for a moment before the pipeline reads it
    setTimeout(() => {
      root.classList.remove('is-intro');
      root.classList.add('is-reading');
      if (still) still.addEventListener('animationend', endIntro, { once: true });
      setTimeout(endIntro, 1500);
    }, Math.max(0, 550 - (performance.now() - t0)));
  }
  if (intro) {
    root.classList.add('is-intro');
    if (still) still.addEventListener('error', () => { still.remove(); endIntro(); }, { once: true });
    // a slow network must not hold the page on the raw frame
    setTimeout(() => { if (!started && !readStarted) { if (video.readyState < 2) posterMode = true; read(); kick(); } }, 2600);
  }

  // ------------------------------------------------------------ stats count-up (once, when seen)
  const counters = [];
  const stats = document.getElementById('hero-stats');
  let statsSeen = false;
  const waiting = [];
  function startCounter(c) {
    c.t0 = performance.now() + 120 * Math.max(0, c.order);
    c.node.data = '0';
    counters.push(c);
    kick();
  }
  if (stats && !reduced && 'MutationObserver' in window) {
    const mo = new MutationObserver((muts) => {
      for (const m of muts) {
        const dd = m.target.closest && m.target.closest('dd[data-stat]');
        if (!dd || counters.some((c) => c.dd === dd) || waiting.some((c) => c.dd === dd)) continue;
        const node = Array.from(dd.childNodes).find((n) => n.nodeType === 3 && /\d/.test(n.data));
        if (!node) continue;
        const final = node.data;
        const target = parseInt(final.replace(/[^\d]/g, ''), 10);
        if (!(target > 1) || !/^[\d,\s]+$/.test(final.trim())) continue;
        const c = { dd, node, final, target, grouped: final.includes(','), order: ['videos', 'tracks', 'events', 'classes'].indexOf(dd.dataset.stat), dur: 1300 };
        // until the bar is on screen the real number stays in place; it counts up when seen
        if (statsSeen) startCounter(c); else waiting.push(c);
      }
    });
    mo.observe(stats, { childList: true, subtree: true, characterData: false });
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        statsSeen = true; io.disconnect();
        while (waiting.length) startCounter(waiting.shift());
      }, { threshold: 0.35 });
      io.observe(stats);
    } else statsSeen = true;
  }
  function stepCounters() {
    const tNow = performance.now();
    for (let i = counters.length - 1; i >= 0; i--) {
      const c = counters[i];
      const k = Math.min(1, Math.max(0, (tNow - c.t0) / c.dur));
      if (k >= 1) { c.node.data = c.final; counters.splice(i, 1); continue; }
      const v = Math.round(c.target * (1 - Math.pow(1 - k, 3)));
      c.node.data = c.grouped ? v.toLocaleString('en-US') : String(v);
    }
  }

  // ------------------------------------------------------------ the loop
  function tick() {
    raf = 0;
    render(now());
    if (counters.length) stepCounters();
    if ((!video.paused && onScreen) || counters.length) raf = requestAnimationFrame(tick);
  }
  function kick() { if (!raf) raf = requestAnimationFrame(tick); }

  // ------------------------------------------------------------ playback control
  function sync() {
    const paused = video.paused;
    root.classList.toggle('is-playing', !paused && started);
    root.classList.toggle('is-paused', paused);
    if (btnLabel) btnLabel.textContent = paused ? 'Play' : 'Pause';
    if (recLabel) recLabel.textContent = paused ? 'Paused' : 'Rec';
  }
  function play() {
    if (failed) return;
    // leaving the poster: continue from the poster's own frame instead of jumping to the start
    if (!started && posterMode) { try { video.currentTime = CLIP.poster; } catch (e) { /* not seekable yet */ } }
    let p;
    try { p = video.play(); } catch (e) { p = null; }
    if (p && typeof p.catch === 'function') {
      p.catch((err) => {
        if (err && err.name === 'AbortError') return; // paused again before playback began
        if (!started) posterMode = true;
        userPaused = true; // autoplay refused: wait for the visitor
        sync(); read(); kick();
      });
    }
  }
  const wanted = () => !userPaused && onScreen && !document.hidden && !failed;
  function reconcile() {
    if (wanted()) { if (video.paused) play(); }
    else if (!video.paused) video.pause();
  }

  video.addEventListener('playing', () => { started = true; posterMode = false; sync(); read(); kick(); });
  video.addEventListener('play', () => { sync(); kick(); });
  video.addEventListener('pause', () => { sync(); kick(); });
  video.addEventListener('seeked', kick);
  video.addEventListener('loadedmetadata', () => {
    // the real clip length decides which classes cover the whole clip and where the segments sit
    if (Math.abs(duration() - CLIP.duration) > 0.05) { model = buildModel(srcEvents, srcRisk); buildLog(); }
    else layoutTimeline();
    measure(); kick();
  });
  // The poster stays when the footage cannot load; the HUD then describes the poster's moment.
  function fail() {
    if (failed) return;
    failed = true; started = false; posterMode = true;
    root.classList.add('is-failed');
    if (btn) btn.hidden = true;
    endIntro(); sync(); kick();
  }
  video.addEventListener('error', fail);

  if (btn && !video.error) {
    btn.hidden = false;
    btn.addEventListener('click', () => {
      if (failed) return;
      userPaused = !video.paused;
      if (!userPaused) { onScreen = true; play(); } else video.pause();
    });
  }

  if (reducedMq) {
    const onChange = () => {
      reduced = reducedMq.matches;
      if (reduced && !video.paused) { userPaused = true; video.pause(); }
    };
    if (reducedMq.addEventListener) reducedMq.addEventListener('change', onChange);
    else if (reducedMq.addListener) reducedMq.addListener(onChange);
  }

  // pause while the hero is off screen or the tab is hidden; resume unless the visitor paused it
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      onScreen = entries[entries.length - 1].isIntersecting;
      reconcile();
    }, { threshold: 0 }).observe(root.querySelector('.hc-stage') || root);
  }
  document.addEventListener('visibilitychange', reconcile);

  let remeasure = 0;
  const later = () => { if (!remeasure) remeasure = requestAnimationFrame(() => { remeasure = 0; measure(); kick(); }); };
  if ('ResizeObserver' in window) new ResizeObserver(later).observe(fx);
  else window.addEventListener('resize', later, { passive: true });
  root.addEventListener('animationend', (e) => { if (e.target.closest && e.target.closest('.hc-copy, .hc-log')) later(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(later);

  // ------------------------------------------------------------ data: built-in copy now, results.json on idle
  let srcEvents = FALLBACK_EVENTS, srcRisk = FALLBACK_RISK;
  model = buildModel(srcEvents, srcRisk);
  buildLog();
  function refine() {
    fetch('data/results.json').then((r) => (r.ok ? r.json() : null)).then((data) => {
      const v = data && Array.isArray(data.videos) && data.videos.find((x) => x && x.name === CLIP.id);
      if (!v || !Array.isArray(v.events)) return;
      const risk = Array.isArray(v.risk) && v.risk.length ? v.risk : FALLBACK_RISK;
      const next = buildModel(v.events, risk);
      const sig = (m) => JSON.stringify([m.ongoing.map((e) => e.id), m.events.map((e) => [e.id, e.a, e.b]), m.risk.map((p) => p[1].toFixed(3))]);
      if (sig(next) === sig(model)) return;
      srcEvents = v.events; srcRisk = risk; model = next;
      buildLog(); measure(); kick();
    }).catch(() => { /* keep the built-in copy */ });
  }
  const idle = (fn) => (window.requestIdleCallback ? window.requestIdleCallback(fn, { timeout: 4000 }) : setTimeout(fn, 1500));
  if (document.readyState === 'complete') idle(refine);
  else window.addEventListener('load', () => idle(refine), { once: true });

  root.classList.add('hc-js');
  measure();
  // the load may already have failed before this module ran (the error event is gone by then)
  if (video.error || video.networkState === 3 /* NETWORK_NO_SOURCE */) fail();
  else if (!userPaused) {
    try { video.currentTime = CLIP.first; } catch (e) { /* applied once metadata is known */ }
    play(); // play() flips video.paused synchronously
  }
  sync();
  render(now());
}
