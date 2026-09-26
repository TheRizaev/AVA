// Pipeline diagram as inline SVG. Wide screens get a left-to-right spine with side inputs;
// narrow screens get a vertical flow at true pixel size so the text stays readable.

import { s } from './core.js?v=team2';

const A = {
  decode: { n: 1, type: 'io', title: 'Decode', text: 'PyAV, reference frames only (≈10 fps) · 4K → 1920×1080 · background thread' },
  detect: { n: 2, type: 'learned', title: 'Detect & track', text: 'YOLO26-L, COCO weights · FP16, imgsz 1280, batch 8 · ByteTrack' },
  register: { n: 3, type: 'geo', title: 'Register', text: 'SIFT + RANSAC homography every 5 s · 3 lighting refs → one reference frame' },
  layout: { n: 4, type: 'geo', title: 'Scene layout + flow field', text: 'Drawn once: carriageways, median, islands, 3 crossings, stop line, solid dividers (rays from the vanishing point), lamps · flow field learned from sample tracks' },
  signal: { n: 5, type: 'rule', title: 'Signal phase', text: 'Lamps of the 3-lamp head on the median tip, per-lamp normalisation · fixed-time cycle of 75–80 s: red 36–39 s, green 36–38 s, yellow 3 s' },
  rules: { n: 6, type: 'rule', title: 'Event rules', text: '14 rules (12 reported) on trajectories, layout and signal · YOLOE-26L for obstacles, fire, smoke & crashed cars' },
  post: { n: 7, type: 'rule', title: 'Post-process', text: 'Merge same-class overlaps · clip to duration · drop < 0.2 s' },
};
const B = [
  { type: 'learned', title: 'Online perception', text: 'Same detector at imgsz 960 on every 3rd frame · online ByteTrack · registration every 5 s' },
  { type: 'rule', title: 'Conflict cues', text: 'Crossing-path TTC · same-lane rear-end closing · pedestrian conflict · hard braking' },
  { type: 'rule', title: 'Fuse & smooth', text: 'Cues combined as independent evidence · fast attack, slow release' },
  { type: 'rule', title: 'Calibrate', text: 'Monotone map: the level with ≈0.2 false alarms/min on the samples → 0.5' },
];

const CHAR_W = 6.35;   // average glyph width of 12px Inter
const LINE_H = 15;

function wrap(text, maxChars) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = '';
  for (const w of words) {
    const next = cur ? cur + ' ' + w : w;
    if (next.length > maxChars && cur) { lines.push(cur); cur = w; } else cur = next;
  }
  if (cur) lines.push(cur);
  return lines;
}
const cpl = (w) => Math.max(10, Math.floor((w - 26) / CHAR_W));
const TITLE_W = 8.1;    // average glyph width of the 14px semibold title
const TITLE_H = 18;

/** Wrap the title and the text of a node for a box of width w; returns the lines and the height. */
function fit(node, w) {
  const prefix = node.n ? 3 : 0;
  const titleLines = wrap(node.title, Math.max(8, Math.floor((w - 26) / TITLE_W) - prefix));
  const lines = wrap(node.text, cpl(w));
  return { titleLines, lines, h: boxH(lines.length) + (titleLines.length - 1) * TITLE_H };
}
const boxH = (nLines) => 44 + nLines * LINE_H + 8;

function drawBox(g, x, y, w, hgt, node, f) {
  g.append(s('rect', { class: `pl-box ${node.type}`, x, y, width: w, height: hgt, rx: 10 }));
  f.titleLines.forEach((tl, i) => {
    const title = s('text', { class: 'pl-title', x: x + 13, y: y + 25 + i * TITLE_H });
    if (node.n && i === 0) title.append(s('tspan', { class: 'pl-num' }, `${node.n}  `));
    title.append(s('tspan', null, tl));
    g.append(title);
  });
  const dy = (f.titleLines.length - 1) * TITLE_H;
  f.lines.forEach((ln, i) => g.append(s('text', { class: 'pl-text', x: x + 13, y: y + 46 + dy + i * LINE_H }, ln)));
}

function arrow(g, d) { g.append(s('path', { class: 'pl-edge', d, 'marker-end': 'url(#pl-arrow)' })); }

function pill(g, cx, cy, w, label) {
  const hh = 38;
  g.append(s('rect', { class: 'pl-out', x: cx - w / 2, y: cy - hh / 2, width: w, height: hh, rx: hh / 2 }));
  g.append(s('text', { class: 'pl-out-text', x: cx, y: cy + 4.5, 'text-anchor': 'middle' }, label));
}

function defs() {
  return s('defs', null, s('marker', { id: 'pl-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    s('path', { class: 'pl-arrowhead', d: 'M0,0 L10,5 L0,10 Z' })));
}

function wide() {
  const W = 1160, m = 10, bw = 172, gap = 30;
  const colX = (i) => m + i * (bw + gap);
  const spine = [A.decode, A.detect, A.register, A.rules, A.post];
  const spineFit = spine.map((n) => fit(n, bw));
  const hS = Math.max(...spineFit.map((f) => f.h));
  const sideW = 2 * bw + gap;
  const topFit = fit(A.layout, sideW), botFit = fit(A.signal, sideW);
  const hT = topFit.h, hB = botFit.h;
  const topY = 30, spineY = topY + hT + 36, botY = spineY + hS + 36;

  const g = s('g');
  g.append(s('text', { class: 'pl-lane', x: m, y: 16 }, 'Part A · offline, whole clip'));
  spine.forEach((n, i) => drawBox(g, colX(i), spineY, bw, hS, n, spineFit[i]));
  drawBox(g, colX(2), topY, sideW, hT, A.layout, topFit);
  drawBox(g, colX(2), botY, sideW, hB, A.signal, botFit);
  const midY = spineY + hS / 2;
  for (let i = 0; i < spine.length - 1; i++) arrow(g, `M${colX(i) + bw + 2},${midY} L${colX(i + 1) - 3},${midY}`);
  const rx = colX(3) + bw / 2;
  arrow(g, `M${rx},${topY + hT + 2} L${rx},${spineY - 3}`);
  arrow(g, `M${rx},${botY - 2} L${rx},${spineY + hS + 3}`);
  // lamp pixels come from the decoded frames
  const dx = colX(0) + bw / 2, ly = botY + hB / 2;
  arrow(g, `M${dx},${spineY + hS + 2} L${dx},${ly} L${colX(2) - 3},${ly}`);
  g.append(s('text', { class: 'pl-text', x: colX(1) + bw / 2, y: ly - 7, 'text-anchor': 'middle' }, 'lamp pixels'));
  const outX = colX(5) + 65;
  arrow(g, `M${colX(4) + bw + 2},${midY} L${outX - 68},${midY}`);
  pill(g, outX, midY, 130, '[s, e, label]');

  // Part B
  const divY = botY + hB + 28;
  g.append(s('line', { class: 'pl-divider', x1: m, x2: W - m, y1: divY, y2: divY }));
  g.append(s('text', { class: 'pl-lane', x: m, y: divY + 24 }, 'Part B · causal, frame by frame'));
  const bY = divY + 36;
  const bwB = (colX(5) - gap - m - 3 * gap) / 4;
  const bFit = B.map((n) => fit(n, bwB));
  const hBB = Math.max(...bFit.map((f) => f.h));
  const bx = (i) => m + i * (bwB + gap);
  B.forEach((n, i) => drawBox(g, bx(i), bY, bwB, hBB, n, bFit[i]));
  const bMid = bY + hBB / 2;
  for (let i = 0; i < B.length - 1; i++) arrow(g, `M${bx(i) + bwB + 2},${bMid} L${bx(i + 1) - 3},${bMid}`);
  arrow(g, `M${bx(3) + bwB + 2},${bMid} L${outX - 68},${bMid}`);
  pill(g, outX, bMid, 130, 'risk(t) ∈ [0,1]');
  const H = bY + hBB + 8;
  return s('svg', { viewBox: `0 0 ${W} ${H}`, 'aria-hidden': 'true', focusable: 'false' }, defs(), g);
}

function vertical(W) {
  const m = 2, bw = W - 2 * m, gapY = 28;
  const g = s('g');
  let y = 0;
  const lane = (label) => { g.append(s('text', { class: 'pl-lane', x: m, y: y + 14 }, label)); y += 26; };
  const cx = W / 2;
  const chain = (nodes) => {
    nodes.forEach((n, i) => {
      const f = fit(n, bw);
      const hh = f.h;
      drawBox(g, m, y, bw, hh, n, f);
      y += hh;
      if (i < nodes.length - 1) { arrow(g, `M${cx},${y + 2} L${cx},${y + gapY - 3}`); y += gapY; }
    });
  };
  lane('Part A · offline, whole clip');
  chain([A.decode, A.detect, A.register]);
  // side inputs in one row with the spine arrow passing through the middle gap
  const mid = 30, w2 = (bw - mid) / 2;
  const l1 = fit(A.layout, w2), l2 = fit(A.signal, w2);
  const hh = Math.max(l1.h, l2.h);
  const rowY = y + gapY;
  const rulesY = rowY + hh + gapY;
  arrow(g, `M${cx},${y + 2} L${cx},${rulesY - 3}`);
  drawBox(g, m, rowY, w2, hh, A.layout, l1);
  drawBox(g, m + w2 + mid, rowY, w2, hh, A.signal, l2);
  arrow(g, `M${m + w2 / 2},${rowY + hh + 2} L${m + w2 / 2},${rulesY - 3}`);
  arrow(g, `M${m + w2 + mid + w2 / 2},${rowY + hh + 2} L${m + w2 + mid + w2 / 2},${rulesY - 3}`);
  y = rulesY;
  chain([A.rules, A.post]);
  arrow(g, `M${cx},${y + 2} L${cx},${y + gapY - 3}`);
  y += gapY;
  pill(g, cx, y + 19, 160, '[s, e, label]');
  y += 38 + 26;
  g.append(s('line', { class: 'pl-divider', x1: m, x2: W - m, y1: y - 10, y2: y - 10 }));
  y += 4;
  lane('Part B · causal, frame by frame');
  chain(B);
  arrow(g, `M${cx},${y + 2} L${cx},${y + gapY - 3}`);
  y += gapY;
  pill(g, cx, y + 19, 170, 'risk(t) ∈ [0,1]');
  y += 42;
  return s('svg', { viewBox: `0 0 ${W} ${y}`, width: W, height: y, 'aria-hidden': 'true', focusable: 'false' }, defs(), g);
}

export function renderPipeline(host) {
  let lastMode = '', lastW = 0;
  const draw = () => {
    const W = Math.floor(host.clientWidth);
    if (!W) return;
    const mode = W >= 940 ? 'wide' : 'vertical';
    if (mode === lastMode && (mode === 'wide' || Math.abs(W - lastW) < 4)) return;
    lastMode = mode; lastW = W;
    host.replaceChildren(mode === 'wide' ? wide() : vertical(W));
  };
  draw();
  if ('ResizeObserver' in window) new ResizeObserver(() => requestAnimationFrame(draw)).observe(host);
  else window.addEventListener('resize', draw);
}
