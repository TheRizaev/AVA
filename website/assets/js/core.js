// Small DOM, formatting and theme helpers shared by every module.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function appendChildren(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === '') continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** Create an HTML element. Text always goes through text nodes (never innerHTML). */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  appendChildren(el, children);
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
/** Create an SVG element. */
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.setAttribute('class', v);
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, String(v));
  }
  appendChildren(el, children);
  return el;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const sum = (arr) => arr.reduce((a, b) => a + (isNum(b) ? b : 0), 0);

export function median(arr) {
  const a = arr.filter(isNum).slice().sort((x, y) => x - y);
  if (!a.length) return NaN;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** m:ss, or m:ss.s with tenths. */
export function fmtTime(t, tenths = false) {
  if (!isNum(t)) return '–';
  const neg = t < 0; t = Math.abs(t);
  let m = Math.floor(t / 60);
  let sec = t - m * 60;
  let secStr;
  if (tenths) {
    secStr = sec.toFixed(1);
    if (secStr === '60.0') { m += 1; secStr = '0.0'; }
    if (parseFloat(secStr) < 10) secStr = '0' + secStr;
  } else {
    sec = Math.floor(sec);
    secStr = String(sec).padStart(2, '0');
  }
  return (neg ? '-' : '') + m + ':' + secStr;
}

export function fmtDur(sec) {
  if (!isNum(sec)) return '–';
  if (sec < 10) return sec.toFixed(1) + ' s';
  if (sec < 90) return Math.round(sec) + ' s';
  return fmtTime(sec) + ' min';
}

export function fmtInt(n) {
  return isNum(n) ? Math.round(n).toLocaleString('en-US') : '–';
}

export function fmtBytes(b) {
  if (!isNum(b)) return '–';
  if (b < 1024) return b + ' B';
  if (b < 1024 ** 2) return (b / 1024).toFixed(0) + ' KB';
  if (b < 1024 ** 3) return (b / 1024 ** 2).toFixed(1) + ' MB';
  return (b / 1024 ** 3).toFixed(2) + ' GB';
}

/** A readable step for a time axis spanning `span` seconds with about `target` ticks. */
export function timeStep(span, target = 6) {
  const steps = [1, 2, 5, 10, 15, 20, 30, 60, 90, 120, 180, 300, 600, 900, 1800, 3600];
  const raw = span / Math.max(1, target);
  return steps.find((x) => x >= raw) || steps[steps.length - 1];
}

export function prettyLabel(id) {
  const s = String(id || '').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

/** Run fn on the next animation frame at most once per frame. */
export function rafThrottle(fn) {
  let pending = false, lastArgs;
  return (...args) => {
    lastArgs = args;
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; fn(...lastArgs); });
  };
}

export const store = {
  get(k, fallback = null) { try { const v = localStorage.getItem(k); return v == null ? fallback : v; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};
export const session = {
  get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* ignore */ } },
  del(k) { try { sessionStorage.removeItem(k); } catch { /* ignore */ } },
};

// ---------- theme ----------
const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
const themeListeners = new Set();

export function isDark() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return !!(mq && mq.matches);
}

export function onThemeChange(cb) { themeListeners.add(cb); return () => themeListeners.delete(cb); }
export function emitThemeChange() { for (const cb of themeListeners) { try { cb(isDark()); } catch (e) { console.warn(e); } } }
if (mq) {
  const handler = () => { if (!document.documentElement.hasAttribute('data-theme')) emitThemeChange(); };
  if (mq.addEventListener) mq.addEventListener('change', handler); else if (mq.addListener) mq.addListener(handler);
}

export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

export const prefersReducedMotion = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/** Render an error note into a container instead of letting one section break the page. */
export function sectionError(root, what, err) {
  console.error(what, err);
  root.replaceChildren(h('div', { class: 'section-error', role: 'alert' },
    `Could not load ${what}. `, h('span', { class: 'muted' }, String(err && err.message ? err.message : err))));
}

/** Text that may start with "TODO" gets a visible TODO badge. */
export function todoText(text) {
  const t = String(text ?? '');
  const m = t.match(/^\s*TODO[:\s-]*/i);
  if (!m) return [t];
  return [h('span', { class: 'todo' }, 'TODO'), t.slice(m[0].length)];
}

export const ICONS = {
  warn: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5M12 18v.5"/></svg>',
  error: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5v.5"/></svg>',
  info: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>',
  ok: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/></svg>',
};

/** Banner with a static (trusted) icon and untrusted text content. */
export function banner(kind, ...content) {
  const icon = document.createElement('span');
  icon.innerHTML = ICONS[kind] || ICONS.info; // static markup only
  return h('div', { class: `banner banner-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    icon.firstChild, h('div', null, ...content));
}
