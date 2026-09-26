// Entry point: loads the JSON data, renders every section, wires the nav, theme and lightbox.
// Each section is rendered independently so one failure never takes the page down.

import { $, $$, h, fetchJSON, fmtInt, sum, sectionError } from './js/core.js?v=team2';
import { injectPaletteCSS, CLASSES } from './js/palette.js?v=team2';
import { renderPipeline } from './js/pipeline.js?v=team2';
import { renderEDA } from './js/eda.js?v=team2';
import { renderResults } from './js/results.js?v=team2';
import { renderDemo } from './js/demo.js?v=demo1';
import { renderClassGrid, renderReport, renderTeam, renderLinks } from './js/content.js?v=team3';
import { loadChart } from './js/charts.js?v=team2';
import { renderAnalysis } from './js/analysis.js?v=team2';

injectPaletteCSS();
loadChart(); // start fetching Chart.js early; charts render when it arrives

// ---------------------------------------------------------------- nav
function initNav() {
  const nav = $('#site-nav');
  const btn = $('#menu-toggle');
  const close = () => { nav.classList.remove('is-open'); btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-label', 'Open menu'); };
  btn.addEventListener('click', () => {
    const open = !nav.classList.contains('is-open');
    nav.classList.toggle('is-open', open);
    btn.setAttribute('aria-expanded', String(open));
    btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  });
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  document.addEventListener('click', (e) => { if (!e.target.closest('#site-nav') && !e.target.closest('#menu-toggle')) close(); });

  // scroll-spy
  const links = $$('#site-nav a');
  const byId = Object.fromEntries(links.map((a) => [a.getAttribute('href').slice(1), a]));
  if ('IntersectionObserver' in window) {
    const visible = new Map();
    const io = new IntersectionObserver((entries) => {
      for (const en of entries) visible.set(en.target.id, en.isIntersecting ? en.intersectionRatio : 0);
      let best = null, bestR = 0;
      for (const [id, r] of visible) if (r > bestR) { best = id; bestR = r; }
      links.forEach((a) => a.classList.toggle('is-current', a === byId[best]));
      links.forEach((a) => { if (a === byId[best]) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
    }, { rootMargin: '-45% 0px -50% 0px', threshold: [0, 0.01, 0.5, 1] });
    for (const id of Object.keys(byId)) { const sec = document.getElementById(id); if (sec) io.observe(sec); }
  }
}

// ---------------------------------------------------------------- lightbox
function initLightbox() {
  const dlg = $('#lightbox');
  const img = $('#lightbox-img');
  const cap = $('#lightbox-caption');
  const supported = dlg && typeof dlg.showModal === 'function';
  if (supported) dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  return (src, alt, caption) => {
    if (!supported) { window.open(src, '_blank', 'noopener'); return; }
    img.src = src; img.alt = alt || ''; cap.textContent = caption || '';
    dlg.showModal();
  };
}

// ---------------------------------------------------------------- hero numbers
function renderHero(eda, results) {
  const set = (k, v) => { const el = $(`[data-stat="${k}"]`); if (el) el.replaceChildren(...[].concat(v)); };
  const rv = (results && results.videos) || [];
  const ev = (eda && eda.videos) || [];
  set('videos', String(rv.length || ev.length || 0));
  const tracks = sum(ev.map((v) => sum(Object.values(v.tracks_by_kind || {}))));
  set('tracks', tracks ? fmtInt(tracks) : '–');
  const events = sum(rv.map((v) => (Array.isArray(v.events) ? v.events.length : 0)));
  set('events', fmtInt(events));
  const seen = new Set(rv.flatMap((v) => (Array.isArray(v.events) ? v.events.map((e) => e && e[2]) : [])).filter(Boolean));
  set('classes', [String(seen.size), h('small', null, ` / ${CLASSES.length}`)]);
}

function classCounts(results) {
  const out = {};
  for (const v of (results && results.videos) || []) for (const e of v.events || []) if (Array.isArray(e) && e[2]) out[e[2]] = (out[e[2]] || 0) + 1;
  return out;
}

// ---------------------------------------------------------------- boot
async function main() {
  initNav();
  const lightbox = initLightbox();

  try { renderPipeline($('#pipeline-diagram')); } catch (e) { sectionError($('#pipeline-diagram'), 'the pipeline diagram', e); }
  try { renderDemo($('#demo-root')); } catch (e) { sectionError($('#demo-root'), 'the live demo', e); }

  const files = ['eda', 'results', 'team', 'metrics', 'report', 'links', 'ablations', 'confusion'];
  const got = await Promise.allSettled(files.map((f) => fetchJSON(`data/${f}.json`)));
  const data = {}, errors = {};
  files.forEach((f, i) => { if (got[i].status === 'fulfilled') data[f] = got[i].value; else errors[f] = got[i].reason; });

  const guard = (sel, what, fn) => { const root = $(sel); try { fn(root); } catch (e) { sectionError(root, what, e); } };
  renderHero(data.eda, data.results);
  guard('#class-grid', 'the class list', (r) => renderClassGrid(r, classCounts(data.results)));
  guard('#eda-root', 'the EDA data (data/eda.json)', (r) => { if (errors.eda) throw errors.eda; renderEDA(r, data.eda, lightbox); });
  guard('#results-root', 'the results (data/results.json)', (r) => { if (errors.results) throw errors.results; renderResults(r, data.results, { eda: data.eda, metrics: data.metrics, report: data.report }); });
  guard('#analysis-root', 'the ablations (data/ablations.json)', (r) => renderAnalysis(r, { ablations: data.ablations, confusion: data.confusion }));
  guard('#report-root', 'the report (data/report.json)', (r) => { if (errors.report) throw errors.report; renderReport(r, data.report); });
  guard('#team-root', 'the team (data/team.json)', (r) => { if (errors.team) throw errors.team; renderTeam(r, data.team); });
  guard('#links-root', 'the links (data/links.json)', (r) => { if (errors.links) throw errors.links; renderLinks(r, data.links); });
}

main().catch((e) => console.error(e));
