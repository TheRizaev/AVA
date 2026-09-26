// Static-ish sections rendered from JSON: class grid, report, team, links.

import { h, todoText } from './core.js?v=team2';
import { CLASSES, METHOD_LABEL, classVar } from './palette.js?v=team2';

// '#', empty or TODO means "not filled in yet"; script URLs are never rendered as links.
const isPlaceholder = (url) => !url || typeof url !== 'string' || url.trim() === '#' || /^\s*todo/i.test(url) || /^\s*(javascript|data|vbscript):/i.test(url);

export function renderClassGrid(root, seenCounts = {}) {
  root.replaceChildren(...CLASSES.map((c) => {
    const n = seenCounts[c.id] || 0;
    return h('div', { class: 'class-card' + (c.method === 'none' || c.method === 'off' ? ' is-off' : '') },
      h('header', null, h('span', { class: 'swatch', style: { background: classVar(c.id) } }), h('h4', null, c.label),
        h('span', { class: 'method' }, METHOD_LABEL[c.method] || '')),
      h('p', null, c.desc),
      h('p', { class: 'muted small', style: { marginTop: '6px' } }, h('code', null, c.id), n ? ` · ${n} on the samples` : (c.method === 'none' ? '' : ' · none on the samples')));
  }));
}

export function renderReport(root, report) {
  const r = report || {};
  const slides = [
    {key:'worked', tab:'What worked', title:'Built around the scene.', points:[
      ['Faster decoding', 'Reference frames cut CPU decoding time nearly in half.'],
      ['Stable geometry', 'Registration keeps tracks aligned across camera shifts and lighting changes.'],
      ['Readable traffic rules', 'Signal phases and crossing geometry turn trajectories into events.']]},
    {key:'did_not', tab:'Limitations', title:'Where the model falls short.', points:[
      ['Event boundaries', 'Lane changes are easier to spot than to time precisely.'],
      ['Occlusion & perspective', 'Long vehicles and hidden road users still confuse box-based geometry.'],
      ['Accident anticipation', 'On external CCTV crashes, warnings usually arrive too close to contact.']]},
    {key:'next', tab:'Next steps', title:'The next questions to solve.', points:[
      ['Measure in metres', 'Calibrate the ground plane for more meaningful distances and speeds.'],
      ['Improve perception', 'Fine-tune on this camera and validate on more labelled footage.'],
      ['Learn from motion', 'Train a temporal model on CCTV crashes for earlier warnings.']]},
  ];
  let current = 0;
  const tabs = h('div', {class:'report-tabs', role:'tablist', 'aria-label':'Report chapters'});
  const panel = h('div', {class:'report-slide', id:'report-slide', role:'tabpanel', tabindex:'0'});
  const counter = h('span', {class:'report-counter', 'aria-live':'polite'});
  const prev = h('button', {type:'button', class:'report-arrow', 'aria-label':'Previous report slide', onclick:()=>show(current-1)}, '←');
  const next = h('button', {type:'button', class:'report-arrow', 'aria-label':'Next report slide', onclick:()=>show(current+1)}, '→');
  const buttons = slides.map((slide,i) => {
    const button = h('button', {type:'button', role:'tab', id:`report-tab-${i}`, 'aria-controls':'report-slide', onclick:()=>show(i)}, slide.tab);
    button.addEventListener('keydown', e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      show(e.key==='Home'?0:e.key==='End'?2:current+(e.key==='ArrowRight'?1:-1));
      buttons[current].focus();
    });
    tabs.append(button); return button;
  });
  function show(index) {
    current = (index + slides.length) % slides.length;
    const slide = slides[current];
    buttons.forEach((button,i)=>{button.setAttribute('aria-selected',String(i===current));button.tabIndex=i===current?0:-1;});
    panel.setAttribute('aria-labelledby',`report-tab-${current}`);
    counter.textContent = `0${current+1} / 03`;
    panel.replaceChildren(h('h3', null, slide.title),
      h('div',{class:'report-highlights'},slide.points.map(([title,text],i)=>h('article',null,
        h('span',{class:'report-point-number','aria-hidden':'true'},String(i+1).padStart(2,'0')),
        h('h4',null,title),h('p',null,text)))),
      h('details',{class:'report-details'},h('summary',null,'Read more'),
        h('ul',null,(Array.isArray(r[slide.key])?r[slide.key]:[]).map(text=>h('li',null,...todoText(text))))));
  }
  root.replaceChildren(h('div',{class:'report-deck'},
    h('div',{class:'report-toolbar'},tabs,h('div',{class:'report-controls'},counter,prev,next)),panel),
    h('a',{class:'report-evidence-link',href:'#results'},'Explore the video evidence ↗'));
  show(0);
}

const ICON = {
  github: 'M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.7 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z',
  linkedin: 'M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z',
  resume: 'M6 2h8l6 6v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm7 1.5V9h5.5L13 3.5zM8 13v1.6h8V13H8zm0 3.4V18h5.5v-1.6H8z',
  web: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm6.93 6h-2.95a15.7 15.7 0 0 0-1.38-3.56A8.03 8.03 0 0 1 18.93 8zM12 4.04c.83 1.2 1.48 2.53 1.91 3.96h-3.82c.43-1.43 1.08-2.76 1.91-3.96zM4.26 14a8.2 8.2 0 0 1 0-4h3.38a16.5 16.5 0 0 0 0 4H4.26zm.81 2h2.95c.32 1.25.78 2.45 1.38 3.56A7.99 7.99 0 0 1 5.07 16zm2.95-8H5.07a7.99 7.99 0 0 1 4.33-3.56A15.7 15.7 0 0 0 8.02 8zM12 19.96c-.83-1.2-1.48-2.53-1.91-3.96h3.82c-.43 1.43-1.08 2.76-1.91 3.96zM14.34 14H9.66a14.7 14.7 0 0 1 0-4h4.68a14.7 14.7 0 0 1 0 4zm.26 5.56c.6-1.11 1.06-2.31 1.38-3.56h2.95a8.03 8.03 0 0 1-4.33 3.56zM16.36 14a16.5 16.5 0 0 0 0-4h3.38a8.2 8.2 0 0 1 0 4h-3.38z',
};
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path'); p.setAttribute('d', ICON[name]);
  svg.append(p);
  return svg;
}
function socialLink(kind, label, url, who, aria = `${who} on ${label}`) {
  if (isPlaceholder(url)) return null;
  return h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', 'aria-label': aria }, icon(kind), label);
}
const host = (url) => String(url || '').replace(/^https?:\/\//, '').replace(/\/+$/, '');
const initials = (name) => String(name || '?').split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join('');
const AVATAR_BG = ['#2a78d6', '#0d7f62', '#6247d1'];

export function renderTeam(root, team) {
  const t = team || {};
  if (!t.members?.some(m => m.name && !/^Member \d+$/.test(m.name) && !/TODO/i.test(m.name))) {
    root.replaceChildren(h('div', {class:'team-pending'},
      h('span', {class:'team-stamp', 'aria-hidden':'true'}, 'AVA / 26'),
      h('div', null, h('h3', null, 'AVA'),
        h('p', null, t.university || 'WIUT Hackathon · Computer-vision track'),
        h('p', {class:'muted small'}, 'Team profiles and individual contributions will be added before submission.'))));
    return;
  }

  const members = Array.isArray(t) ? t : (Array.isArray(t.members) ? t.members : []);
  if (!members.length) { root.replaceChildren(h('p', { class: 'muted' }, 'Team information coming soon.')); return; }
  const intro = [];
  if (t.team_name || t.university) intro.push(h('p', { class: 'section-lead', style: { marginTop: '-12px' } }, ...todoText([t.team_name, t.university].filter(Boolean).join(' · '))));
  root.replaceChildren(...intro, h('div', { class: 'team-grid' }, members.map((m, i) => {
    const photo = m.photo && !isPlaceholder(m.photo) ? h('img', { src: m.photo, alt: `Photo of ${m.name}`, loading: 'lazy' }) : initials(m.name);
    return h('article', { class: 'card member' + (m.is_captain ? ' member-captain' : '') },
      h('div', { class: 'member-topline' },
        h('span', { class: 'member-index', 'aria-hidden': 'true' }, String(i + 1).padStart(2, '0')),
        m.is_captain ? h('span', { class: 'captain-badge' }, h('span', { 'aria-hidden': 'true' }, '↗'), 'Team Captain') : null),
      h('div', { class: 'member-head' },
        m.photo && !isPlaceholder(m.photo) ? h('div', { class: 'avatar' }, photo) : null,
        h('div', null, h('h3', null,
          h('span', { class: 'member-surname' }, (m.name || 'Member').split(' ')[0]),
          h('span', { class: 'member-given' }, (m.name || '').split(' ').slice(1).join(' '))),
          h('p', { class: 'member-role' }, ...todoText(m.role || '')))),
      // the block is always present, so the cards share their grid rows and line up side by side
      Array.isArray(m.contributions) && m.contributions.length ? h('div', { class: 'member-built' }, h('h4', null, 'Built for this project'), h('ul', null, m.contributions.map((c) => h('li', null, ...todoText(c))))) : h('div', { class: 'member-built' }),
      h('div', { class: 'social' }, socialLink('github', 'GitHub', m.github, m.name), socialLink('resume', 'Resume', m.resume, m.name, `Resume of ${m.name} (PDF)`),
        socialLink('web', host(m.website), m.website, m.name, `Website of ${m.name}`), socialLink('linkedin', 'LinkedIn', m.linkedin, m.name), socialLink('web', 'Portfolio', m.portfolio, m.name)));
  })));
}

export function renderLinks(root, links) {
  const L = links || {};
  const card = (kind, item) => {
    const it = typeof item === 'string' ? { url: item } : (item || {});
    const pending = isPlaceholder(it.url);
    const body = [h('span', { class: 'lc-kind' }, kind), h('span', { class: 'lc-title' }, it.label || kind),
      it.note ? h('span', { class: 'lc-note' }, ...todoText(it.note)) : null,
      h('span', { class: 'lc-url' }, pending ? 'link coming soon' : it.url)];
    return pending ? h('div', { class: 'link-card is-pending' }, ...body)
      : h('a', { class: 'link-card', href: it.url, target: /^https?:/.test(it.url) ? '_blank' : null, rel: 'noopener noreferrer' }, ...body);
  };
  const extra = Array.isArray(L.extra) ? L.extra.filter((x) => x && !isPlaceholder(x.url)) : [];
  const parts = [h('div', { class: 'link-grid' }, card('Repository', L.repo), card('Weights', L.weights), card('Predictions', L.predictions))];
  if (extra.length) parts.push(h('ul', { class: 'extra-links' }, extra.map((x) => h('li', null, h('a', { href: x.url, target: '_blank', rel: 'noopener noreferrer' }, x.label || x.url), x.note ? ` · ${x.note}` : ''))));
  parts.push(h('p', { class: 'note' }, 'Site data: ', h('a', { href: 'data/eda.json' }, 'eda.json'), ' · ', h('a', { href: 'data/results.json' }, 'results.json'), ' (events and risk shown on this page).'));
  root.replaceChildren(...parts);
}
