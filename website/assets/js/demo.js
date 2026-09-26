// Live demo: upload -> queued -> running -> done | error, polling the FastAPI job API.
// Every response is treated as untrusted: unexpected shapes become readable errors, never crashes.

import { h, clamp, isNum, fmtBytes, fmtTime, session, banner } from './core.js?v=team2';
import { EventViewer } from './viewer.js?v=team2';

const META_API = document.querySelector('meta[name="demo-api"]');
const API = ((META_API && META_API.content) || '/api').replace(/\/+$/, '');
const MAX_SECONDS = 150;
const MAX_BYTES = 400 * 1024 * 1024;
const POLL_MS = 1500;
const MAX_POLL_FAILURES = 20;
const EXT_RE = /\.(mp4|mov|m4v)$/i;
const JOB_KEY = 'jw-demo-job';
const OFFLINE_MSG = META_API && /^https?:/.test(API)
  // a separate demo server (Hugging Face Space): it sleeps after two days without visitors
  ? `The demo server is not answering yet. It sleeps after two days without visitors and takes 2–3 minutes to wake up: open ${new URL(API).origin} to wake it, then reload this page.`
  : 'The live-demo API is not reachable from this page. This copy of the site is probably served as static files. Start the demo server from the repository root with "uvicorn demo.app:app --host 0.0.0.0 --port 7860" and open the site from there.';

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
function parseJSON(text) { try { return JSON.parse(text); } catch { return null; } }
function detailOf(body) {
  if (!body || typeof body !== 'object') return null;
  const d = body.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map((x) => (x && x.msg) || JSON.stringify(x)).join('; ');
  return null;
}
function explainHttp(status, body, contentType) {
  const d = detailOf(body);
  if (d) return cap(d.replace(/\.$/, '')) + '.';
  if (status === 0) return 'The server did not answer. Check your connection and that the demo server is running.';
  if (status === 413) return 'The file is larger than the server accepts (400 MB).';
  if ([404, 405, 501].includes(status) || !/json/.test(contentType || '')) return OFFLINE_MSG;
  if (status >= 500) return `The server hit an internal error (HTTP ${status}). Please try again in a moment.`;
  return `The upload was rejected (HTTP ${status}).`;
}
function safeVideoUrl(u) {
  if (typeof u !== 'string' || !u) return null;
  // the server answers with a path on its own host, which differs from this page's when <meta name="demo-api"> is set
  try { const url = new URL(u, new URL(`${API}/`, location.href)); return /^https?:$/.test(url.protocol) ? url.href : null; } catch { return null; }
}
function readDuration(file) {
  return new Promise((resolve) => {
    let url = null;
    const v = document.createElement('video');
    v.preload = 'metadata'; v.muted = true;
    const done = (d) => { clearTimeout(timer); v.removeAttribute('src'); try { if (url) URL.revokeObjectURL(url); } catch { /* ignore */ } resolve(d); };
    const timer = setTimeout(() => done(null), 6000);
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    try { url = URL.createObjectURL(file); v.src = url; } catch { done(null); }
  });
}

export async function probeApi() {
  // demo/app.py serves this site through uvicorn, which names itself in the Server header:
  // detecting it avoids a (harmless but noisy) 404 in the console.
  try {
    const r0 = await fetch('data/links.json', { method: 'HEAD', cache: 'no-store' });
    if (/uvicorn|hypercorn/i.test(r0.headers.get('server') || '')) return 'online';
  } catch { /* fall through to the API probe */ }
  try {
    const r = await fetch(`${API}/jobs/__ping__${Date.now()}`, { cache: 'no-store' });
    const ct = r.headers.get('content-type') || '';
    if (/json/.test(ct)) { const j = await r.json().catch(() => null); if (j && typeof j === 'object') return 'online'; }
    return 'offline';
  } catch { return 'offline'; }
}

export function renderDemo(root) {
  const st = {
    phase: 'idle', file: null, fileDur: null, problems: [], jobId: null, token: 0, xhr: null, timer: 0, tick: 0,
    uploadPct: 0, stage: '', progress: null, position: null, error: null, errorKind: null, result: null,
    startedAt: 0, finishedAt: 0, failures: 0, netNote: '', resuming: false, server: 'checking',
  };
  let viewer = null;

  // ---------- upload card
  const input = h('input', { type: 'file', id: 'demo-file', accept: 'video/mp4,video/quicktime,.mp4,.mov,.m4v' });
  const dzTitle = h('span', { class: 'dz-title' }, 'Bring your footage.');
  const dropzone = h('label', { class: 'dropzone', for: 'demo-file' });
  const dzIcon = document.createElement('span');
  dzIcon.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>';
  dropzone.append(dzIcon.firstChild, dzTitle, h('span', { class: 'dz-instruction' }, 'Drop a video here or browse files'), h('span', { class: 'dz-sub' }, 'MP4 / MOV · 150 seconds max · 400 MB max'), input);
  const fileInfo = h('div', { class: 'file-info', 'aria-live': 'polite' });
  const btnAnalyse = h('button', { type: 'button', class: 'btn btn-primary', disabled: true }, 'Analyse clip');
  const btnClear = h('button', { type: 'button', class: 'btn btn-ghost', hidden: true }, 'Clear');
  const btnCancel = h('button', { type: 'button', class: 'btn btn-ghost', hidden: true }, 'Cancel upload');
  const pill = h('span', { class: 'server-pill', role: 'status' }, h('span', { class: 'dot' }), h('span', { class: 'txt' }, 'Checking the demo server…'));
  const uploadCard = h('article', { class: 'card demo-upload' },
    h('div', { class: 'card-head' }, h('h3', null, 'Upload a recording'), pill),
    dropzone, fileInfo,
    h('div', { class: 'demo-actions' }, btnAnalyse, btnCancel, btnClear),
    h('details', { class: 'demo-notes' }, h('summary', null, 'Formats, processing & limitations'), h('ul', { class: 'limits' },
      h('li', null, 'Formats: .mp4 or .mov (H.264 works best). Length up to 150 s, size up to 400 MB.'),
      h('li', null, 'Runs on 2 CPU cores with the small detector (YOLO26-S, 960 px, 10 fps): expect roughly 3× the clip length for 1080p footage (about 8 minutes for 150 s) and up to 6× for 4K originals, plus waiting time if other jobs are queued.'),
      h('li', null, 'Footage from our calibrated camera gets every layout-based class; accident, road-obstacle and fire/smoke need the open-vocabulary hazard pass, which the CPU demo skips. Other footage gets tracking and the accident-risk curve only.'),
      h('li', null, 'The server keeps only the last 20 jobs; older uploads are deleted automatically.'))));

  // ---------- status card
  const steps = ['Upload', 'Queue', 'Analyse', 'Done'].map((s) => h('li', null, s));
  const stepsEl = h('ol', { class: 'status-steps', 'aria-label': 'Progress steps' }, steps);
  const titleEl = h('p', { class: 'status-title' });
  const subEl = h('p', { class: 'status-sub' });
  const bar = h('div');
  const progress = h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-label': 'Job progress' }, bar);
  const metaL = h('span'), metaR = h('span');
  const progressWrap = h('div', null, progress, h('div', { class: 'progress-meta' }, metaL, metaR));
  const statusExtra = h('div');
  const statusActions = h('div', { class: 'demo-actions' });
  const statusCard = h('article', { class: 'card demo-status' },
    h('div', { class: 'card-head' }, h('h3', null, 'From clip to insight')),
    stepsEl, h('div', { 'aria-live': 'polite' }, titleEl, subEl), progressWrap, statusExtra, statusActions);

  // ---------- result area
  const resultEl = h('section', { class: 'demo-result', hidden: true, 'aria-label': 'Demo result' });

  root.replaceChildren(h('div', { class: 'demo-grid' }, uploadCard, statusCard), resultEl);

  // ---------- events
  input.addEventListener('change', () => { const f = input.files && input.files[0]; if (f) pickFile(f); });
  ['dragenter', 'dragover'].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); if (!busy()) dropzone.classList.add('is-drag'); }));
  ['dragleave', 'dragend'].forEach((t) => dropzone.addEventListener(t, () => dropzone.classList.remove('is-drag')));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault(); dropzone.classList.remove('is-drag');
    if (busy()) return;
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) pickFile(f);
  });
  // a file dropped next to the zone must not navigate away from the page
  const hasFiles = (e) => !!(e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files'));
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => { if (hasFiles(e)) e.preventDefault(); });
  btnAnalyse.addEventListener('click', () => upload());
  btnClear.addEventListener('click', () => reset());
  btnCancel.addEventListener('click', () => { if (st.xhr) { st.token++; try { st.xhr.abort(); } catch { /* ignore */ } st.xhr = null; } setPhase(st.file ? 'selected' : 'idle'); });

  function busy() { return ['uploading', 'queued', 'running'].includes(st.phase); }

  // ---------- server status
  function setServer(state) {
    st.server = state;
    pill.classList.toggle('online', state === 'online');
    pill.classList.toggle('offline', state === 'offline');
    pill.querySelector('.txt').textContent = state === 'online' ? 'Demo server online' : state === 'offline' ? 'Demo server not reachable' : 'Checking the demo server…';
    render();
  }
  probeApi().then((s) => {
    setServer(s);
    const saved = session.get(JOB_KEY);
    if (s === 'online' && saved && st.phase === 'idle') { st.resuming = true; startJob(saved, ++st.token, true); }
  });

  // ---------- actions
  function pickFile(f) {
    st.file = f; st.fileDur = undefined; st.problems = [];   // undefined: reading, null: the browser cannot read it
    if (!EXT_RE.test(f.name || '')) st.problems.push('Only .mp4 or .mov files are accepted.');
    if (f.size > MAX_BYTES) st.problems.push(`The file is ${fmtBytes(f.size)}; the limit is 400 MB.`);
    if (f.size === 0) st.problems.push('The file is empty.');
    setPhase('selected');
    readDuration(f).then((d) => {
      if (st.file !== f) return;
      st.fileDur = d;
      if (isNum(d) && d > MAX_SECONDS + 3) st.problems.push(`The clip is ${fmtTime(d)} long; the demo accepts up to ${fmtTime(MAX_SECONDS)}. Trim it first.`);
      render();
    });
  }

  function reset() {
    st.token++;
    clearTimeout(st.timer);
    if (st.xhr) { try { st.xhr.abort(); } catch { /* ignore */ } st.xhr = null; }
    session.del(JOB_KEY);
    Object.assign(st, { file: null, fileDur: null, problems: [], jobId: null, error: null, errorKind: null, result: null, progress: null, position: null, stage: '', netNote: '', startedAt: 0, finishedAt: 0 });
    input.value = '';
    if (viewer) { viewer.destroy(); viewer = null; }
    resultEl.hidden = true; resultEl.replaceChildren();
    setPhase('idle');
  }

  function upload() {
    if (!st.file || st.problems.length || busy()) return;
    const token = ++st.token;
    if (viewer) { viewer.destroy(); viewer = null; }
    resultEl.hidden = true; resultEl.replaceChildren();
    Object.assign(st, { uploadPct: 0, error: null, errorKind: null, result: null, jobId: null, progress: null, position: null, stage: '', netNote: '', startedAt: Date.now(), finishedAt: 0 });
    setPhase('uploading');
    const xhr = new XMLHttpRequest();
    st.xhr = xhr;
    try { xhr.open('POST', `${API}/jobs`); } catch { fail(OFFLINE_MSG, 'upload'); return; }
    xhr.upload.onprogress = (e) => { if (token !== st.token || !e.lengthComputable) return; st.uploadPct = e.loaded / e.total; render(); };
    xhr.onload = () => {
      if (token !== st.token) return;
      st.xhr = null;
      const body = parseJSON(xhr.responseText);
      const ct = xhr.getResponseHeader('content-type') || '';
      if (xhr.status >= 200 && xhr.status < 300 && body && typeof body.id === 'string' && body.id) { startJob(body.id, token); return; }
      const msg = explainHttp(xhr.status, body, ct);
      if (msg === OFFLINE_MSG) setServer('offline');
      fail(msg, 'upload');
    };
    xhr.onerror = () => { if (token !== st.token) return; st.xhr = null; fail('Network error while uploading. Check your connection and that the demo server is running.', 'upload'); };
    xhr.ontimeout = () => { if (token !== st.token) return; st.xhr = null; fail('The upload timed out. Please try again.', 'upload'); };
    const fd = new FormData();
    fd.append('file', st.file, st.file.name);
    try { xhr.send(fd); } catch (e) { fail(`Could not start the upload: ${e && e.message ? e.message : e}`, 'upload'); }
  }

  function startJob(id, token, resumed = false) {
    st.jobId = id;
    session.set(JOB_KEY, id);
    if (!st.startedAt) st.startedAt = Date.now();
    st.failures = 0;
    if (resumed) st.stage = 'reconnecting to your previous job';
    setPhase('queued');
    poll(token);
  }

  function schedule(token, ms = POLL_MS) { clearTimeout(st.timer); st.timer = setTimeout(() => poll(token), ms); }

  function retry(token, note) {
    st.failures++;
    if (st.failures > MAX_POLL_FAILURES) { fail('Lost contact with the demo server. The job may still be running: press "Check again" in a moment.', 'poll'); return; }
    st.netNote = note;
    render();
    schedule(token, Math.min(10000, POLL_MS * Math.min(st.failures, 6)));
  }

  async function poll(token) {
    if (token !== st.token || !st.jobId) return;
    let r = null, body = null;
    try {
      r = await fetch(`${API}/jobs/${encodeURIComponent(st.jobId)}`, { cache: 'no-store' });
      body = await r.json().catch(() => null);
    } catch { r = null; }
    if (token !== st.token) return;
    if (!r) { retry(token, 'Connection lost, retrying…'); return; }
    if (r.status === 404) {
      session.del(JOB_KEY);
      if (st.resuming) { st.resuming = false; reset(); return; }
      fail('The server no longer knows this job (it was probably restarted). Please upload the clip again.', 'lost');
      return;
    }
    if (!r.ok || !body || typeof body !== 'object') { retry(token, `The server answered HTTP ${r.status}, retrying…`); return; }
    st.failures = 0; st.netNote = '';
    const status = body.status;
    if (status === 'done') { finish(body.result); return; }
    if (status === 'error') {
      session.del(JOB_KEY); st.resuming = false;
      fail(typeof body.error === 'string' && body.error ? `Processing failed on the server: ${body.error}` : 'Processing failed on the server.', 'run');
      return;
    }
    if (status === 'queued') {
      st.position = isNum(body.position) ? body.position : null;
      setPhase('queued');
    } else {
      st.stage = typeof body.stage === 'string' && body.stage ? body.stage : 'processing';
      st.progress = isNum(body.progress) ? clamp(body.progress, 0, 1) : null;
      setPhase('running');
    }
    schedule(token);
  }

  function finish(result) {
    session.del(JOB_KEY);
    st.resuming = false;
    if (!result || typeof result !== 'object') { fail('The server reported the job as done but sent no result.', 'run'); return; }
    st.result = result;
    st.finishedAt = Date.now();
    setPhase('done');
    showResult();
  }

  function fail(msg, kind) {
    clearTimeout(st.timer);
    st.error = msg; st.errorKind = kind;
    setPhase('error');
  }

  // ---------- result
  function showResult() {
    const res = st.result;
    const events = Array.isArray(res.events) ? res.events : [];
    const videoUrl = safeVideoUrl(res.video);
    const took = st.startedAt && st.finishedAt ? (st.finishedAt - st.startedAt) / 1000 : null;
    const dur = isNum(res.duration) ? res.duration : null;
    const jsonBtn = h('button', { type: 'button', class: 'btn btn-ghost btn-sm', onclick: () => {
      const blob = new Blob([JSON.stringify({ file: st.file ? st.file.name : null, duration: res.duration, scene_recognised: res.scene_recognised, events: res.events, risk: res.risk }, null, 1)], { type: 'application/json' });
      const a = h('a', { href: URL.createObjectURL(blob), download: `${(st.file && st.file.name ? st.file.name.replace(/\.[^.]+$/, '') : 'clip')}_results.json` });
      document.body.append(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    } }, 'Download results (JSON)');
    const head = h('div', { class: 'demo-result-head' },
      h('div', null, h('h3', null, 'Result', st.file ? h('span', { class: 'muted', style: { fontWeight: '400' } }, ` · ${st.file.name}`) : null),
        h('p', { class: 'muted small', style: { margin: 0 } }, [dur != null ? `${fmtTime(dur)} of video` : null, `${events.length} events`, took != null ? `processed in ${fmtTime(took)}` : null].filter(Boolean).join(' · '))),
      h('div', { class: 'demo-actions', style: { marginTop: 0 } },
        videoUrl ? h('a', { class: 'btn btn-ghost btn-sm', href: videoUrl, download: 'annotated.mp4', target: '_blank', rel: 'noopener' }, 'Download annotated video') : null, jsonBtn));
    const parts = [head];
    if (res.scene_recognised === false) {
      parts.push(banner('warn', h('p', null, h('strong', null, 'This clip is not from the calibrated camera — layout-dependent classes were disabled. '),
        'You get the tracked road users and the accident-risk curve; event classes need the scene layout (accident detection also needs the hazard pass, which the CPU demo skips).')));
    }
    const vr = h('div');
    parts.push(h('div', { class: 'card' }, vr));
    resultEl.replaceChildren(...parts);
    resultEl.hidden = false;
    try {
      viewer = new EventViewer(vr, {
        duration: res.duration, events, risk: res.risk, videoSrc: videoUrl,
        videoLabel: st.file ? st.file.name : 'your clip',
        noVideoTitle: 'Annotated video unavailable',
        noVideoText: 'The server did not return a playable video. The timeline, risk curve and table still work.',
      });
    } catch (e) {
      console.error(e);
      vr.replaceChildren(banner('error', h('p', null, 'The result arrived but could not be displayed. You can still download it as JSON above.')));
    }
    requestAnimationFrame(() => resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }

  // ---------- rendering
  function setPhase(p) { st.phase = p; render(); }

  function elapsed() { return st.startedAt ? ((st.finishedAt || Date.now()) - st.startedAt) / 1000 : 0; }
  function typical() {
    if (!isNum(st.fileDur)) return 'typically 3–6× the clip length';
    return `typically ${fmtTime(st.fileDur * 3)}–${fmtTime(st.fileDur * 6)} for this clip`;
  }
  function setBar(frac, indeterminate = false) {
    progress.classList.toggle('indeterminate', indeterminate);
    bar.style.width = indeterminate ? '' : `${Math.round(clamp(frac, 0, 1) * 100)}%`;
    if (indeterminate) progress.removeAttribute('aria-valuenow'); else progress.setAttribute('aria-valuenow', String(Math.round(clamp(frac, 0, 1) * 100)));
  }

  function render() {
    const p = st.phase;
    // upload card
    const locked = busy();
    dropzone.classList.toggle('is-disabled', locked);
    input.disabled = locked;
    fileInfo.replaceChildren();
    if (st.file) {
      fileInfo.append(h('span', { class: 'fname' }, st.file.name), h('span', { class: 'muted' }, fmtBytes(st.file.size)),
        h('span', { class: 'muted' }, isNum(st.fileDur) ? fmtTime(st.fileDur) : st.fileDur === undefined ? 'reading length…' : 'length checked on upload'));
    }
    btnAnalyse.disabled = !(st.file && !st.problems.length && (p === 'selected' || p === 'error' || p === 'done'));
    btnAnalyse.textContent = p === 'done' || p === 'error' ? 'Analyse again' : 'Analyse clip';
    btnClear.hidden = !(st.file && !locked);
    btnCancel.hidden = p !== 'uploading';
    dzTitle.textContent = st.file && !locked ? 'Drop or click to choose a different clip' : 'Bring your footage.';

    // steps
    const order = { idle: -1, selected: -1, uploading: 0, queued: 1, running: 2, done: 4, error: -2 };
    const cur = order[p];
    steps.forEach((li, i) => {
      li.className = cur === -2 ? '' : i < cur ? 'is-done' : i === cur ? 'is-current' : '';
      if (p === 'done') li.className = 'is-done';
    });
    if (p === 'error') {
      const at = st.errorKind === 'upload' ? 0 : st.errorKind === 'run' ? 2 : 1;
      steps.forEach((li, i) => { li.className = i < at ? 'is-done' : i === at ? 'is-error' : ''; });
    }

    statusExtra.replaceChildren();
    statusActions.replaceChildren();
    progressWrap.hidden = false;
    metaL.textContent = ''; metaR.textContent = '';

    if (p === 'idle') {
      titleEl.textContent = 'Waiting for a clip';
      subEl.textContent = 'Your analysis will appear here once you submit a recording.';
      progressWrap.hidden = true;
      statusExtra.append(h('div', { class: 'demo-deliverables' },
        ...[
          ['01', 'Annotated footage', 'Follow road users, tracks and detected events.'],
          ['02', 'An event timeline', 'Jump straight to the moments that matter.'],
          ['03', 'A risk curve', 'Explore how the estimated risk changes over time.'],
        ].map(([n, title, text]) => h('div', {class:'demo-deliverable'}, h('span', {class:'deliverable-index', 'aria-hidden':'true'}, n), h('div', null, h('h4', null, title), h('p', null, text))))));
      if (st.server === 'offline') statusExtra.append(banner('info', h('p', null, OFFLINE_MSG)));
    } else if (p === 'selected') {
      progressWrap.hidden = true;
      if (st.problems.length) {
        titleEl.textContent = 'This file cannot be analysed';
        subEl.textContent = '';
        statusExtra.append(banner('error', ...st.problems.map((x) => h('p', null, x))));
      } else {
        titleEl.textContent = 'Ready to analyse';
        subEl.textContent = `${st.file.name} · ${isNum(st.fileDur) ? `${fmtTime(st.fileDur)} long, ${typical()}` : 'processing takes roughly 3–6× the clip length'}.`;
        if (st.server === 'offline') statusExtra.append(banner('info', h('p', null, OFFLINE_MSG)));
      }
    } else if (p === 'uploading') {
      titleEl.textContent = 'Uploading…';
      subEl.textContent = `${Math.round(st.uploadPct * 100)} % of ${fmtBytes(st.file ? st.file.size : NaN)}`;
      setBar(st.uploadPct);
      metaL.textContent = `elapsed ${fmtTime(elapsed())}`;
      metaR.textContent = st.uploadPct >= 0.999 ? 'checking the video on the server…' : '';
    } else if (p === 'queued') {
      titleEl.textContent = 'Queued';
      subEl.textContent = st.stage === 'reconnecting to your previous job' ? 'Reconnecting to the job you started before reloading the page…'
        : isNum(st.position) ? `Position ${st.position} in the queue when submitted. Jobs run one at a time.` : 'Waiting for the worker. Jobs run one at a time.';
      setBar(0, true);
      metaL.textContent = `elapsed ${fmtTime(elapsed())}`;
      metaR.textContent = st.netNote || '';
    } else if (p === 'running') {
      titleEl.textContent = 'Processing on CPU';
      subEl.textContent = cap(st.stage) + '…';
      if (isNum(st.progress)) setBar(st.progress); else setBar(0, true);
      metaL.textContent = `${isNum(st.progress) ? Math.round(st.progress * 100) + ' % · ' : ''}elapsed ${fmtTime(elapsed())}`;
      metaR.textContent = st.netNote || typical();
    } else if (p === 'done') {
      titleEl.textContent = 'Done';
      subEl.textContent = `Processed in ${fmtTime(elapsed())}. The result is shown below.`;
      setBar(1);
      statusActions.append(h('button', { type: 'button', class: 'btn btn-ghost', onclick: () => resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, 'Show result'),
        h('button', { type: 'button', class: 'btn btn-ghost', onclick: () => reset() }, 'Analyse another clip'));
    } else if (p === 'error') {
      titleEl.textContent = 'Something went wrong';
      subEl.textContent = '';
      progressWrap.hidden = true;
      statusExtra.append(banner('error', h('p', null, st.error || 'Unknown error.')));
      if (st.errorKind === 'poll' && st.jobId) {
        statusActions.append(h('button', { type: 'button', class: 'btn btn-primary', onclick: () => { st.failures = 0; const t = ++st.token; setPhase('queued'); poll(t); } }, 'Check again'));
      } else if (st.file && !st.problems.length) {
        statusActions.append(h('button', { type: 'button', class: 'btn btn-primary', onclick: () => upload() }, 'Retry'));
      }
      statusActions.append(h('button', { type: 'button', class: 'btn btn-ghost', onclick: () => reset() }, 'Start over'));
    }

    // elapsed-time ticker while something is in flight
    if (busy() && !st.tick) st.tick = setInterval(() => { if (!busy()) { clearInterval(st.tick); st.tick = 0; return; } render(); }, 1000);
  }

  render();
}
