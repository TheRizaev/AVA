// The one place where event-class colours (and the other series palettes) are defined.
// The order of CLASSES is the display / stacking order everywhere. It was checked with a
// colour-vision-deficiency validator: adjacent slots stay distinguishable in light and dark.

import { isDark, prettyLabel } from './core.js?v=team2';

export const CLASSES = [
  { id: 'failure_to_yield', label: 'Failure to yield', light: '#2a78d6', dark: '#3987e5', method: 'rule',
    desc: 'A vehicle drives through a zebra while a pedestrian who is actually crossing (not waiting at the kerb end or standing at a refuge) is on it within a quarter of the crossing length of where the vehicle crosses, measured along the zebra so perspective does not matter.' },
  { id: 'near_miss', label: 'Near miss', light: '#eb6834', dark: '#d95926', method: 'off',
    desc: 'Hard braking with a road user in the vehicle’s path. Detected but not reported: hard braking is everywhere at this junction and every candidate was rejected on inspection.' },
  { id: 'jaywalking', label: 'Jaywalking', light: '#1baf7a', dark: '#199e70', method: 'rule',
    desc: 'A pedestrian on the carriageway more than ~1 m from the kerb and outside the three crossings for at least 1 s; people crossing one after another within 6 s are one event. People riding bicycles or scooters are filtered out.' },
  { id: 'illegal_u_turn', label: 'Illegal U-turn', light: '#eda100', dark: '#c98500', method: 'off',
    desc: 'Approach -> around the median tip -> outbound (origin/destination on the layout). Detected cleanly but not reported: nothing in view prohibits these U-turns.' },
  { id: 'stopped_vehicle', label: 'Stopped vehicle', light: '#e87ba4', dark: '#d55181', method: 'rule',
    desc: 'A vehicle standing for 10 s or more next to the kerb while the traffic beside it keeps moving (dropping off, loading). Queues, turners waiting mid-junction and buses at their stop do not count.' },
  { id: 'solid_line_crossing', label: 'Solid-line crossing', light: '#008300', dark: '#008300', method: 'rule',
    desc: 'A lane change across a solid divider before the stop line, measured as the vehicle\'s angle seen from the vanishing point, with hysteresis.' },
  { id: 'red_light', label: 'Red-light running', light: '#4a3aa7', dark: '#9085e9', method: 'rule',
    desc: 'A vehicle crosses the approach stop line on red (0.5 s grace) and then actually enters the junction.' },
  { id: 'accident', label: 'Accident', light: '#e34948', dark: '#e66767', method: 'rule',
    desc: 'The YOLOE "crashed car" prompt (≥ 0.45) in two consecutive samples at one place on the carriageway. Checked on 36 real CCTV crashes of the TAD benchmark; silent on all sample traffic, also with cheaper detector settings.' },
  { id: 'stop_line', label: 'Stop-line violation', light: '#0e9bc4', dark: '#1a97c0', method: 'rule',
    desc: 'A vehicle stops beyond the approach stop line on red and stays there instead of waiting behind it.' },
  { id: 'illegal_turn', label: 'Illegal turn', light: '#9a7b00', dark: '#a88a10', method: 'rule',
    desc: 'A right turn into the side street from a lane other than the rightmost, solid-separated right-turn lane (measured as the lane angle seen from the vanishing point just before the stop line).' },
  { id: 'wrong_way', label: 'Wrong way', light: '#b0306e', dark: '#c2468a', method: 'rule',
    desc: 'A vehicle moving against a strongly one-way cell of the learned flow field for at least 2 s and 100 px.' },
  { id: 'congestion', label: 'Congestion', light: '#6a5acd', dark: '#7a6ce0', method: 'rule',
    desc: 'The junction interior packed with crawling vehicles (speed in box heights per second), i.e. spillback from the exit, or an approach jam that persists 10 s into green. A red queue that clears is not congestion.' },
  { id: 'road_obstacle', label: 'Road obstacle', light: '#6f8f1f', dark: '#7fa02a', method: 'open-vocab',
    desc: 'YOLOE-26L text prompts (plus COCO animals) once per second: on the carriageway, not explained by a track, seen at the same place for at least 4 s.' },
  { id: 'fire_smoke', label: 'Fire / smoke', light: '#a33b3b', dark: '#c24f4f', method: 'open-vocab',
    desc: 'YOLOE-26L fire and smoke prompts once per second, confidence ≥ 0.35 and seen at the same place for at least 4 s.' },
];

export const METHOD_LABEL = { rule: 'Rule', 'open-vocab': 'Open-vocab + rule', off: 'Rule, not reported', none: 'Not implemented' };

const BY_ID = Object.fromEntries(CLASSES.map((c, i) => [c.id, { ...c, order: i }]));

export function classInfo(id) {
  return BY_ID[id] || { id, label: prettyLabel(id), light: '#898781', dark: '#9a988f', method: 'unknown', desc: '', order: 999 };
}
export const classOrder = (id) => classInfo(id).order;
export const classVar = (id) => (BY_ID[id] ? `var(--cls-${id})` : 'var(--muted)');
export const classColor = (id) => { const c = classInfo(id); return isDark() ? c.dark : c.light; };
export const sortClasses = (ids) => [...ids].sort((a, b) => classOrder(a) - classOrder(b) || String(a).localeCompare(String(b)));

/** Inject --cls-<id> custom properties for both themes, so SVG and CSS can use var(--cls-...). */
export function injectPaletteCSS() {
  const light = CLASSES.map((c) => `--cls-${c.id}:${c.light};`).join('');
  const dark = CLASSES.map((c) => `--cls-${c.id}:${c.dark};`).join('');
  const css = `:root{${light}}` +
    `@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){${dark}}}` +
    `:root[data-theme="dark"]{${dark}}`;
  let el = document.getElementById('palette-css');
  if (!el) { el = document.createElement('style'); el.id = 'palette-css'; document.head.append(el); }
  el.textContent = css;
}

// Traffic-signal states (literal lamp colours; always shown with a text legend).
export const SIGNAL = {
  red: { label: 'Red', cls: 'sig-red', color: '#d03b3b' },
  yellow: { label: 'Yellow', cls: 'sig-yellow', color: '#fab219' },
  green: { label: 'Green', cls: 'sig-green', color: '#0ca30c' },
  unknown: { label: 'Unknown', cls: 'sig-unknown', color: '#bab8b0' },
};

// Generic categorical slots for non-event series (object classes, dx/dy, …), same validated order.
const SLOTS_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const SLOTS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
export const slot = (i) => (isDark() ? SLOTS_DARK : SLOTS_LIGHT)[i % 8];

// Object classes of the detector, fixed slot per class (never re-assigned by rank).
export const OBJECTS = [
  { id: 'car', label: 'Car', slot: 0 },
  { id: 'person', label: 'Person', slot: 1 },
  { id: 'bus', label: 'Bus', slot: 2 },
  { id: 'truck', label: 'Truck', slot: 3 },
  { id: 'motorcycle', label: 'Motorcycle', slot: 4 },
  { id: 'bicycle', label: 'Bicycle', slot: 5 },
];
export const KINDS = [
  { id: 'vehicle', label: 'Vehicles', slot: 0 },
  { id: 'person', label: 'Pedestrians', slot: 1 },
  { id: 'bicycle', label: 'Cyclists', slot: 2 },
  { id: 'animal', label: 'Animals', slot: 3 },
];
