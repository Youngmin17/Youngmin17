/* QA for the deadline tracker: the data, the ICS export, and the rendered page.
   Run: npx tsx scripts/cal-qa.ts [url]   (default http://localhost:4321/Youngmin17/calendar/) */
import { spawn } from 'node:child_process';
import { FIELDS, deadlines, toIcs, type Venue } from '../src/components/cal/types';
import { CHECKED, VENUES } from '../src/components/cal/venues';

const URL = process.argv[2] ?? 'http://localhost:4321/Youngmin17/calendar/';
const PORT = 9355;
const PROFILE = '/private/tmp/claude-501/-Users-mincho-youngmin-git/8b6a1532-e09a-4d58-9241-bba1567f47e4/scratchpad/calqa-profile';
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Fail = { where: string; rule: string; detail: string };
const fails: Fail[] = [];
const bad = (where: string, rule: string, detail: string) => fails.push({ where, rule, detail });
const ISO = /^\d{4}-\d{2}-\d{2}$/;
const real = (d: string) => ISO.test(d) && !Number.isNaN(Date.parse(`${d}T00:00:00Z`))
  && new Date(`${d}T00:00:00Z`).toISOString().slice(0, 10) === d;

/* ---------- 1. data ---------- */
const EXPECTED = ['isca', 'hpca', 'asplos', 'mlsys', 'neurips', 'dac', 'iccad', 'osdi', 'icml', 'iclr',
  'micro', 'fpga', 'sc', 'atc', 'sosp', 'eurosys', 'fast', 'ppopp', 'cgo', 'pact', 'date', 'aspdac',
  'aaai', 'emnlp', 'acl', 'naacl'];

const ids = VENUES.map((v) => v.id);
for (const want of EXPECTED) if (!ids.includes(want)) bad('data', 'venue-missing', want);
for (const id of ids) if (!EXPECTED.includes(id)) bad('data', 'venue-unexpected', id);
if (new Set(ids).size !== ids.length) bad('data', 'venue-duplicate', ids.join(','));
if (!real(CHECKED)) bad('data', 'checked-date', CHECKED);

for (const v of VENUES as Venue[]) {
  const at = `${v.id}`;
  if (!v.name || !v.fullName || !v.edition) bad(at, 'field-empty', JSON.stringify({ n: v.name, f: v.fullName, e: v.edition }));
  if (!FIELDS.some((f) => f.id === v.field)) bad(at, 'field-unknown', v.field);
  if (!['confirmed', 'projected'].includes(v.status)) bad(at, 'status-unknown', v.status);
  if (!v.rounds.length) bad(at, 'no-rounds', '');
  if (v.status === 'projected' && !v.basis) bad(at, 'projected-without-basis', '');
  if (v.status === 'confirmed' && !v.source) bad(at, 'confirmed-without-source', '');
  if (v.site && !/^https?:\/\//.test(v.site)) bad(at, 'site-not-url', v.site);
  if (v.source && !/^https?:\/\//.test(v.source)) bad(at, 'source-not-url', v.source);
  for (const d of [v.confStart, v.confEnd]) if (d && !real(d)) bad(at, 'conf-date-invalid', String(d));
  if (v.confStart && v.confEnd && v.confEnd < v.confStart) bad(at, 'conf-end-before-start', `${v.confStart}..${v.confEnd}`);

  const labels = v.rounds.map((r) => r.label);
  if (v.rounds.length > 1 && new Set(labels).size !== labels.length) bad(at, 'round-labels-duplicate', labels.join('|'));
  let prev = '';
  for (const r of v.rounds) {
    if (!real(r.paper)) { bad(at, 'paper-date-invalid', r.paper); continue; }
    if (r.paper < CHECKED) bad(at, 'deadline-in-past', `${r.paper} < ${CHECKED}`);
    if (r.abstract) {
      if (!real(r.abstract)) bad(at, 'abstract-date-invalid', r.abstract);
      else if (r.abstract > r.paper) bad(at, 'abstract-after-paper', `${r.abstract} > ${r.paper}`);
    }
    if (r.notification) {
      if (!real(r.notification)) bad(at, 'notification-date-invalid', r.notification);
      else if (r.notification < r.paper) bad(at, 'notification-before-paper', `${r.notification} < ${r.paper}`);
      else if (v.confStart && r.notification > v.confStart) bad(at, 'notification-after-conference', `${r.notification} > ${v.confStart}`);
    }
    if (v.confStart && r.paper > v.confStart) bad(at, 'paper-after-conference', `${r.paper} > ${v.confStart}`);
    if (prev && r.paper < prev) bad(at, 'rounds-out-of-order', `${prev} then ${r.paper}`);
    prev = r.paper;
  }
}

const all = deadlines(VENUES as Venue[]);
for (let i = 1; i < all.length; i++) if (all[i].date < all[i - 1].date) bad('data', 'sort-broken', `${all[i - 1].date} then ${all[i].date}`);
if (new Set(all.map((d) => d.key)).size !== all.length) bad('data', 'deadline-key-duplicate', '');

/* ---------- 2. ICS ---------- */
const ics = toIcs(all, '20260822T000000Z');
const lines = ics.split('\r\n').filter(Boolean);
if (!ics.includes('\r\n')) bad('ics', 'not-crlf', '');
if (lines[0] !== 'BEGIN:VCALENDAR') bad('ics', 'no-begin', lines[0]);
if (lines[lines.length - 1] !== 'END:VCALENDAR') bad('ics', 'no-end', lines[lines.length - 1]);
const nBegin = lines.filter((l) => l === 'BEGIN:VEVENT').length;
const nEnd = lines.filter((l) => l === 'END:VEVENT').length;
if (nBegin !== all.length) bad('ics', 'event-count', `${nBegin} events for ${all.length} deadlines`);
if (nBegin !== nEnd) bad('ics', 'vevent-unbalanced', `${nBegin}/${nEnd}`);
const nAlarmB = lines.filter((l) => l === 'BEGIN:VALARM').length;
const nAlarmE = lines.filter((l) => l === 'END:VALARM').length;
if (nAlarmB !== all.length * 2 || nAlarmB !== nAlarmE) bad('ics', 'valarm-count', `${nAlarmB}/${nAlarmE}`);
for (const l of lines) {
  if (Buffer.byteLength(l, 'utf8') > 75) bad('ics', 'line-too-long', `${Buffer.byteLength(l, 'utf8')}B: ${l.slice(0, 40)}`);
  if (/[\r\n]/.test(l)) bad('ics', 'raw-newline', l.slice(0, 40));
}
const starts = lines.filter((l) => l.startsWith('DTSTART'));
if (starts.length !== all.length) bad('ics', 'dtstart-count', String(starts.length));
for (const l of starts) if (!/^DTSTART;VALUE=DATE:\d{8}$/.test(l)) bad('ics', 'dtstart-format', l);
for (const l of lines.filter((x) => x.startsWith('DTEND'))) if (!/^DTEND;VALUE=DATE:\d{8}$/.test(l)) bad('ics', 'dtend-format', l);
if (new Set(lines.filter((l) => l.startsWith('UID:'))).size !== all.length) bad('ics', 'uid-not-unique', '');
const unfolded = ics.replace(/\r\n /g, '');
for (const d of all) if (!unfolded.includes(d.venue.name)) bad('ics', 'summary-missing-venue', d.venue.name);

/* ---------- 3. rendered page ---------- */
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--disable-gpu', '--hide-scrollbars', `--remote-debugging-port=${PORT}`,
   `--user-data-dir=${PROFILE}`, 'about:blank'], { stdio: 'ignore' });
let list: any[] = [];
for (let i = 0; i < 60; i++) {
  try { list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (list.length) break; } catch {}
  await wait(250);
}
const ws = new WebSocket(list.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let msgId = 0;
const pending = new Map<number, (v: any) => void>();
const consoleErrors: string[] = [];
ws.addEventListener('message', (e: any) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)!(m.result); pending.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') consoleErrors.push(m.params?.exceptionDetails?.exception?.description ?? '?');
  if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params?.type))
    consoleErrors.push(`${m.params.type}: ${(m.params.args ?? []).map((a: any) => a.value ?? a.description).join(' ')}`);
});
const send = (method: string, params: any = {}) =>
  new Promise<any>((res) => { const i = ++msgId; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const evalIn = async (expression: string) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? 'eval failed');
  return r.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: URL });
await wait(2600);
await evalIn(`window.__c = {
  f(){ return new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))) },
  chips(){ return [...document.querySelectorAll('.dl-f')] },
  scrape(){ return {
    pins: [...document.querySelectorAll('.dl-pin')].map(p => p.querySelector('b').textContent.trim()),
    rows: [...document.querySelectorAll('.dl-table tbody tr')].map(r => ({
      venue: r.querySelector('.v').textContent.replace('est','').trim(),
      what: r.children[1].textContent.trim(),
      date: r.children[2].textContent.trim(),
      cd: r.children[3].textContent.trim() })),
    months: document.querySelectorAll('.dl-mo').length,
    next: [...document.querySelectorAll('.dl-up')].map(u => u.querySelector('.who').textContent.trim()),
    pressed: [...document.querySelectorAll('.dl-f[aria-pressed="true"]')].map(b => b.textContent.trim()),
    ics: document.querySelector('.dl-ics') ? document.querySelector('.dl-ics').textContent.trim() : null,
    sw: document.documentElement.scrollWidth, iw: window.innerWidth,
  } }
}; 'ok'`);

const base = await evalIn(`(async()=>{ await __c.f(); return __c.scrape() })()`);
if (base.pins.length !== all.length) bad('ui', 'pin-count', `${base.pins.length} pins for ${all.length} deadlines`);
if (base.rows.length !== all.length) bad('ui', 'row-count', `${base.rows.length} rows for ${all.length} deadlines`);
if (!base.ics || !base.ics.includes(String(all.length))) bad('ui', 'ics-button-count', String(base.ics));
const wantNext = all.slice(0, 3).map((d) => `${d.venue.name}${d.round.label ? ` · ${d.round.label}` : ''}`);
if (JSON.stringify(base.next) !== JSON.stringify(wantNext)) bad('ui', 'next-three', `dom=${JSON.stringify(base.next)} want=${JSON.stringify(wantNext)}`);
for (const junk of ['NaN', 'undefined', 'Invalid Date', '[object']) if (JSON.stringify(base).includes(junk)) bad('ui', 'junk-value', junk);
if (base.sw > base.iw + 1) bad('ui', 'overflow-x', `${base.sw} > ${base.iw}`);
const rowDates = base.rows.map((r: any) => r.date);
for (let i = 1; i < rowDates.length; i++) {
  const a = Date.parse(rowDates[i - 1].slice(0, 12)), b = Date.parse(rowDates[i].slice(0, 12));
  if (!Number.isNaN(a) && !Number.isNaN(b) && b < a) bad('ui', 'table-unsorted', `${rowDates[i - 1]} then ${rowDates[i]}`);
}

for (const f of FIELDS) {
  const n = all.filter((d) => d.venue.field === f.id).length;
  const got = await evalIn(`(async()=>{
    const chips = __c.chips();
    chips.filter(c => c.getAttribute('aria-pressed') !== 'true').forEach(c => c.click());
    await __c.f();
    chips.find(c => c.textContent.trim() === ${JSON.stringify(f.label)}).click();
    await __c.f();
    return __c.scrape();
  })()`);
  if (got.rows.length !== n) bad(`ui:${f.id}`, 'filter-row-count', `${got.rows.length} rows, expected ${n}`);
  if (got.pins.length !== n) bad(`ui:${f.id}`, 'filter-pin-count', `${got.pins.length} pins, expected ${n}`);
  if (got.months !== base.months) bad(`ui:${f.id}`, 'axis-jumped', `${got.months} vs ${base.months}`);
  if (got.pressed.length !== 1 || got.pressed[0] !== f.label) bad(`ui:${f.id}`, 'filter-state', got.pressed.join('|'));
}

for (const w of [390, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 600 });
  await wait(220);
  const r = await evalIn(`(async()=>{ await __c.f(); const d = __c.scrape();
    return { over: d.sw - d.iw, off: [...document.querySelectorAll('.dl *')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 1).map(e => e.className).slice(0,5) } })()`);
  if (r.over > 1) bad(`viewport ${w}`, 'overflow-x', `${r.over}px ${r.off.join('|')}`);
}
const a11y = await evalIn(`(() => ({
  chipsNoAria: [...document.querySelectorAll('.dl-f')].filter(c => !c.hasAttribute('aria-pressed')).length,
  notButtons: [...document.querySelectorAll('.dl-f, .dl-pin, .dl-ics')].filter(c => c.tagName !== 'BUTTON').length,
  th: document.querySelectorAll('.dl-table thead th').length,
}))()`);
if (a11y.chipsNoAria) bad('a11y', 'chip-aria-pressed', String(a11y.chipsNoAria));
if (a11y.notButtons) bad('a11y', 'interactive-not-button', String(a11y.notButtons));
if (a11y.th < 4) bad('a11y', 'table-headers', String(a11y.th));
for (const e of consoleErrors) bad('console', 'console-error', e.slice(0, 200));

/* ---------- report ---------- */
console.log(`venues        : ${VENUES.length} (confirmed ${VENUES.filter((v) => v.status === 'confirmed').length}, projected ${VENUES.filter((v) => v.status === 'projected').length})`);
console.log(`deadlines     : ${all.length}`);
console.log(`ics           : ${lines.length} lines, ${nBegin} events, ${nAlarmB} alarms`);
console.log(`console errors: ${consoleErrors.length}`);
console.log(`failures      : ${fails.length}`);
const byRule = new Map<string, Fail[]>();
for (const f of fails) { if (!byRule.has(f.rule)) byRule.set(f.rule, []); byRule.get(f.rule)!.push(f); }
for (const [rule, l] of [...byRule].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`\n  x ${rule}  (${l.length})`);
  for (const f of l.slice(0, 6)) console.log(`      ${f.where}: ${f.detail}`);
  if (l.length > 6) console.log(`      ... ${l.length - 6} more`);
}
if (!fails.length) console.log('\nData, ICS and rendered page all agree.');
ws.close(); chrome.kill();
process.exit(fails.length ? 1 : 0);
