/* QA for the deadline tracker: the data, the ICS export, and the rendered page.
   Run: npx tsx scripts/cal-qa.ts [url]   (default http://localhost:4321/Youngmin17/calendar/) */
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FIELDS, deadlines, instantOf, resolveDeadline, toIcs, type Venue } from '../src/components/cal/types';
import { CHECKED, VENUES } from '../src/components/cal/venues';

const URL = process.argv[2] ?? 'http://localhost:4321/Youngmin17/calendar/';
const PORT = 9355;
const PROFILE = join(tmpdir(), 'calqa-profile');
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Fail = { where: string; rule: string; detail: string };
const fails: Fail[] = [];
const bad = (where: string, rule: string, detail: string) => fails.push({ where, rule, detail });
const ISO = /^\d{4}-\d{2}-\d{2}$/;
/* Research notes get pasted in truncated. A field ending mid-clause reaches the card as one. */
const CHOPPED = /(…|\.\.\.|[;,(–-])\s*$/;
const real = (d: string) => ISO.test(d) && !Number.isNaN(Date.parse(`${d}T00:00:00Z`))
  && new Date(`${d}T00:00:00Z`).toISOString().slice(0, 10) === d;

/* ---------- 0. deadline resolution ----------
   These instants were worked out by hand from the wording each call uses and the definition of
   the zone it names — AoE is UTC-12, PST is UTC-8, EST is UTC-5 — not by running the module. A
   test that asks instantOf() to agree with instantOf() passes whatever the function does. */
const CASES: [string, string, string, string][] = [
  ['2026-09-09', 'AoE', '2026-09-10T11:59:00Z', 'a bare AoE deadline is 23:59 twelve hours behind UTC'],
  ['2026-10-08', 'AoE (11:59 pm, UTC-12)', '2026-10-09T11:59:00Z', '11:59 pm is 23:59, not 11:59'],
  ['2026-09-10', 'AoE (UTC-12h), 23:59:59', '2026-09-11T11:59:00Z', 'trailing seconds do not shift the hour'],
  ['2026-10-30', '12:00 PM PDT', '2026-10-30T19:00:00Z', 'noon PDT is 19:00 UTC — same calendar day'],
  ['2026-11-17', 'PST (5:00 PM)', '2026-11-18T01:00:00Z', '5 PM PST rolls into the next UTC day'],
  ['2026-12-08', '5:59 pm EST (10:59 pm UTC)', '2026-12-08T22:59:00Z', 'the call states its own UTC equivalent'],
  ['2027-07-16', 'AoE (5:00 PM)', '2027-07-17T05:00:00Z', 'AoE but not 23:59'],
  ['2027-04-07', 'EDT', '2027-04-08T03:59:00Z', 'EDT is UTC-4 and an unstated hour falls back to 23:59'],
  ['2027-06-10', 'not stated', '2027-06-11T11:59:00Z', 'unrecognised wording falls back to 23:59 AoE'],
];
for (const [date, tz, want, why] of CASES) {
  const got = new Date(instantOf(date, tz)).toISOString().replace('.000', '');
  if (got !== want) bad('resolve', 'wrong-instant', `${date} "${tz}" -> ${got}, want ${want} (${why})`);
}
if (resolveDeadline('2027-06-10', 'not stated').namedZone) bad('resolve', 'unknown-zone-not-flagged', 'not stated');
if (!resolveDeadline('2026-11-17', 'PST (5:00 PM)').namedZone) bad('resolve', 'known-zone-flagged-unknown', 'PST');
if (resolveDeadline('2026-09-09', 'AoE').namedTime) bad('resolve', 'assumed-time-not-flagged', 'AoE');

/* Every deadline in the dataset, pinned. Read off the call wording once, by hand, and locked here
   so that a change to the parser or to a venue's wording has to be looked at rather than absorbed. */
const GOLDEN: [string, string][] = [
  ['asplos-0-p', '2026-09-10T11:59:00Z'],
  ['cgo-0-p', '2026-09-11T11:59:00Z'],
  ['fast-0-p', '2026-09-16T11:59:00Z'],
  ['date-0-a', '2026-09-14T11:59:00Z'],
  ['date-0-p', '2026-09-21T11:59:00Z'],
  ['eurosys-0-a', '2026-09-18T11:59:00Z'],
  ['eurosys-0-p', '2026-09-25T11:59:00Z'],
  ['iclr-0-a', '2026-09-19T11:59:00Z'],
  ['iclr-0-p', '2026-09-26T11:59:00Z'],
  ['fpga-0-a', '2026-10-02T11:59:00Z'],
  ['fpga-0-p', '2026-10-09T11:59:00Z'],
  ['acl-0-p', '2026-10-13T11:59:00Z'],
  ['acl-1-p', '2027-01-05T11:59:00Z'],
  ['naacl-0-p', '2026-10-13T11:59:00Z'],
  ['mlsys-0-p', '2026-10-30T19:00:00Z'],
  ['isca-0-a', '2026-11-10T11:59:00Z'],
  ['isca-0-p', '2026-11-17T11:59:00Z'],
  ['isca-1-a', '2026-12-05T11:59:00Z'],
  ['isca-1-p', '2026-12-12T11:59:00Z'],
  ['dac-0-a', '2026-11-11T01:00:00Z'],
  ['dac-0-p', '2026-11-18T01:00:00Z'],
  ['osdi-0-a', '2026-12-01T22:59:00Z'],
  ['osdi-0-p', '2026-12-08T22:59:00Z'],
  ['icml-0-a', '2027-01-23T11:59:00Z'],
  ['icml-0-p', '2027-01-29T11:59:00Z'],
  ['sosp-0-a', '2027-03-26T11:59:00Z'],
  ['sosp-0-p', '2027-04-02T11:59:00Z'],
  ['micro-0-a', '2027-04-01T03:59:00Z'],
  ['micro-0-p', '2027-04-08T03:59:00Z'],
  ['iccad-0-a', '2027-04-07T11:59:00Z'],
  ['iccad-0-p', '2027-04-14T11:59:00Z'],
  ['sc-0-a', '2027-04-08T11:59:00Z'],
  ['sc-0-p', '2027-04-15T11:59:00Z'],
  ['pact-0-a', '2027-04-17T11:59:00Z'],
  ['pact-0-p', '2027-04-24T11:59:00Z'],
  ['neurips-0-a', '2027-05-04T11:59:00Z'],
  ['neurips-0-p', '2027-05-08T11:59:00Z'],
  ['emnlp-0-p', '2027-05-25T11:59:00Z'],
  ['atc-0-p', '2027-06-11T11:59:00Z'],
  ['aspdac-0-a', '2027-07-10T05:00:00Z'],
  ['aspdac-0-p', '2027-07-17T05:00:00Z'],
  ['aaai-0-a', '2027-07-21T11:59:00Z'],
  ['aaai-0-p', '2027-07-28T11:59:00Z'],
  ['hpca-0-a', '2027-07-24T11:59:00Z'],
  ['hpca-0-p', '2027-07-31T11:59:00Z'],
  ['ppopp-0-p', '2027-08-03T11:59:00Z'],
];
{
  const got = new Map(deadlines(VENUES as Venue[]).map((d) => [d.key, new Date(d.at).toISOString().replace('.000', '')]));
  const want = new Map(GOLDEN);
  for (const [key, iso] of want) {
    if (!got.has(key)) bad('resolve', 'golden-deadline-missing', key);
    else if (got.get(key) !== iso) bad('resolve', 'golden-instant-changed', `${key}: ${got.get(key)}, was ${iso}`);
  }
  for (const key of got.keys()) if (!want.has(key)) bad('resolve', 'golden-deadline-unlisted', key);
}

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
  if (v.site && !/^https?:\/\/\S+$/.test(v.site)) bad(at, 'site-not-url', v.site);
  for (const [k, txt] of Object.entries({ name: v.name, fullName: v.fullName, edition: v.edition,
    location: v.location, basis: v.basis, notes: v.notes })) {
    if (typeof txt === 'string' && CHOPPED.test(txt.trim())) bad(at, 'text-truncated', `${k}: …${txt.trim().slice(-44)}`);
  }
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
    if (r.timezone && CHOPPED.test(r.timezone.trim())) bad(at, 'text-truncated', `timezone: ${r.timezone}`);
    /* A call whose zone we cannot read is silently rounded to 23:59 AoE, so it has to say so. */
    const zr = resolveDeadline(r.paper, r.timezone);
    if (!zr.namedZone && !/not stated|unknown|no zone/i.test(r.timezone ?? '')) {
      bad(at, 'timezone-unreadable', `"${r.timezone}" names no zone this page knows`);
    }
    if (r.abstract && instantOf(r.abstract, r.timezone) > instantOf(r.paper, r.timezone)) {
      bad(at, 'abstract-instant-after-paper', `${r.abstract} vs ${r.paper}`);
    }
    if (prev && r.paper < prev) bad(at, 'rounds-out-of-order', `${prev} then ${r.paper}`);
    prev = r.paper;
  }
}

const all = deadlines(VENUES as Venue[]);
for (let i = 1; i < all.length; i++) if (all[i].at < all[i - 1].at) bad('data', 'sort-broken', `${all[i - 1].key} then ${all[i].key}`);
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
for (const l of starts) if (!/^DTSTART:\d{8}T\d{6}Z$/.test(l)) bad('ics', 'dtstart-format', l);
for (const l of lines.filter((x) => x.startsWith('DTEND'))) if (!/^DTEND:\d{8}T\d{6}Z$/.test(l)) bad('ics', 'dtend-format', l);
/* The block has to end ON the deadline, with the alarms measured back from that end — an all-day
   event would fire a Tokyo reader's one-day warning most of a day early on a 23:59 AoE call. */
{
  const stamp = (ms: number) => new Date(ms).toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
  const ends = lines.filter((l) => l.startsWith('DTEND:'));
  for (const [i, d] of all.entries()) {
    if (ends[i] !== `DTEND:${stamp(d.at)}`) bad('ics', 'dtend-not-deadline', `${d.key}: ${ends[i]}`);
    if (starts[i] !== `DTSTART:${stamp(d.at - 3_600_000)}`) bad('ics', 'dtstart-not-hour-before', `${d.key}: ${starts[i]}`);
  }
  const rel = lines.filter((l) => l.startsWith('TRIGGER'));
  for (const l of rel) if (!/^TRIGGER;RELATED=END:-P[17]D$/.test(l)) bad('ics', 'alarm-not-from-end', l);
  if (rel.length !== all.length * 2) bad('ics', 'alarm-count', `${rel.length}`);
}
if (new Set(lines.filter((l) => l.startsWith('UID:'))).size !== all.length) bad('ics', 'uid-not-unique', '');
const unfolded = ics.replace(/\r\n /g, '');
for (const d of all) if (!unfolded.includes(d.venue.name)) bad('ics', 'summary-missing-venue', d.venue.name);

/* ---------- 3. rendered page ---------- */
/* Cards are ordered by the deadline each is counting down to, not by the paper deadline: FAST's
   paper lands before DATE's, but DATE's abstract lands before FAST's paper. Both clocks read "now"
   a second or two apart, which only matters for a deadline that falls inside that window. */
const QA_NOW = Date.now();
const rounds = (VENUES as Venue[]).flatMap((v) => v.rounds.map((r, i) => {
  const due = instantOf(r.paper, r.timezone);
  const abs = r.abstract ? instantOf(r.abstract, r.timezone) : undefined;
  return { v, r, i, due, target: abs != null && abs >= QA_NOW ? abs : due };
}));
rounds.sort((a, b) => a.target - b.target || a.due - b.due);

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
  btns(){ return [...document.querySelectorAll('.ddl-btn')] },
  press(text){ const b = this.btns().find(x => x.textContent.trim() === text); if(!b) throw new Error('no button ' + text); b.click(); },
  card(el){ return {
    tag: el.querySelector('.ddl-tag') ? el.querySelector('.ddl-tag').textContent.trim() : null,
    est: !!el.querySelector('.ddl-inferred'),
    name: el.querySelector('.ddl-conf') ? el.querySelector('.ddl-conf').textContent.trim() : null,
    year: el.querySelector('.ddl-year') ? el.querySelector('.ddl-year').textContent.trim() : null,
    cycle: el.querySelector('.ddl-cycle') ? el.querySelector('.ddl-cycle').textContent.trim() : null,
    full: el.querySelector('.ddl-full') ? el.querySelector('.ddl-full').textContent.trim() : null,
    sub: el.querySelector('.ddl-primary time') ? el.querySelector('.ddl-primary time').textContent.replace(/\\s+/g,' ').trim() : null,
    basis: !!el.querySelector('.ddl-basis'),
    days: el.querySelector('.ddl-num') ? el.querySelector('.ddl-num').textContent.trim() : null,
    pct: el.querySelector('.ddl-progress span') ? el.querySelector('.ddl-progress span').style.width : null,
  } },
  scrape(){ return {
    upcoming: [...document.querySelectorAll('.ddl-upcoming .ddl-card')].map(c => this.card(c)),
    past: [...document.querySelectorAll('.ddl-past .ddl-card')].map(c => this.card(c)),
    counts: [...document.querySelectorAll('.ddl-count')].map(c => c.textContent.trim()),
    pressed: this.btns().filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.textContent.trim()),
    exportLabel: document.querySelector('.ddl-export') ? document.querySelector('.ddl-export').textContent.trim() : null,
    sw: document.documentElement.scrollWidth, iw: window.innerWidth,
  } }
}; 'ok'`);

const base = await evalIn(`(async()=>{ await __c.f(); return __c.scrape() })()`);
const total = base.upcoming.length + base.past.length;
if (total !== rounds.length) bad('ui', 'card-count', `${total} cards for ${rounds.length} rounds`);
if (base.past.length !== 0) bad('ui', 'unexpected-past', `${base.past.length} past cards, data has none before now`);
if (!base.exportLabel || !base.exportLabel.includes(String(all.length))) bad('ui', 'export-count', String(base.exportLabel));
if (base.counts[0] !== String(base.upcoming.length)) bad('ui', 'section-count', `${base.counts[0]} vs ${base.upcoming.length}`);

for (const [i, c] of base.upcoming.entries()) {
  const want = rounds[i];
  const at = `${want.v.id}#${i}`;
  if (c.name !== want.v.name) bad(at, 'card-order', `dom=${c.name} want=${want.v.name}`);
  if (!c.tag) bad(at, 'card-no-tag', '');
  if (!c.full) bad(at, 'card-no-fullname', '');
  if (!c.sub) bad(at, 'card-no-submission', '');
  if (!c.days) bad(at, 'card-no-countdown', '');
  if (c.est !== (want.v.status === 'projected')) bad(at, 'card-est-badge', `dom=${c.est} status=${want.v.status}`);
  if (want.v.status === 'projected' && !c.basis) bad(at, 'card-no-basis', '');
  if ((c.cycle ?? '') !== (want.r.label ?? '')) bad(at, 'card-cycle', `dom=${c.cycle} want=${want.r.label}`);
  if (!c.year) bad(at, 'card-no-year', '');
  const pct = parseFloat(c.pct ?? '0');
  if (!(pct >= 0 && pct <= 100)) bad(at, 'progress-range', String(c.pct));
}
for (const junk of ['NaN', 'undefined', 'Invalid Date', '[object']) if (JSON.stringify(base).includes(junk)) bad('ui', 'junk-value', junk);
if (base.sw > base.iw + 1) bad('ui', 'overflow-x', `${base.sw} > ${base.iw}`);

/* category filtering */
for (const f of FIELDS) {
  const n = rounds.filter((x) => x.v.field === f.id).length;
  const got = await evalIn(`(async()=>{ __c.press('All'); await __c.f(); __c.press(${JSON.stringify(f.label)}); await __c.f(); return __c.scrape() })()`);
  const shownN = got.upcoming.length + got.past.length;
  if (shownN !== n) bad(`ui:${f.id}`, 'filter-count', `${shownN} cards, expected ${n}`);
  if (!got.pressed.includes(f.label)) bad(`ui:${f.id}`, 'filter-state', got.pressed.join('|'));
  if (got.upcoming.some((c: any) => c.tag !== f.label.toUpperCase() && c.tag !== f.label)) bad(`ui:${f.id}`, 'filter-leak', got.upcoming.map((c: any) => c.tag).join('|'));
}
const restored = await evalIn(`(async()=>{ __c.press('All'); await __c.f(); return __c.scrape() })()`);
if (restored.upcoming.length + restored.past.length !== rounds.length) bad('ui', 'filter-reset', String(restored.upcoming.length));

/* the timezone selector must actually move the clock: AoE is UTC-12 */
const aoe = await evalIn(`(async()=>{ __c.press('AoE'); await __c.f(); return __c.scrape() })()`);
const utc = await evalIn(`(async()=>{ __c.press('UTC'); await __c.f(); return __c.scrape() })()`);
const loc = await evalIn(`(async()=>{ __c.press('Local'); await __c.f(); return __c.scrape() })()`);
if (aoe.upcoming[0].sub === utc.upcoming[0].sub) bad('ui', 'timezone-inert', `aoe=${aoe.upcoming[0].sub} utc=${utc.upcoming[0].sub}`);
if (!aoe.upcoming[0].sub.includes('AoE')) bad('ui', 'timezone-label', aoe.upcoming[0].sub);
if (!utc.upcoming[0].sub.includes('UTC')) bad('ui', 'timezone-label', utc.upcoming[0].sub);
if (!loc.upcoming[0].sub.includes('Local')) bad('ui', 'timezone-label', loc.upcoming[0].sub);
{
  // Expectations built from the GOLDEN instant above — hand-checked against each call's wording —
  // rather than from instantOf(), which would shift with any bug in it and still pass.
  const MO = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const p2 = (n: number) => String(n).padStart(2, '0');
  const at = new Map(GOLDEN).get(`${rounds[0].v.id}-${rounds[0].i}-p`);
  if (!at) bad('ui', 'golden-missing-for-first-card', `${rounds[0].v.id}`);
  else {
    const u = new Date(at);                                  // the deadline as a real instant
    const a = new Date(u.getTime() - 12 * 60 * 60_000);       // the same instant read in AoE (UTC-12)
    const show = (d: Date, z: string) =>
      `${MO[d.getUTCMonth()]} ${p2(d.getUTCDate())}, ${d.getUTCFullYear()} ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())} ${z}`;
    if (aoe.upcoming[0].sub !== show(a, 'AoE')) bad('ui', 'aoe-text', `dom="${aoe.upcoming[0].sub}" want="${show(a, 'AoE')}"`);
    if (utc.upcoming[0].sub !== show(u, 'UTC')) bad('ui', 'utc-text', `dom="${utc.upcoming[0].sub}" want="${show(u, 'UTC')}"`);
  }
  // the six calls that are not a plain 23:59 AoE deadline must say so on the card rather than be
  // quietly restated; the rest must not carry the note
  const odd = new Set(rounds.filter((x) => {
    const R = resolveDeadline(x.r.paper, x.r.timezone);
    return !(R.namedZone && R.offsetMin === -720 && R.h === 23 && R.m === 59);
  }).map((x) => x.v.name));
  const noted: string[] = await evalIn(`[...document.querySelectorAll('.ddl-upcoming .ddl-card')]
    .filter(c => c.querySelector('.ddl-asis')).map(c => c.querySelector('.ddl-conf').textContent.trim())`);
  for (const n of odd) if (!noted.includes(n)) bad('ui', 'wording-note-missing', n);
  for (const n of noted) if (!odd.has(n)) bad('ui', 'wording-note-spurious', n);
}
await evalIn(`(async()=>{ __c.press('AoE'); await __c.f() })()`);

for (const w of [390, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 600 });
  await wait(220);
  const r = await evalIn(`(async()=>{ await __c.f(); const d = __c.scrape();
    return { over: d.sw - d.iw, off: [...document.querySelectorAll('.ddl *')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 1).map(e => e.className).slice(0,5) } })()`);
  if (r.over > 1) bad(`viewport ${w}`, 'overflow-x', `${r.over}px ${r.off.join('|')}`);
}
const a11y = await evalIn(`(() => ({
  noAria: [...document.querySelectorAll('.ddl-btn')].filter(c => !c.hasAttribute('aria-pressed')).length,
  notButtons: [...document.querySelectorAll('.ddl-btn, .ddl-export')].filter(c => c.tagName !== 'BUTTON').length,
  headings: document.querySelectorAll('.ddl-section-title').length,
  timeNoDatetime: [...document.querySelectorAll('.ddl-card time')].filter(t => !t.getAttribute('datetime')).length,
  extLinksUnsafe: [...document.querySelectorAll('.ddl-name a')].filter(a => a.target === '_blank' && !a.rel.includes('noopener')).length,
}))()`);
if (a11y.noAria) bad('a11y', 'btn-aria-pressed', String(a11y.noAria));
if (a11y.notButtons) bad('a11y', 'interactive-not-button', String(a11y.notButtons));
if (!a11y.headings) bad('a11y', 'no-section-heading', '');
if (a11y.timeNoDatetime) bad('a11y', 'time-without-datetime', String(a11y.timeNoDatetime));
if (a11y.extLinksUnsafe) bad('a11y', 'link-without-noopener', String(a11y.extLinksUnsafe));
for (const e of consoleErrors) bad('console', 'console-error', e.slice(0, 200));

/* ---------- report ---------- */
console.log(`venues        : ${VENUES.length} (confirmed ${VENUES.filter((v) => v.status === 'confirmed').length}, projected ${VENUES.filter((v) => v.status === 'projected').length})`);
console.log(`rounds / cards: ${rounds.length}`);
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
