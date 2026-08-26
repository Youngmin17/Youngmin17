export type Field = 'architecture' | 'systems' | 'eda' | 'ml' | 'nlp' | 'hpc';

export interface Round {
  label: string;        // "" for a single-round venue, else "Round 1", "Cycle 2", …
  abstract?: string;    // YYYY-MM-DD
  paper: string;        // YYYY-MM-DD
  notification?: string;
  timezone?: string;    // usually AoE
}

export interface Venue {
  id: string;
  name: string;
  fullName: string;
  field: Field;
  edition: string;
  site?: string;
  location?: string;
  confStart?: string;
  confEnd?: string;
  status: 'confirmed' | 'projected';
  rounds: Round[];
  source?: string;
  basis?: string;       // for projected: which past editions the estimate comes from
  notes?: string;
}

export const FIELDS: { id: Field; label: string; color: string }[] = [
  { id: 'architecture', label: 'Architecture', color: '#2563eb' },
  { id: 'systems', label: 'Systems', color: '#0891b2' },
  { id: 'ml', label: 'Machine learning', color: '#c2410c' },
  { id: 'nlp', label: 'Language', color: '#7c3aed' },
  { id: 'eda', label: 'EDA', color: '#059669' },
  { id: 'hpc', label: 'HPC & FPGA', color: '#b45309' },
];

export const fieldOf = (f: Field) => FIELDS.find((x) => x.id === f)!;

/* A call for papers states its deadline in whatever zone it likes. Most say 23:59
   Anywhere-on-Earth, but DAC says 5:00 PM PST, MLSys says 12:00 PM PDT, OSDI says 5:59 pm EST,
   and ASP-DAC says 5:00 PM AoE — same day, seven hours apart. Resolving the wording to a real
   instant is what lets the page restate every one of them in a single zone and count down in
   hours rather than in whole days. */
export const AOE_OFFSET_MIN = -12 * 60;

/* Fixed offsets only. These are the zones the calls actually name, and a submission deadline is
   a fixed instant, so a named offset like EDT is unambiguous where "US Eastern" would not be. */
const ZONES: { re: RegExp; min: number }[] = [
  { re: /\bAoE\b/i, min: -12 * 60 },
  { re: /\bCEST\b/, min: 2 * 60 },
  { re: /\bCET\b/, min: 60 },
  { re: /\b(KST|JST)\b/, min: 9 * 60 },
  { re: /\bEST\b/, min: -5 * 60 },
  { re: /\bEDT\b/, min: -4 * 60 },
  { re: /\bCST\b/, min: -6 * 60 },
  { re: /\bCDT\b/, min: -5 * 60 },
  { re: /\bMST\b/, min: -7 * 60 },
  { re: /\bMDT\b/, min: -6 * 60 },
  { re: /\bPST\b/, min: -8 * 60 },
  { re: /\bPDT\b/, min: -7 * 60 },
  { re: /\b(UTC|GMT)\b/i, min: 0 },
];

export interface Resolved {
  at: number;          // the deadline as a real instant
  offsetMin: number;   // the zone the call quoted, in minutes from UTC
  h: number; m: number;// the clock time the call quoted
  namedZone: boolean;  // false when the wording names no zone we recognise
  namedTime: boolean;  // false when the wording gives no clock time and 23:59 was assumed
}

/* "AoE (11:59 pm, UTC-12)" must read as 23:59, not 11:59, so the 12-hour form is tried first. */
function clockOf(tz: string): { h: number; m: number; named: boolean } {
  const ampm = tz.match(/\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\.?/i);
  if (ampm) {
    const h = Number(ampm[1]) % 12;
    return { h: /p/i.test(ampm[3]) ? h + 12 : h, m: ampm[2] ? Number(ampm[2]) : 0, named: true };
  }
  const h24 = tz.match(/\b(\d{1,2}):(\d{2})\b/);
  if (h24 && Number(h24[1]) < 24) return { h: Number(h24[1]), m: Number(h24[2]), named: true };
  return { h: 23, m: 59, named: false };
}

/* An explicit "UTC-12" beats a bare zone name; otherwise the leftmost zone in the sentence wins,
   which is what picks EST out of "5:59 pm EST (10:59 pm UTC)". */
function zoneOf(tz: string): { min: number; named: boolean } {
  const signed = tz.match(/\bUTC\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?/i);
  const explicit = signed ? { i: signed.index!, min: (signed[1] === '-' ? -1 : 1) * (Number(signed[2]) * 60 + Number(signed[3] ?? 0)) } : null;
  let best: { i: number; min: number } | null = explicit;
  for (const z of ZONES) {
    const m = tz.match(z.re);
    if (m && (best == null || m.index! < best.i)) best = { i: m.index!, min: z.min };
  }
  return best ? { min: best.min, named: true } : { min: AOE_OFFSET_MIN, named: false };
}

export function resolveDeadline(dateISO: string, tz = 'AoE'): Resolved {
  const zone = zoneOf(tz ?? '');
  const clock = clockOf(tz ?? '');
  const midnight = Date.parse(`${dateISO}T00:00:00Z`);
  return {
    at: midnight + (clock.h * 60 + clock.m - zone.min) * 60_000,
    offsetMin: zone.min, h: clock.h, m: clock.m,
    namedZone: zone.named, namedTime: clock.named,
  };
}

export const instantOf = (dateISO: string, tz = 'AoE'): number => resolveDeadline(dateISO, tz).at;

/* True for the ordinary case the footer describes; anything else gets its wording shown verbatim
   on the card rather than silently restated as 23:59. */
export const isPlainAoe = (r: Resolved) =>
  r.namedZone && r.offsetMin === AOE_OFFSET_MIN && r.h === 23 && r.m === 59;

export type Zone = 'aoe' | 'local' | 'utc';

export function formatIn(ms: number, zone: Zone): string {
  const d = new Date(zone === 'aoe' ? ms + AOE_OFFSET_MIN * 60_000 : ms);
  const get = zone === 'local'
    ? { mo: d.getMonth(), da: d.getDate(), y: d.getFullYear(), h: d.getHours(), mi: d.getMinutes() }
    : { mo: d.getUTCMonth(), da: d.getUTCDate(), y: d.getUTCFullYear(), h: d.getUTCHours(), mi: d.getUTCMinutes() };
  const MO = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const p = (n: number) => String(n).padStart(2, '0');
  return `${MO[get.mo]} ${p(get.da)}, ${get.y} ${p(get.h)}:${p(get.mi)}`;
}

export const ZONE_LABEL: Record<Zone, string> = { aoe: 'AoE', local: 'Local', utc: 'UTC' };

/* One deadline, flattened out of the venue/round nesting for sorting and display. */
export interface Deadline {
  venue: Venue;
  round: Round;
  kind: 'abstract' | 'paper';
  date: string;
  at: number;
  key: string;
}

export function deadlines(venues: Venue[]): Deadline[] {
  const out: Deadline[] = [];
  for (const v of venues) {
    for (const [i, r] of v.rounds.entries()) {
      if (r.abstract) out.push({ venue: v, round: r, kind: 'abstract', date: r.abstract, at: instantOf(r.abstract, r.timezone), key: `${v.id}-${i}-a` });
      if (r.paper) out.push({ venue: v, round: r, kind: 'paper', date: r.paper, at: instantOf(r.paper, r.timezone), key: `${v.id}-${i}-p` });
    }
  }
  return out.sort((a, b) => a.at - b.at || a.key.localeCompare(b.key));
}

/* RFC 5545. A deadline is an instant, not a day: an all-day event on the date would land a
   Tokyo reader most of a day early on a 23:59 AoE call, and its alarms with it. The event is the
   last hour before the deadline and the alarms hang off its end, so -P1D means twenty-four hours
   before the deadline itself wherever the reader happens to be. */
export function toIcs(list: Deadline[], stamp: string): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const esc = (s: string) => s.replace(/([,;\\])/g, '\\$1').replace(/\n/g, '\\n');
  const utc = (ms: number) => {
    const t = new Date(ms);
    return `${t.getUTCFullYear()}${pad(t.getUTCMonth() + 1)}${pad(t.getUTCDate())}`
      + `T${pad(t.getUTCHours())}${pad(t.getUTCMinutes())}${pad(t.getUTCSeconds())}Z`;
  };
  const lines = [
    'BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Youngmin Cho//Deadline tracker//EN',
    'CALSCALE:GREGORIAN', 'METHOD:PUBLISH', 'X-WR-CALNAME:Paper deadlines',
  ];
  for (const d of list) {
    const r = d.round.label ? ` ${d.round.label}` : '';
    const title = `${d.venue.name}${r} — ${d.kind}${d.venue.status === 'projected' ? ' (estimated)' : ''}`;
    const desc = [
      d.venue.fullName,
      d.venue.edition,
      d.round.timezone ? `Call states ${d.round.timezone}` : '',
      d.venue.location ? `Held in ${d.venue.location}` : '',
      d.round.notification ? `Notification ${d.round.notification}` : '',
      d.venue.status === 'projected' ? `Estimated from ${d.venue.basis ?? 'previous editions'} — confirm on the official site` : '',
      d.venue.site ?? '',
    ].filter(Boolean).join('\n');
    lines.push(
      'BEGIN:VEVENT',
      `UID:${d.key}@youngmin17.github.io`,
      `DTSTAMP:${stamp}`,
      `DTSTART:${utc(d.at - 3_600_000)}`,
      `DTEND:${utc(d.at)}`,
      `SUMMARY:${esc(title)}`,
      `DESCRIPTION:${esc(desc)}`,
      ...(d.venue.site ? [`URL:${d.venue.site}`] : []),
      'BEGIN:VALARM', 'TRIGGER;RELATED=END:-P7D', 'ACTION:DISPLAY', `DESCRIPTION:${esc(title)} in one week`, 'END:VALARM',
      'BEGIN:VALARM', 'TRIGGER;RELATED=END:-P1D', 'ACTION:DISPLAY', `DESCRIPTION:${esc(title)} tomorrow`, 'END:VALARM',
      'END:VEVENT',
    );
  }
  lines.push('END:VCALENDAR');
  return lines.flatMap(fold).join('\r\n') + '\r\n';
}

/* RFC 5545 §3.1: fold at 75 OCTETS, never inside a multi-octet character, continuation lines
   beginning with one space. Folding on string length instead splits em dashes down the middle. */
function fold(line: string): string[] {
  const enc = new TextEncoder();
  const parts: string[] = [];
  let cur = '';
  let used = 0;
  for (const ch of line) {
    const n = enc.encode(ch).length;
    const limit = parts.length === 0 ? 75 : 74; // continuations spend one octet on the leading space
    if (used + n > limit) { parts.push(cur); cur = ''; used = 0; }
    cur += ch;
    used += n;
  }
  if (cur || !parts.length) parts.push(cur);
  return parts.map((p, i) => (i === 0 ? p : ` ${p}`));
}
