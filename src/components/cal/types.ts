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

/* Deadlines are quoted at 23:59 Anywhere-on-Earth unless a call says otherwise, and AoE is
   UTC−12. Resolving each one to a real instant is what lets the page show it in your own zone
   and count down in hours rather than whole days. */
export const AOE_OFFSET_MIN = -12 * 60;

export function instantOf(dateISO: string, tz = 'AoE'): number {
  const offset = /aoe/i.test(tz) || !tz ? AOE_OFFSET_MIN : 0; // anything else is treated as UTC
  return Date.parse(`${dateISO}T23:59:00Z`) - offset * 60_000;
}

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
  key: string;
}

export function deadlines(venues: Venue[]): Deadline[] {
  const out: Deadline[] = [];
  for (const v of venues) {
    for (const [i, r] of v.rounds.entries()) {
      if (r.abstract) out.push({ venue: v, round: r, kind: 'abstract', date: r.abstract, key: `${v.id}-${i}-a` });
      if (r.paper) out.push({ venue: v, round: r, kind: 'paper', date: r.paper, key: `${v.id}-${i}-p` });
    }
  }
  return out.sort((a, b) => a.date.localeCompare(b.date));
}

/* RFC 5545. All-day events with a one-week and a one-day alarm, which is what a deadline wants. */
export function toIcs(list: Deadline[], stamp: string): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const cal = (d: string) => d.replace(/-/g, '');
  const esc = (s: string) => s.replace(/([,;\\])/g, '\\$1').replace(/\n/g, '\\n');
  const next = (d: string) => {
    const t = new Date(`${d}T00:00:00Z`);
    t.setUTCDate(t.getUTCDate() + 1);
    return `${t.getUTCFullYear()}${pad(t.getUTCMonth() + 1)}${pad(t.getUTCDate())}`;
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
      d.venue.location ? `Held in ${d.venue.location}` : '',
      d.round.notification ? `Notification ${d.round.notification}` : '',
      d.venue.status === 'projected' ? `Estimated from ${d.venue.basis ?? 'previous editions'} — confirm on the official site` : '',
      d.venue.site ?? '',
    ].filter(Boolean).join('\n');
    lines.push(
      'BEGIN:VEVENT',
      `UID:${d.key}@youngmin17.github.io`,
      `DTSTAMP:${stamp}`,
      `DTSTART;VALUE=DATE:${cal(d.date)}`,
      `DTEND;VALUE=DATE:${next(d.date)}`,
      `SUMMARY:${esc(title)}`,
      `DESCRIPTION:${esc(desc)}`,
      ...(d.venue.site ? [`URL:${d.venue.site}`] : []),
      'BEGIN:VALARM', 'TRIGGER:-P7D', 'ACTION:DISPLAY', `DESCRIPTION:${esc(title)} in one week`, 'END:VALARM',
      'BEGIN:VALARM', 'TRIGGER:-P1D', 'ACTION:DISPLAY', `DESCRIPTION:${esc(title)} tomorrow`, 'END:VALARM',
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
