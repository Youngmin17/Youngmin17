import { useEffect, useMemo, useState } from 'react';
import {
  FIELDS, ZONE_LABEL, fieldOf, formatIn, instantOf, isPlainAoe, resolveDeadline, toIcs,
  type Deadline, type Field, type Venue, type Zone,
} from './types';
import { CHECKED, VENUES } from './venues';

/* The conference date is the reliable source — some editions are written "FAST '27". */
const yearOf = (v: Venue) => {
  if (v.confStart) return String(new Date(`${v.confStart}T00:00:00Z`).getUTCFullYear());
  const four = v.edition.match(/\b(20\d{2})\b/);
  if (four) return four[1];
  const two = v.edition.match(/-(\d{2})\b/);       // AAAI-28
  return two ? `20${two[1]}` : '';
};
import './cal.css';

const MO = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const day = (iso: string) => {
  const d = new Date(`${iso}T00:00:00Z`);
  return `${MO[d.getUTCMonth()]} ${d.getUTCDate()}`;
};
const dayYear = (iso: string) => `${day(iso)}, ${new Date(`${iso}T00:00:00Z`).getUTCFullYear()}`;
const HOUR = 3_600_000;
const RUNWAY = 120 * 24 * HOUR; // the window a progress bar fills over

/* One card per round, not per date: abstract and paper belong to the same submission. */
interface Card {
  id: string;
  venue: Venue;
  round: Venue['rounds'][number];
  due: number;          // the instant that matters — the paper deadline
  abstractAt?: number;
  asWritten?: string;   // set only when the call is not a plain 23:59 AoE deadline
}

function cards(venues: Venue[]): Card[] {
  const out: Card[] = [];
  for (const v of venues) {
    for (const [i, r] of v.rounds.entries()) {
      const resolved = resolveDeadline(r.paper, r.timezone);
      out.push({
        id: `${v.id}-${i}`,
        venue: v,
        round: r,
        due: resolved.at,
        abstractAt: r.abstract ? instantOf(r.abstract, r.timezone) : undefined,
        asWritten: isPlainAoe(resolved) ? undefined
          : !resolved.namedZone ? `${r.timezone || 'no zone given'} — read as 23:59 AoE`
          : resolved.namedTime ? r.timezone
          : `${r.timezone} — hour not stated, read as 23:59`,
      });
    }
  }
  return out.sort((a, b) => a.due - b.due);
}

/* The deadline a card is actually counting down to: the abstract while one is still ahead, since
   abstract registration is mandatory at most of these venues and missing it forfeits the cycle.
   The list has to be ordered by this and not by the paper deadline, or a card reading 21 days sits
   above one reading 19. */
const targetOf = (c: Card, at: number | null) =>
  at != null && c.abstractAt != null && c.abstractAt >= at ? c.abstractAt : c.due;

/* Ticks every minute so the hours field is honest rather than decorative. */
function useNow() {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export default function Calendar() {
  const now = useNow();
  const [on, setOn] = useState<Set<Field> | null>(null); // null = every category
  const [zone, setZone] = useState<Zone>('aoe');

  const all = useMemo(() => cards(VENUES as Venue[]), []);
  const shown = useMemo(() => (on ? all.filter((c) => on.has(c.venue.field)) : all), [all, on]);

  const upcoming = (now == null ? shown : shown.filter((c) => c.due >= now))
    .slice().sort((a, b) => targetOf(a, now) - targetOf(b, now) || a.due - b.due);
  const past = now == null ? [] : shown.filter((c) => c.due < now);

  const download = () => {
    const list: Deadline[] = [];
    for (const c of shown) {
      if (c.round.abstract && c.abstractAt != null)
        list.push({ venue: c.venue, round: c.round, kind: 'abstract', date: c.round.abstract, at: c.abstractAt, key: `${c.id}-a` });
      list.push({ venue: c.venue, round: c.round, kind: 'paper', date: c.round.paper, at: c.due, key: `${c.id}-p` });
    }
    const n = new Date();
    const p = (x: number) => String(x).padStart(2, '0');
    const stamp = `${n.getUTCFullYear()}${p(n.getUTCMonth() + 1)}${p(n.getUTCDate())}T${p(n.getUTCHours())}${p(n.getUTCMinutes())}${p(n.getUTCSeconds())}Z`;
    const blob = new Blob([toIcs(list, stamp)], { type: 'text/calendar;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'paper-deadlines.ics';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const nIcs = shown.reduce((t, c) => t + (c.round.abstract ? 2 : 1), 0);

  return (
    <div className="ddl">
      <div className="ddl-filters">
        <div className="ddl-group">
          <span className="ddl-glabel">Category</span>
          <div className="ddl-btns">
            <button type="button" className="ddl-btn" aria-pressed={on === null} onClick={() => setOn(null)}>All</button>
            {FIELDS.map((f) => (
              <button key={f.id} type="button" className="ddl-btn" aria-pressed={on !== null && on.has(f.id)}
                style={{ ['--c' as string]: f.color }}
                onClick={() => setOn((s) => {
                  if (s === null) return new Set([f.id]);
                  const n = new Set(s);
                  if (n.has(f.id)) n.delete(f.id); else n.add(f.id);
                  return n.size ? n : null;
                })}>
                <i />{f.label}
              </button>
            ))}
          </div>
        </div>
        <div className="ddl-group">
          <span className="ddl-glabel">Times in</span>
          <div className="ddl-btns">
            {(['aoe', 'utc', 'local'] as Zone[]).map((z) => (
              <button key={z} type="button" className="ddl-btn" aria-pressed={zone === z} onClick={() => setZone(z)}>
                {ZONE_LABEL[z]}
              </button>
            ))}
          </div>
        </div>
        <button type="button" className="ddl-export" onClick={download}>Export {nIcs} dates (.ics)</button>
      </div>

      <Section title="Upcoming deadlines" count={upcoming.length} kind="upcoming">
        {upcoming.map((c) => <Card key={c.id} c={c} now={now} zone={zone} />)}
      </Section>

      {past.length > 0 && (
        <Section title="Passed" count={past.length} kind="past">
          {past.slice().reverse().map((c) => <Card key={c.id} c={c} now={now} zone={zone} />)}
        </Section>
      )}

      <p className="ddl-foot">
        Checked against official calls for papers on {dayYear(CHECKED)}. Entries marked{' '}
        <span className="ddl-inferred">estimated</span> had no published call at that point — their
        dates come from the editions named on the card and should be confirmed before you plan around
        one. Each deadline is resolved from the wording its own call uses &mdash; most say 23:59 AoE,
        some do not &mdash; and then restated in the zone you pick. Countdowns run in your browser.
      </p>
    </div>
  );
}

function Section({ title, count, kind, children }: {
  title: string; count: number; kind: string; children: React.ReactNode;
}) {
  return (
    <section className={`ddl-section ddl-${kind}`}>
      <h2 className="ddl-section-title">{title}<span className="ddl-count">{count}</span></h2>
      <div className="ddl-grid">{children}</div>
    </section>
  );
}

function Card({ c, now, zone }: { c: Card; now: number | null; zone: Zone }) {
  const f = fieldOf(c.venue.field);
  const target = targetOf(c, now);
  const onAbstract = target !== c.due;
  const left = now == null ? null : target - now;
  const overdue = left != null && left < 0;
  const days = left == null ? null : Math.floor(Math.abs(left) / (24 * HOUR));
  const hrs = left == null ? null : Math.floor((Math.abs(left) % (24 * HOUR)) / HOUR);
  const urgency = left == null || overdue ? '' : left < 7 * 24 * HOUR ? ' ddl-urgent' : left < 30 * 24 * HOUR ? ' ddl-soon' : '';
  const pct = left == null ? 0 : Math.max(0, Math.min(100, (1 - left / RUNWAY) * 100));

  return (
    <article className={`ddl-card${overdue ? ' ddl-past-card' : ''}${urgency}`} style={{ ['--c' as string]: f.color }}>
      <header className="ddl-card-head">
        <div className="ddl-badges">
          <span className="ddl-tag">{f.label}</span>
          {c.venue.status === 'projected' && <span className="ddl-inferred">estimated</span>}
          {onAbstract && <span className="ddl-stage">to abstract</span>}
        </div>
        <div className="ddl-countdown">
          {days == null ? <span className="ddl-num">—</span> : (
            <>
              <span className="ddl-num">{days}</span><span className="ddl-unit">{days === 1 ? 'day' : 'days'}</span>
              <span className="ddl-num">{hrs}</span><span className="ddl-unit">{hrs === 1 ? 'hr' : 'hrs'}</span>
              {overdue && <span className="ddl-unit ddl-ago">ago</span>}
            </>
          )}
        </div>
      </header>

      <div className="ddl-card-body">
        <h3 className="ddl-name">
          {c.venue.site
            ? <a href={c.venue.site} target="_blank" rel="noopener noreferrer">
                <span className="ddl-conf">{c.venue.name}</span>
                {yearOf(c.venue) && <span className="ddl-year">{yearOf(c.venue)}</span>}
                {c.round.label && <span className="ddl-cycle">{c.round.label}</span>}
                <span className="ddl-ext" aria-hidden="true">↗</span>
              </a>
            : <>
                <span className="ddl-conf">{c.venue.name}</span>
                {yearOf(c.venue) && <span className="ddl-year">{yearOf(c.venue)}</span>}
                {c.round.label && <span className="ddl-cycle">{c.round.label}</span>}
              </>}
        </h3>
        <p className="ddl-full">{c.venue.fullName}</p>
      </div>

      <footer className="ddl-card-foot">
        {c.abstractAt != null && (
          <div className="ddl-row">
            <span className="ddl-label">Abstract</span>
            <time dateTime={new Date(c.abstractAt).toISOString()}>{formatIn(c.abstractAt, zone)} <span className="ddl-zone">{ZONE_LABEL[zone]}</span></time>
          </div>
        )}
        <div className="ddl-row ddl-primary">
          <span className="ddl-label">Submission</span>
          <time dateTime={new Date(c.due).toISOString()}>{formatIn(c.due, zone)} <span className="ddl-zone">{ZONE_LABEL[zone]}</span></time>
        </div>
        {c.asWritten && (
          <p className="ddl-asis">
            <b>Call states</b> {c.asWritten}
          </p>
        )}
        {c.round.notification && (
          <div className="ddl-row">
            <span className="ddl-label">Notification</span>
            <time dateTime={c.round.notification}>{dayYear(c.round.notification)}</time>
          </div>
        )}
        {c.venue.confStart && (
          <div className="ddl-row">
            <span className="ddl-label">Conference</span>
            <time dateTime={c.venue.confStart}>
              {c.venue.confEnd ? `${day(c.venue.confStart)} – ${dayYear(c.venue.confEnd)}` : dayYear(c.venue.confStart)}
            </time>
          </div>
        )}
        {c.venue.location && (
          <div className="ddl-row">
            <span className="ddl-label">Where</span>
            <span className="ddl-where">{c.venue.location}</span>
          </div>
        )}
        {c.venue.status === 'projected' && c.venue.basis && (
          <p className="ddl-basis"><b>Estimated from</b> {c.venue.basis}</p>
        )}
      </footer>
      <div className="ddl-progress" aria-hidden="true"><span style={{ width: `${pct}%` }} /></div>
    </article>
  );
}
