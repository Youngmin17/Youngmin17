import { useEffect, useMemo, useState } from 'react';
import { FIELDS, deadlines, fieldOf, toIcs, type Deadline, type Field, type Venue } from './types';
import { CHECKED, VENUES } from './venues';
import './cal.css';

const DAY = 86_400_000;
const utc = (iso: string) => Date.parse(`${iso}T00:00:00Z`);
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const pretty = (iso: string) => {
  const d = new Date(utc(iso));
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
};
const withYear = (iso: string) => `${pretty(iso)} ${new Date(utc(iso)).getUTCFullYear()}`;

const countdown = (days: number) =>
  days < 0 ? 'passed'
    : days === 0 ? 'today'
    : days === 1 ? 'tomorrow'
    : days < 21 ? `${days} days`
    : days < 60 ? `${Math.round(days / 7)} weeks`
    : `${Math.round(days / 30.4)} months`;

/* Today, in UTC, resolved in the browser so a page built months ago still counts down correctly. */
function useToday() {
  const [t, setT] = useState<number | null>(null);
  useEffect(() => {
    const set = () => {
      const n = new Date();
      setT(Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate()));
    };
    set();
    const id = setInterval(set, 60 * 60 * 1000);
    return () => clearInterval(id);
  }, []);
  return t;
}

export default function Calendar() {
  const today = useToday();
  const [on, setOn] = useState<Set<Field>>(new Set(FIELDS.map((f) => f.id)));
  const [open, setOpen] = useState<string | null>(null);

  const all = useMemo(() => deadlines(VENUES as Venue[]), []);
  const shown = useMemo(() => all.filter((d) => on.has(d.venue.field)), [all, on]);

  const toggle = (f: Field) => setOn((s) => {
    const n = new Set(s);
    if (n.has(f) && n.size === FIELDS.length) return new Set([f]); // first click isolates
    if (n.has(f)) n.delete(f); else n.add(f);
    return n.size ? n : new Set(FIELDS.map((x) => x.id));
  });

  const daysTo = (iso: string) => (today == null ? null : Math.round((utc(iso) - today) / DAY));
  const upcoming = today == null ? [] : shown.filter((d) => utc(d.date) >= today);
  const next3 = (upcoming.length ? upcoming : shown).slice(0, 3);

  /* From this month through the last deadline on file, so nothing sits off the edge. Spanned over
     every venue rather than the filtered set, so the axis does not jump when you filter. */
  const months = useMemo(() => {
    if (today == null || !all.length) return [];
    const base = new Date(today);
    const last = new Date(utc(all[all.length - 1].date));
    const span = Math.min(18, Math.max(12,
      (last.getUTCFullYear() - base.getUTCFullYear()) * 12 + (last.getUTCMonth() - base.getUTCMonth()) + 1));
    return Array.from({ length: span }, (_, i) => {
      const y = base.getUTCFullYear();
      const m = base.getUTCMonth() + i;
      return { y: y + Math.floor(m / 12), m: ((m % 12) + 12) % 12, i };
    });
  }, [today, all]);

  const download = () => {
    const n = new Date();
    const p = (x: number) => String(x).padStart(2, '0');
    const stamp = `${n.getUTCFullYear()}${p(n.getUTCMonth() + 1)}${p(n.getUTCDate())}T${p(n.getUTCHours())}${p(n.getUTCMinutes())}${p(n.getUTCSeconds())}Z`;
    const blob = new Blob([toIcs(shown, stamp)], { type: 'text/calendar;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'paper-deadlines.ics';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const detail = open ? shown.find((d) => d.key === open) : null;

  return (
    <div className="dl">
      {/* ---- next up ---- */}
      <div className="dl-next">
        {next3.map((d) => {
          const n = daysTo(d.date);
          const c = fieldOf(d.venue.field).color;
          return (
            <div key={d.key} className={`dl-up${n != null && n < 0 ? ' past' : n != null && n <= 21 ? ' soon' : ''}`}
              style={{ ['--c' as string]: c }}>
              <div className="days">
                {n == null ? '—' : n < 0 ? 'past' : n}
                <small>{n == null ? '' : n < 0 ? '' : n === 1 ? 'day left' : 'days left'}</small>
              </div>
              <div className="who">{d.venue.name}{d.round.label && ` · ${d.round.label}`}</div>
              <div className="what">
                {d.kind === 'abstract' ? 'abstract' : 'paper'} · {withYear(d.date)}
                {d.venue.status === 'projected' && ' · estimated'}
              </div>
            </div>
          );
        })}
      </div>

      {/* ---- filters ---- */}
      <div className="dl-bar">
        {FIELDS.map((f) => (
          <button key={f.id} type="button" className="dl-f" aria-pressed={on.has(f.id)}
            style={{ ['--c' as string]: f.color }} onClick={() => toggle(f.id)}>
            <i />{f.label}
          </button>
        ))}
        <button type="button" className="dl-ics" onClick={download}>
          Download {shown.length} deadlines (.ics)
        </button>
      </div>

      {/* ---- the year ---- */}
      <div className="dl-year">
        <div className="dl-months" style={{ ['--n' as string]: months.length }}>
          {months.map(({ y, m, i }) => {
            const inMonth = shown.filter((d) => {
              const t = new Date(utc(d.date));
              return t.getUTCFullYear() === y && t.getUTCMonth() === m;
            });
            return (
              <div key={`${y}-${m}`} className={`dl-mo${i === 0 ? ' now' : ''}`}>
                <h4>{MONTHS[m]}{m === 0 || i === 0 ? ` ’${String(y).slice(2)}` : ''}</h4>
                <div className="stack">
                  {inMonth.map((d) => (
                    <button key={d.key} type="button"
                      className={`dl-pin${d.venue.status === 'projected' ? ' est' : ''}${d.kind === 'abstract' ? ' abs' : ''}`}
                      style={{ ['--c' as string]: fieldOf(d.venue.field).color }}
                      aria-expanded={open === d.key}
                      onClick={() => setOpen(open === d.key ? null : d.key)}>
                      <b>{d.venue.name}</b>
                      <span>{new Date(utc(d.date)).getUTCDate()}{d.kind === 'abstract' ? ' abs' : ''}</span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
        {detail && <Detail d={detail} />}
      </div>

      {/* ---- table ---- */}
      <div className="dl-wrap">
        <div className="head">
          <h3>Every deadline, in order</h3>
          <span>{shown.length} across {new Set(shown.map((d) => d.venue.id)).size} venues</span>
        </div>
        <div className="dl-scroll">
          <table className="dl-table">
            <thead>
              <tr><th>Venue</th><th>What</th><th>Date</th><th className="r">Countdown</th><th>Conference</th></tr>
            </thead>
            <tbody>
              {shown.map((d) => {
                const n = daysTo(d.date);
                return (
                  <tr key={d.key} className={n != null && n < 0 ? 'past' : ''}>
                    <td>
                      <span className="v" style={{ ['--c' as string]: fieldOf(d.venue.field).color }}>
                        <i />{d.venue.name}
                        {d.venue.status === 'projected' && <span className="dl-est">est</span>}
                      </span>
                      <span className="sub">{d.venue.edition}</span>
                    </td>
                    <td>{d.kind === 'abstract' ? 'Abstract' : 'Full paper'}{d.round.label && <span className="sub">{d.round.label}</span>}</td>
                    <td className="d">{withYear(d.date)}{d.round.timezone && <span className="sub">{d.round.timezone}</span>}</td>
                    <td className={`r cd${n != null && n >= 0 && n <= 21 ? ' soon' : ''}`}>{n == null ? '—' : countdown(n)}</td>
                    <td>
                      {d.venue.confStart ? withYear(d.venue.confStart) : '—'}
                      {d.venue.location && <span className="sub">{d.venue.location}</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="dl-foot">
        Checked against official calls for papers on {withYear(CHECKED)}. Dates marked{' '}
        <span className="dl-est">est</span> had no published call at that point and are projected from
        the editions named in each entry — confirm one before planning around it. Countdowns run in
        your browser, so they stay right however old this page is.
      </p>
    </div>
  );
}

function Detail({ d }: { d: Deadline }) {
  const v = d.venue;
  return (
    <div className="dl-detail">
      <h4>{v.name} — {v.fullName}</h4>
      <p className="meta">
        {v.edition}
        {v.location && ` · ${v.location}`}
        {v.confStart && ` · ${withYear(v.confStart)}${v.confEnd ? `–${pretty(v.confEnd)}` : ''}`}
      </p>
      <p>
        {d.round.label && <>{d.round.label}: </>}
        {d.round.abstract && <>abstract {withYear(d.round.abstract)}, </>}
        paper {withYear(d.round.paper)}
        {d.round.timezone && ` ${d.round.timezone}`}
        {d.round.notification && <>, notification {withYear(d.round.notification)}</>}.
      </p>
      {v.status === 'projected' && <p><b>Estimated.</b> {v.basis ?? 'Projected from previous editions.'}</p>}
      {v.notes && <p>{v.notes}</p>}
      {v.site && <p><a href={v.site} target="_blank" rel="noopener noreferrer">Official site →</a></p>}
    </div>
  );
}
