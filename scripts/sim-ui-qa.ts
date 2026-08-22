/* Interactive QA: drives the real published UI through Chrome DevTools Protocol and checks that
   what the page renders equals what the engine computes for the same configuration.
   Catches wiring bugs (a control writing to the wrong field, a stale memo, a formatter mismatch)
   that a pure-function sweep cannot see.

   Run: npx tsx scripts/sim-ui-qa.ts [url]   (default http://localhost:4321/Youngmin17/simulator/) */
import { spawn } from 'node:child_process';
import { BATCH_STEPS, CTX_STEPS, GPUS, MODELS, PRECISIONS, SERVING, VIDEO_SHAPES } from '../src/components/sim/data';
import { simulate, bytes, compact, num, secs, type Cfg } from '../src/components/sim/engine';

const URL = process.argv[2] ?? 'http://localhost:4321/Youngmin17/simulator/';
const PORT = 9333;
const PROFILE = '/private/tmp/claude-501/-Users-mincho-youngmin-git/8b6a1532-e09a-4d58-9241-bba1567f47e4/scratchpad/uiqa-profile';
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--disable-gpu', '--hide-scrollbars', `--remote-debugging-port=${PORT}`,
   `--user-data-dir=${PROFILE}`, 'about:blank'], { stdio: 'ignore' });

let list: any[] = [];
for (let i = 0; i < 60; i++) {
  try { list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (list.length) break; } catch {}
  await wait(250);
}
const target = list.find((t) => t.type === 'page');
if (!target) { console.error('no chrome target'); process.exit(1); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));

let msgId = 0;
const pending = new Map<number, (v: any) => void>();
const consoleErrors: string[] = [];
ws.addEventListener('message', (e: any) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)!(m.result); pending.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') consoleErrors.push('exception: ' + (m.params?.exceptionDetails?.exception?.description ?? '?'));
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
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: URL });
await wait(2600);

/* ---- page-side driver: apply a config by operating the real controls, then scrape ---- */
const DRIVER = `
window.__qa = {
  cell(labelText) {
    return [...document.querySelectorAll('.rf-cell')].find(
      (c) => c.querySelector('.rf-lab span')?.textContent?.trim() === labelText);
  },
  chip(labelText, chipText) {
    const scope = this.cell(labelText);
    if (!scope) throw new Error('no cell: ' + labelText);
    const b = [...scope.querySelectorAll('.rf-chip')].find(
      (x) => x.childNodes[0]?.textContent?.trim() === chipText);
    if (!b) throw new Error('no chip "' + chipText + '" in "' + labelText + '" — have: '
      + [...scope.querySelectorAll('.rf-chip')].map((x) => x.childNodes[0]?.textContent?.trim()).join('|'));
    return b;
  },
  press(labelText, chipText) {
    const b = this.chip(labelText, chipText);
    if (b.getAttribute('aria-pressed') !== 'true') b.click();
  },
  slide(ariaLabel, index) {
    const el = document.querySelector('input[aria-label="' + ariaLabel + '"]');
    if (!el) throw new Error('no slider: ' + ariaLabel);
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, String(index));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  },
  frame() { return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); },
  scrape() {
    const ro = [...document.querySelectorAll('.rf-ro')].map((d) => ({
      k: d.querySelector('.k').textContent.trim().toUpperCase(),
      v: d.querySelector('.v').textContent.trim(),
      s: d.querySelector('.s')?.textContent.replace(/\\s+/g, ' ').trim() ?? '',
    }));
    const sheet = [...document.querySelectorAll('.rf-tbl tbody tr')].map((tr) => {
      const td = [...tr.querySelectorAll('td')];
      return td.length === 3 ? { q: td[0].textContent.trim(), size: td[1].textContent.trim(), time: td[2].textContent.trim() } : null;
    }).filter(Boolean);
    const rungs = [...document.querySelectorAll('.rf-rung')].map((r) => ({
      label: r.querySelector('.t').textContent.trim(),
      value: r.querySelector('.v').textContent.trim(),
      width: parseFloat(r.querySelector('.fill').style.width),
      live: r.classList.contains('live'),
    }));
    const pts = [...document.querySelectorAll('.rf-plot .pt-lab')].map((t) => ({
      k: t.textContent.trim(), x: +t.getAttribute('x'), y: +t.getAttribute('y'),
    }));
    const pressed = [...document.querySelectorAll('.rf-chip[aria-pressed="true"]')]
      .map((b) => b.childNodes[0]?.textContent?.trim());
    const legend = [...document.querySelectorAll('.rf-leg')].map((l) => ({
      k: l.querySelector('span').textContent.trim(), n: l.querySelector('.n').textContent.trim(),
    }));
    const notes = [...document.querySelectorAll('.rf-note b')].map((b) => b.textContent.trim());
    return {
      ro, sheet, rungs, pts, pressed, legend, notes,
      bound: document.querySelector('.rf-bound')?.textContent?.trim() ?? null,
      punch: document.querySelector('.rf-punch')?.textContent?.replace(/\\s+/g, ' ').trim() ?? '',
      docScrollW: document.documentElement.scrollWidth,
      innerW: window.innerWidth,
      plotBox: (() => { const s = document.querySelector('.rf-plot'); return s ? s.getAttribute('viewBox') : null; })(),
    };
  },
};
'ready'`;
await evalIn(DRIVER);

const apply = async (c: Cfg) => {
  const m = MODELS.find((x) => x.id === c.modelId)!;
  const g = GPUS.find((x) => x.id === c.gpuId)!;
  const ctx = CTX_STEPS.filter((x) => x <= m.maxCtx);
  const isDit = m.kind === 'dit';
  const steps: string[] = [
    `__qa.press('Model', ${JSON.stringify(m.name)})`,
    `await __qa.frame()`,
    `__qa.press('Number format', ${JSON.stringify(PRECISIONS.find((p) => p.id === c.precision)!.label)})`,
    `__qa.press('Accelerator', ${JSON.stringify(g.short)})`,
    `__qa.slide(${JSON.stringify(isDit ? 'Concurrent clips' : 'Batch size')}, ${BATCH_STEPS.indexOf(c.batch)})`,
    isDit
      ? `__qa.press('Output', ${JSON.stringify(VIDEO_SHAPES.find((v) => v.id === c.videoId)!.label)})`
      : `__qa.slide('Context length', ${ctx.indexOf(c.seq)})`,
    `__qa.press('GPU count', ${JSON.stringify(c.gpuOverride == null ? 'Auto' : String(c.gpuOverride))})`,
    isDit
      ? `__qa.press('Denoise steps', ${JSON.stringify(String(c.steps))})`
      : `__qa.press('Serving discipline', ${JSON.stringify(SERVING.find((s) => s.id === c.serving)!.label)})`,
    `await __qa.frame()`,
  ];
  // offload is a toggle: press only when the desired state differs from the current one
  steps.splice(steps.length - 1, 0,
    `{ const b = __qa.chip('GPU count', 'SSD offload');
       if ((b.getAttribute('aria-pressed') === 'true') !== ${c.offload}) b.click(); }`);
  await evalIn(`(async () => { ${steps.join(';\n')}; await __qa.frame(); return __qa.scrape(); })()`);
  return evalIn(`(async () => { await __qa.frame(); return __qa.scrape(); })()`);
};

/* ---- expectations, formatted exactly as the component formats them ---- */
type Fail = { cfg: string; rule: string; detail: string };
const fails: Fail[] = [];
const tagOf = (c: Cfg) => `${c.modelId}/${c.precision}/${c.gpuId}/b${c.batch}/s${c.seq}/n${c.gpuOverride ?? 'auto'}/${c.serving}${c.offload ? '/ssd' : ''}${c.videoId}/st${c.steps}`;

function expectations(c: Cfg) {
  const s = simulate(c);
  const isDit = s.model.kind === 'dit';
  const pct = ((s.video ? s.video.ach : s.achDecode) / s.peak) * 100;
  const e: Record<string, string> = {
    MACHINES: `${s.nGpu} × ${s.gpu.short}`,
    THROUGHPUT: isDit
      ? `${(3600 / s.video!.total * c.batch).toFixed(1)} clips/h`
      : `${compact(s.sysTokPerSec)} tok/s`,
  };
  if (isDit) {
    e['ONE CLIP'] = secs(s.video!.total);
    e['SEQUENCE'] = num(s.video!.tokens);
  } else {
    e['PER OUTPUT TOKEN'] = secs(s.tpot);
    e['TIME TO FIRST TOKEN'] = secs(s.ttft);
  }
  return { s, e, pct };
}

/* ---- config matrix ---- */
const base: Cfg = { modelId: 'l70b', precision: 'fp8', gpuId: 'h200', batch: 32, seq: 8192, videoId: '480p5', steps: 50, gpuOverride: null, serving: 'sarathi', offload: false };
const cfgs: Cfg[] = [];
for (const m of MODELS) {
  const ctx = CTX_STEPS.filter((x) => x <= m.maxCtx);
  for (const p of PRECISIONS) for (const g of GPUS)
    cfgs.push({ ...base, modelId: m.id, precision: p.id, gpuId: g.id, seq: ctx[Math.floor(ctx.length / 2)], batch: 8 });
}
for (const b of BATCH_STEPS) cfgs.push({ ...base, batch: b });
for (const seq of CTX_STEPS) cfgs.push({ ...base, seq });
for (const sv of SERVING) cfgs.push({ ...base, serving: sv.id });
for (const ov of [null, 1, 2, 4, 8, 16, 32]) cfgs.push({ ...base, gpuOverride: ov });
for (const v of VIDEO_SHAPES) for (const st of [20, 30, 50, 100]) cfgs.push({ ...base, modelId: 'wan14b', videoId: v.id, steps: st, batch: 1 });
cfgs.push({ ...base, modelId: 'l405b', precision: 'bf16', gpuId: 'a100', gpuOverride: 2, offload: true, serving: 'pd' });
cfgs.push({ ...base, modelId: 'dsv3', precision: 'fp8', gpuOverride: 32, batch: 64, seq: 32768, serving: 'orca' });
cfgs.push({ ...base, modelId: 'l13b', precision: 'bf16', seq: 4096, batch: 64, gpuOverride: 8 });

console.log(`driving ${cfgs.length} configurations through the real UI…\n`);

for (const c of cfgs) {
  const tag = tagOf(c);
  let dom: any;
  try { dom = await apply(c); } catch (err: any) { fails.push({ cfg: tag, rule: 'drive', detail: String(err.message).slice(0, 200) }); continue; }
  const { s, e, pct } = expectations(c);

  for (const [k, want] of Object.entries(e)) {
    const cell = dom.ro.find((r: any) => r.k === k);
    if (!cell) { fails.push({ cfg: tag, rule: 'readout-missing', detail: k }); continue; }
    if (cell.v.replace(/\s+/g, ' ') !== want.replace(/\s+/g, ' '))
      fails.push({ cfg: tag, rule: `readout:${k}`, detail: `dom="${cell.v}" engine="${want}"` });
  }
  const wantBound = { memory: 'memory bound', compute: 'compute bound', interconnect: 'fabric bound', storage: 'storage bound' }[s.bound];
  if (dom.bound !== wantBound) fails.push({ cfg: tag, rule: 'bound-badge', detail: `dom="${dom.bound}" engine="${wantBound}"` });

  // no placeholder or broken values anywhere the user can see
  const text = JSON.stringify(dom);
  for (const junk of ['NaN', 'undefined', 'Infinity', '[object']) if (text.includes(junk))
    fails.push({ cfg: tag, rule: 'junk-value', detail: junk });

  // spec sheet: weights row must equal the engine's byte count
  const wrow = dom.sheet.find((r: any) => r.q.startsWith('Weights —'));
  if (!wrow) fails.push({ cfg: tag, rule: 'sheet-weights-missing', detail: '' });
  else if (wrow.size !== bytes(s.weightBytes)) fails.push({ cfg: tag, rule: 'sheet-weights', detail: `dom="${wrow.size}" engine="${bytes(s.weightBytes)}"` });
  const ridge = dom.sheet.find((r: any) => r.q.startsWith('Ridge point'));
  if (ridge && ridge.size !== `${Math.round(s.ridge)} FLOP/byte`)
    fails.push({ cfg: tag, rule: 'sheet-ridge', detail: `dom="${ridge.size}" engine="${Math.round(s.ridge)}"` });
  const pctRow = dom.sheet.find((r: any) => r.q.startsWith('This '));
  const wantPct = `${pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}% of peak`;
  if (pctRow && pctRow.time !== wantPct) fails.push({ cfg: tag, rule: 'sheet-pct', detail: `dom="${pctRow.time}" engine="${wantPct}"` });

  // the ladder must stay sorted and the live rows must be present
  const secsOf = dom.rungs.map((r: any) => r.width);
  for (let i = 1; i < secsOf.length; i++) if (secsOf[i] < secsOf[i - 1] - 1e-9)
    fails.push({ cfg: tag, rule: 'ladder-sort', detail: `row ${i}: ${dom.rungs[i - 1].label} (${secsOf[i - 1]}%) then ${dom.rungs[i].label} (${secsOf[i]}%)` });
  if (!dom.rungs.some((r: any) => r.live)) fails.push({ cfg: tag, rule: 'ladder-no-live', detail: '' });
  if (dom.rungs.length < 18) fails.push({ cfg: tag, rule: 'ladder-short', detail: `${dom.rungs.length} rows` });

  // roofline points must stay inside the plot box
  const vb = (dom.plotBox ?? '0 0 560 336').split(' ').map(Number);
  for (const p of dom.pts) if (p.x < 0 || p.x > vb[2] || p.y < 0 || p.y > vb[3])
    fails.push({ cfg: tag, rule: 'plot-point-outside', detail: `${p.k} at ${p.x},${p.y} box ${vb[2]}x${vb[3]}` });
  const wantPts = s.video ? ['denoise step'] : ['prefill', 'decode'];
  if (dom.pts.length !== wantPts.length) fails.push({ cfg: tag, rule: 'plot-point-count', detail: `${dom.pts.map((p: any) => p.k).join(',')}` });

  // controls must reflect the requested state
  const g = GPUS.find((x) => x.id === c.gpuId)!;
  if (!dom.pressed.includes(g.short)) fails.push({ cfg: tag, rule: 'chip-state-gpu', detail: dom.pressed.join('|') });
  if (dom.pressed.includes('SSD offload') !== c.offload) fails.push({ cfg: tag, rule: 'chip-state-offload', detail: dom.pressed.join('|') });

  // horizontal overflow
  if (dom.docScrollW > dom.innerW + 1) fails.push({ cfg: tag, rule: 'overflow-x', detail: `${dom.docScrollW} > ${dom.innerW}` });
}

/* ---- keyboard + responsive spot checks ---- */
for (const w of [390, 768, 1024, 1440]) {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 600 });
  await wait(220);
  const r = await evalIn(`(async () => { await __qa.frame(); const d = __qa.scrape();
    return { over: d.docScrollW - d.innerW,
      offscreen: [...document.querySelectorAll('.rf *')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 1).map(e => e.className).slice(0, 6) }; })()`);
  if (r.over > 1) fails.push({ cfg: `viewport ${w}`, rule: 'overflow-x', detail: `${r.over}px, ${r.offscreen.join('|')}` });
}
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
const a11y = await evalIn(`(() => {
  const chips = [...document.querySelectorAll('.rf-chip')];
  const sliders = [...document.querySelectorAll('.rf-slide input[type=range]')];
  return {
    chipsNoAria: chips.filter(c => !c.hasAttribute('aria-pressed')).length,
    chipsNotButton: chips.filter(c => c.tagName !== 'BUTTON').length,
    slidersNoLabel: sliders.filter(s => !s.getAttribute('aria-label')).length,
    svgNoLabel: [...document.querySelectorAll('.rf-plot')].filter(s => !s.getAttribute('aria-label')).length,
    tableHeaders: document.querySelectorAll('.rf-tbl thead th').length,
  };
})()`);
if (a11y.chipsNoAria) fails.push({ cfg: 'a11y', rule: 'chip-aria-pressed', detail: String(a11y.chipsNoAria) });
if (a11y.chipsNotButton) fails.push({ cfg: 'a11y', rule: 'chip-not-button', detail: String(a11y.chipsNotButton) });
if (a11y.slidersNoLabel) fails.push({ cfg: 'a11y', rule: 'slider-aria-label', detail: String(a11y.slidersNoLabel) });
if (a11y.svgNoLabel) fails.push({ cfg: 'a11y', rule: 'svg-aria-label', detail: String(a11y.svgNoLabel) });

for (const e of consoleErrors) fails.push({ cfg: 'console', rule: 'console-error', detail: e.slice(0, 200) });

/* ---- report ---- */
console.log(`configurations driven : ${cfgs.length}`);
console.log(`console errors        : ${consoleErrors.length}`);
console.log(`failures              : ${fails.length}`);
const byRule = new Map<string, Fail[]>();
for (const f of fails) { if (!byRule.has(f.rule)) byRule.set(f.rule, []); byRule.get(f.rule)!.push(f); }
for (const [rule, l] of [...byRule].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`\n  ✗ ${rule}  (${l.length})`);
  for (const f of l.slice(0, 5)) console.log(`      ${f.cfg}\n        ${f.detail}`);
  if (l.length > 5) console.log(`      … ${l.length - 5} more`);
}
if (!fails.length) console.log('\nUI matches the engine on every configuration driven.');

ws.close(); chrome.kill();
process.exit(fails.length ? 1 : 0);
