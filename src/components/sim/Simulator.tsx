import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BATCH_STEPS, CLASSIC, CTX_STEPS, GPUS, LINK, MODELS, PRECISIONS, SERVING, VIDEO_SHAPES,
  type Precision, type Serving,
} from './data';
import { bytes, compact, num, secs, simulate, type Cfg } from './engine';
import './sim.css';

/* Accelerator-era counterparts to the 2010 table. Same units, same log scale. */
const MODERN: { label: string; ns: number }[] = [
  { label: 'GPU shared-memory reference', ns: 20 },
  { label: 'Read 1 MB from HBM3e at 8 TB/s', ns: 125 },
  { label: 'HBM3e reference', ns: 400 },
  { label: 'NVLink 5 hop, 1 MB', ns: 2_600 },
  { label: 'NVMe Gen5 random read', ns: 16_000 },
  { label: 'InfiniBand NDR hop, 1 MB', ns: 22_000 },
];

const DEFAULT: Cfg = {
  modelId: 'l70b', precision: 'fp8', gpuId: 'h200', batch: 32, seq: 8192,
  videoId: '480p5', steps: 50, gpuOverride: null, serving: 'sarathi', offload: false,
};

const useReducedMotion = () => {
  const [r, setR] = useState(false);
  useEffect(() => {
    const q = window.matchMedia('(prefers-reduced-motion: reduce)');
    setR(q.matches);
    const on = () => setR(q.matches);
    q.addEventListener('change', on);
    return () => q.removeEventListener('change', on);
  }, []);
  return r;
};

/* A literal picture of tokens per second: words arrive at exactly the simulated rate. */
const SAMPLE = ('every weight has to cross the memory bus before this word can exist which is '
  + 'why the tensor cores spend most of their life idle waiting on HBM ').split(' ');

function Stream({ tpot, still }: { tpot: number; still: boolean }) {
  const [n, setN] = useState(0);
  const acc = useRef(0);
  useEffect(() => {
    if (still || !isFinite(tpot) || tpot <= 0) return;
    let raf = 0, last = performance.now();
    const period = tpot * 1000;
    const loop = (t: number) => {
      acc.current += t - last; last = t;
      if (acc.current >= period) {
        const k = Math.min(24, Math.floor(acc.current / period));
        acc.current -= k * period;
        setN((v) => v + k);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [tpot, still]);

  if (still) return <div className="rf-stream">…every weight crosses the memory bus per token</div>;
  const out: string[] = [];
  let len = 0;
  for (let i = n; i >= 0 && len < 30; i--) {
    const w = SAMPLE[i % SAMPLE.length];
    if (len + w.length + 1 > 30) break;
    out.unshift(w); len += w.length + 1;
  }
  return <div className="rf-stream" aria-hidden="true"><b>{out.join(' ')} </b><i>▌</i></div>;
}

/* ---- roofline plot ------------------------------------------------------ */
const W = 560, H = 336, ML = 44, MR = 14, MT = 26, MB = 34;
const PW = W - ML - MR, PH = H - MT - MB;
const AX0 = Math.log10(0.3), AX1 = Math.log10(2e5);
const AY0 = Math.log10(0.05), AY1 = Math.log10(25000);
const px = (ai: number) => ML + (Math.log10(Math.max(ai, 1e-3)) - AX0) / (AX1 - AX0) * PW;
const pxc = (ai: number) => Math.min(ML + PW - 4, Math.max(ML + 4, px(ai)));
const py = (tf: number) => MT + (1 - (Math.log10(Math.max(tf, 1e-3)) - AY0) / (AY1 - AY0)) * PH;
const decade = (e: number) => (e < 0 ? `0.${'0'.repeat(-e - 1)}1` : e === 0 ? '1' : `1e${e}`);

function Roofline({ cfg, sim }: { cfg: Cfg; sim: ReturnType<typeof simulate> }) {
  const roofs = GPUS.map((g) => {
    const nat = g.flops[cfg.precision] != null;
    const peakT = (g.flops[cfg.precision] ?? g.flops.bf16!) * (nat ? 1 : 0.85);
    const ridge = peakT / g.bw;
    return { g, peakT, ridge, on: g.id === cfg.gpuId };
  });
  const pts = sim.video
    ? [{ k: 'denoise step', ai: sim.video.ai, ach: sim.video.ach / 1e12, c: 'var(--cmp)' }]
    : [
        { k: 'prefill', ai: sim.aiPrefill, ach: sim.achPrefill / 1e12, c: 'var(--cmp)' },
        { k: 'decode', ai: sim.aiDecode, ach: sim.achDecode / 1e12, c: 'var(--mem)' },
      ];

  return (
    <svg className="rf-plot" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Roofline. Ridge point ${Math.round(sim.ridge)} FLOP per byte. Decode sits at ${sim.aiDecode.toFixed(1)} FLOP per byte.`}>
      {[0, 1, 2, 3, 4, 5].map((e) => (
        <g key={`x${e}`}>
          <line className="ax" x1={px(10 ** e)} y1={MT} x2={px(10 ** e)} y2={MT + PH} />
          <text x={px(10 ** e)} y={MT + PH + 14} textAnchor="middle">{decade(e)}</text>
        </g>
      ))}
      {[-1, 0, 1, 2, 3, 4].map((e) => (
        <g key={`y${e}`}>
          <line className="ax" x1={ML} y1={py(10 ** e)} x2={ML + PW} y2={py(10 ** e)} />
          <text x={ML - 7} y={py(10 ** e) + 3} textAnchor="end">{decade(e)}</text>
        </g>
      ))}
      <text x={ML + PW} y={MT + PH + 28} textAnchor="end">arithmetic intensity — FLOP / byte</text>
      <text x={6} y={11}>TFLOP/s per GPU</text>

      {roofs.filter((r) => !r.on).map((r) => (
        <path key={r.g.id} className="ghost"
          d={`M${px(0.3)},${py(r.g.bw * 0.3)} L${px(r.ridge)},${py(r.peakT)} L${px(2e5)},${py(r.peakT)}`} />
      ))}
      {roofs.filter((r) => r.on).map((r) => (
        <g key={r.g.id}>
          <path className="roof" stroke="var(--ink)"
            d={`M${px(0.3)},${py(r.g.bw * 0.3)} L${px(r.ridge)},${py(r.peakT)} L${px(2e5)},${py(r.peakT)}`} />
          <line x1={px(r.ridge)} y1={py(r.peakT)} x2={px(r.ridge)} y2={MT + PH}
            stroke="var(--ink)" strokeWidth="1" strokeDasharray="2 3" opacity=".35" />
          <text x={px(r.ridge) + 5} y={MT + PH - 6} fill="var(--ink-2)" fontSize="9.5">
            ridge {Math.round(r.ridge)}
          </text>
          <text x={px(r.ridge) + 6} y={py(r.peakT) - 7} fill="var(--ink)" fontSize="10" fontWeight="600">
            {r.g.short} {cfg.precision.toUpperCase()} peak
          </text>
        </g>
      ))}

      {pts.map((p) => (
        <g key={p.k}>
          <circle cx={pxc(p.ai)} cy={py(p.ach)} r="11" fill={p.c} opacity=".14" />
          <circle cx={pxc(p.ai)} cy={py(p.ach)} r="4.5" fill={p.c} stroke="#fff" strokeWidth="1.5" />
          <text className="pt-lab" fill={p.c} y={py(p.ach) + 3.5}
            x={pxc(p.ai) > ML + PW * 0.72 ? pxc(p.ai) - 11 : pxc(p.ai) + 11}
            textAnchor={pxc(p.ai) > ML + PW * 0.72 ? 'end' : 'start'}>{p.k}</text>
        </g>
      ))}
    </svg>
  );
}

/* ---- controls ----------------------------------------------------------- */
function Chip({ on, label, sub, onClick, disabled, warn }: {
  on: boolean; label: string; sub?: string; onClick: () => void; disabled?: boolean; warn?: boolean;
}) {
  return (
    <button type="button" className={`rf-chip${warn ? ' warn' : ''}`} aria-pressed={on}
      disabled={disabled} onClick={onClick}>
      {label}{sub && <small>{sub}</small>}
    </button>
  );
}

function Slider({ label, hint, steps, value, onChange, fmt, unit }: {
  label: string; hint: string; steps: number[]; value: number;
  onChange: (v: number) => void; fmt: (v: number) => string; unit: string;
}) {
  const i = Math.max(0, steps.indexOf(value));
  return (
    <div className="rf-cell rf-slide">
      <p className="rf-lab"><span>{label}</span><em>{hint}</em></p>
      <div className="val">{fmt(value)}<span>{unit}</span></div>
      <input type="range" min={0} max={steps.length - 1} step={1} value={i}
        aria-label={label} aria-valuetext={`${fmt(value)} ${unit}`}
        onChange={(e) => onChange(steps[+e.target.value])} />
      <div className="rf-ticks"><span>{fmt(steps[0])}</span><span>{fmt(steps[steps.length - 1])}</span></div>
    </div>
  );
}

/* ---- main --------------------------------------------------------------- */
export default function Simulator() {
  const [cfg, setCfg] = useState<Cfg>(DEFAULT);
  const still = useReducedMotion();
  const set = (p: Partial<Cfg>) => setCfg((c) => ({ ...c, ...p }));

  const model = MODELS.find((m) => m.id === cfg.modelId)!;
  const isDit = model.kind === 'dit';
  const ctxSteps = useMemo(() => CTX_STEPS.filter((c) => c <= model.maxCtx), [model]);
  const seq = Math.min(cfg.seq, ctxSteps[ctxSteps.length - 1]);
  const sim = useMemo(() => simulate({ ...cfg, seq }), [cfg, seq]);

  const g = sim.gpu;
  const pctOfPeak = ((sim.video ? sim.video.ach : sim.achDecode) / sim.peak) * 100;

  /* ---- ladder ---------------------------------------------------------- */
  const live = useMemo(() => {
    const rows: { label: string; ns: number; live: true }[] = [];
    const add = (label: string, s: number) => { if (isFinite(s) && s > 0) rows.push({ label, ns: s * 1e9, live: true }); };
    if (isDit && sim.video) {
      add('one denoise step', sim.video.perStep);
      add(`full ${sim.video.label} clip`, sim.video.total);
      add('read the weights once', sim.weightBytes / (sim.nGpu * g.bw * 1e12));
    } else {
      add('read this token’s KV cache', sim.kvPerToken / (g.bw * 1e12));
      add('all-reduce, one layer', sim.tComm / Math.max(1, model.layers * 2));
      add('one decode step', sim.tpot);
      add('time to first token', sim.ttft);
    }
    return rows;
  }, [sim, isDit, model, g]);

  const rungs = useMemo(() => {
    const all = [
      ...CLASSIC.map((r) => ({ ...r, kind: 'classic' as const })),
      ...MODERN.map((r) => ({ ...r, kind: 'modern' as const })),
      ...live.map((r) => ({ label: r.label, ns: r.ns, kind: 'live' as const })),
    ].sort((a, b) => a.ns - b.ns);
    const top = Math.max(9, Math.ceil(Math.log10(all[all.length - 1].ns)));
    return { all, top };
  }, [live]);

  const tpotNs = sim.tpot * 1e9;
  const dcTrips = tpotNs / 500_000;

  /* ---- field notes ------------------------------------------------------ */
  const notes: { k: string; t: string; body: string }[] = [];
  if (!sim.native) notes.push({
    k: 'cmp', t: `${g.short} has no native ${cfg.precision.toUpperCase()}`,
    body: `The 4-bit weights still halve what you drag across the bus, so the memory win is real. The math is not: values are dequantized to BF16 in-kernel and run at ${num(g.flops.bf16! / 1e3, 2)} PFLOP/s, so prefill gets no faster.`,
  });
  if (model.attn === 'mla') notes.push({
    k: 'mem', t: 'MLA caches a latent, not heads',
    body: `One 576-element vector per token per layer instead of 2 × kv_heads × head_dim — ${bytes(sim.kvPerToken)} a token, roughly 9× under a GQA model of this depth. It is also replicated on every tensor-parallel rank, which is why DeepSeek runs attention data-parallel instead.`,
  });
  if (model.attn === 'mha' && sim.kvBytes > sim.weightBytes) notes.push({
    k: 'mem', t: 'The KV cache outweighs the model',
    body: `MHA keeps a K and a V for all ${model.heads} heads: ${bytes(sim.kvPerToken)} per token. At batch ${cfg.batch} × ${num(seq)} tokens that is ${bytes(sim.kvBytes)} of cache against ${bytes(sim.weightBytes)} of weights. Grouped-query attention exists for exactly this reason.`,
  });
  if (model.routed > 0) notes.push({
    k: 'mem', t: `${(sim.moeFrac * 100).toFixed(0)}% of the experts get touched`,
    body: `Each token picks ${model.topk} of ${model.experts} experts, so a batch of ${cfg.batch} lights up ${(sim.moeFrac * 100).toFixed(0)}% of them and you read ${bytes((model.dense + model.routed * sim.moeFrac) * sim.bytesPerWeight)} per step. All ${bytes(sim.weightBytes)} still has to be resident. Sparse compute, dense capacity.`,
  });
  if (sim.crossNode) notes.push({
    k: 'net', t: 'Tensor parallelism left the node',
    body: `${sim.nGpu} GPUs is ${sim.nodes} nodes, so all-reduce drops from NVLink at ${num(g.nvlink / 2)} GB/s to InfiniBand at ${num(LINK.ibBw / 1e9)} GB/s — ${(g.nvlink / 2 / (LINK.ibBw / 1e9)).toFixed(0)}× slower, ${model.layers * 2} times per token.`,
  });
  if (!sim.fits && !cfg.offload) notes.push({
    k: 'ssd', t: `${bytes(sim.spill)} short`,
    body: `${sim.nGpu} × ${g.short} holds ${bytes(sim.nGpu * g.hbm * 1e9 * 0.9)} of usable HBM and this configuration wants ${bytes(sim.memNeed)}. Give it ${sim.nAuto} GPUs, or turn on SSD offload and watch what happens.`,
  });
  if (cfg.offload && sim.spill > 0) notes.push({
    k: 'ssd', t: 'Streaming weights off NVMe',
    body: `${bytes(sim.spill)} does not fit, so it comes off the drive at ${num(LINK.ssdBw / 1e9)} GB/s — ${(g.bw * 1e12 / LINK.ssdBw).toFixed(0)}× slower than HBM, on the critical path of every single token. Offload buys capacity and pays for it in latency.`,
  });
  if (cfg.precision === 'fp4' && sim.native) notes.push({
    k: 'cmp', t: 'Low precision moves the roof, not the floor',
    body: `FP4 quadruples peak FLOP/s but bandwidth is unchanged, so the ridge point slides out to ${Math.round(sim.ridge)} FLOP/byte. Decode sits at ${sim.aiDecode.toFixed(1)}. Cheaper math makes a memory-bound kernel more memory-bound, not less.`,
  });
  if (isDit && sim.video) notes.push({
    k: 'cmp', t: `Compute bound at ${pctOfPeak.toFixed(0)}% of peak`,
    body: `Attention over ${num(sim.video.tokens)} tokens is quadratic, so a step does ${compact(sim.video.ach * sim.video.perStep)} FLOP against only ${bytes(sim.weightBytes / sim.nGpu)} of weights — ${compact(sim.video.ai)} FLOP per byte, far right of the ${Math.round(sim.ridge)} ridge. Video generation is the rare workload where the tensor cores are the thing you are waiting on, which is why step-distillation and sparse attention are where the wins are.`,
  });
  if (notes.length < 2 && !isDit && sim.bound === 'memory') notes.push({
    k: 'mem', t: `Running at ${pctOfPeak < 1 ? pctOfPeak.toFixed(2) : pctOfPeak.toFixed(1)}% of peak`,
    body: `Decode reads ${bytes(sim.weightBytes)} of weights to produce ${cfg.batch} token${cfg.batch > 1 ? 's' : ''}, which is ${sim.aiDecode.toFixed(1)} FLOP per byte against a ridge point of ${Math.round(sim.ridge)}. The tensor cores are not the problem. Batching is the only lever that moves this point right.`,
  });
  const shown = notes.slice(0, 4);

  const serving = SERVING.find((s) => s.id === cfg.serving)!;
  const boundColor = { memory: 'var(--mem)', compute: 'var(--cmp)', interconnect: 'var(--net)', storage: 'var(--ssd)' }[sim.bound];
  const boundWord = { memory: 'memory bound', compute: 'compute bound', interconnect: 'fabric bound', storage: 'storage bound' }[sim.bound];

  /* ---- time breakdown --------------------------------------------------- */
  const vid = sim.video;
  const memWins = vid ? vid.tMem >= vid.tCompute : sim.tWeights + sim.tKv >= sim.tCompute;
  const segs = vid
    ? ([
        memWins
          ? { k: 'Weights + activations through HBM', v: vid.tMem, c: 'var(--mem)' }
          : { k: 'Tensor cores — attention over every token pair', v: vid.tCompute, c: 'var(--cmp)' },
        memWins
          ? { k: 'Tensor cores (hidden under the loads)', v: 0, c: 'var(--cmp)' }
          : { k: 'HBM traffic (hidden under the math)', v: 0, c: 'var(--mem)' },
        { k: 'All-reduce over the fabric', v: sim.tComm, c: 'var(--net)' },
      ] as { k: string; v: number; c: string; fade?: boolean }[])
    : (memWins
    ? [
        { k: 'Weight bytes from HBM', v: sim.tWeights, c: 'var(--mem)' },
        { k: 'KV cache from HBM', v: sim.tKv, c: 'var(--mem)', fade: true },
        { k: 'Tensor cores (hidden under the loads)', v: 0, c: 'var(--cmp)' },
      ]
    : [
        { k: 'Tensor cores', v: sim.tCompute, c: 'var(--cmp)' },
        { k: 'HBM traffic (hidden under the math)', v: 0, c: 'var(--mem)' },
      ]
  ).concat([
    { k: 'All-reduce over the fabric', v: sim.tComm, c: 'var(--net)' },
    { k: 'Weights off NVMe', v: sim.tSsd, c: 'var(--ssd)' },
    { k: 'Launch + scheduler', v: 1.2e-4, c: 'var(--faint)' },
  ] as { k: string; v: number; c: string; fade?: boolean }[]);
  const segTotal = segs.reduce((a, s) => a + s.v, 0) || 1;

  return (
    <div className="rf">
      {/* ---------------- console ---------------- */}
      <div className="rf-console">
        <div className="rf-row">
          <div className="rf-cell">
            <p className="rf-lab"><span>Model</span><em>{model.family}</em></p>
            <div className="rf-chips">
              {MODELS.map((m) => (
                <Chip key={m.id} on={m.id === cfg.modelId} label={m.name} sub={m.tag}
                  onClick={() => set({ modelId: m.id, seq: Math.min(cfg.seq, m.maxCtx) })} />
              ))}
            </div>
          </div>
        </div>

        <div className="rf-row two">
          <div className="rf-cell">
            <p className="rf-lab"><span>Number format</span>
              <em>{sim.native ? 'native tensor-core path' : 'no native path — emulated'}</em></p>
            <div className="rf-chips">
              {PRECISIONS.map((p) => (
                <Chip key={p.id} on={p.id === cfg.precision} label={p.label}
                  sub={g.flops[p.id] ? p.note : 'emulated'}
                  onClick={() => set({ precision: p.id as Precision })} />
              ))}
            </div>
          </div>
          <div className="rf-cell">
            <p className="rf-lab"><span>Accelerator</span><em>{g.mem} · {g.year}</em></p>
            <div className="rf-chips">
              {GPUS.map((x) => (
                <Chip key={x.id} on={x.id === cfg.gpuId} label={x.short}
                  sub={`${x.hbm} GB · ${x.bw} TB/s`} onClick={() => set({ gpuId: x.id })} />
              ))}
            </div>
          </div>
        </div>

        <div className="rf-row two">
          <Slider label={isDit ? 'Concurrent clips' : 'Batch size'}
            hint={isDit ? 'requests in flight' : 'sequences decoding together'}
            steps={BATCH_STEPS} value={cfg.batch} onChange={(v) => set({ batch: v })}
            fmt={(v) => String(v)} unit={isDit ? 'clips' : 'seqs'} />
          {isDit ? (
            <div className="rf-cell">
              <p className="rf-lab"><span>Output</span><em>{cfg.steps} denoise steps × 2 for CFG</em></p>
              <div className="rf-chips">
                {VIDEO_SHAPES.map((v) => (
                  <Chip key={v.id} on={v.id === cfg.videoId} label={v.label}
                    sub={`${compact(v.tokens)} tok`} onClick={() => set({ videoId: v.id })} />
                ))}
              </div>
            </div>
          ) : (
            <Slider label="Context length" hint={`up to ${compact(model.maxCtx)} for this model`}
              steps={ctxSteps} value={seq} onChange={(v) => set({ seq: v })}
              fmt={(v) => (v >= 1024 ? `${v / 1024}K` : String(v))} unit="tokens" />
          )}
        </div>

        <div className="rf-row two">
          <div className="rf-cell">
            <p className="rf-lab"><span>GPU count</span>
              <em>{cfg.gpuOverride == null ? `auto — smallest power of two that fits` : 'pinned by hand'}</em></p>
            <div className="rf-chips">
              <Chip on={cfg.gpuOverride == null} label="Auto" sub={`${sim.nAuto} GPU${sim.nAuto > 1 ? 's' : ''}`}
                onClick={() => set({ gpuOverride: null })} />
              {[1, 2, 4, 8, 16, 32].map((n) => (
                <Chip key={n} on={cfg.gpuOverride === n} label={String(n)}
                  sub={n > g.perNode ? `${n / g.perNode} nodes` : 'nvlink'}
                  onClick={() => set({ gpuOverride: n })} />
              ))}
              <Chip on={cfg.offload} warn label="SSD offload" sub={`${LINK.ssdBw / 1e9} GB/s`}
                onClick={() => set({ offload: !cfg.offload })} />
            </div>
          </div>
          {isDit ? (
            <div className="rf-cell">
              <p className="rf-lab"><span>Denoise steps</span><em>every step is run twice for classifier-free guidance</em></p>
              <div className="rf-chips">
                {[20, 30, 50, 100].map((n) => (
                  <Chip key={n} on={cfg.steps === n} label={String(n)} sub={`${n * 2} forwards`}
                    onClick={() => set({ steps: n })} />
                ))}
              </div>
            </div>
          ) : (
            <div className="rf-cell">
              <p className="rf-lab"><span>Serving discipline</span><em>{serving.sub}</em></p>
              <div className="rf-serve">
                {SERVING.map((s) => (
                  <Chip key={s.id} on={s.id === cfg.serving} label={s.label} sub={s.sub}
                    onClick={() => set({ serving: s.id as Serving })} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ---------------- readout ---------------- */}
      <div className="rf-readout">
        <div className="rf-ro">
          <div className="k">Machines</div>
          <div className="v">{sim.nGpu} × {g.short}</div>
          <div className="s">
            <b>{bytes(sim.aggHbm * 1e9)}</b> of HBM{sim.crossNode && <> across {sim.nodes} nodes</>}
            <br /><b>{(sim.aggFlops / 1e15).toFixed(2)} PFLOP/s</b> · {(sim.aggBw / 1e12).toFixed(1)} TB/s
          </div>
        </div>
        {isDit && sim.video ? (
          <>
            <div className="rf-ro">
              <div className="k">One clip</div>
              <div className="v">{secs(sim.video.total)}</div>
              <div className="s"><b>{secs(sim.video.perStep)}</b> per denoise step × {cfg.steps} × 2</div>
            </div>
            <div className="rf-ro">
              <div className="k">Sequence</div>
              <div className="v">{num(sim.video.tokens)}</div>
              <div className="s">video tokens · {sim.video.px} · attention is <b>quadratic</b> in this</div>
            </div>
          </>
        ) : (
          <>
            <div className="rf-ro">
              <div className="k">Per output token</div>
              <div className="v">{secs(sim.tpot)}</div>
              <div className="s"><b>{num(sim.tokPerSec, sim.tokPerSec < 10 ? 2 : 0)} tok/s</b> for one user · p99 gap {secs(sim.itlP99)}</div>
              <Stream tpot={sim.tpot} still={still} />
            </div>
            <div className="rf-ro">
              <div className="k">Time to first token</div>
              <div className="v">{secs(sim.ttft)}</div>
              <div className="s">{num(seq)}-token prefill{cfg.serving === 'pd' && <> · +{secs(sim.kvXfer)} KV hop</>}</div>
            </div>
          </>
        )}
        <div className="rf-ro">
          <div className="k">Throughput</div>
          {isDit && sim.video ? (
            <>
              <div className="v">{(3600 / sim.video.total * cfg.batch).toFixed(1)}
                <span style={{ fontSize: '14px', opacity: .6 }}> clips/h</span></div>
              <div className="s">
                {secs(sim.video.total / (cfg.steps * 2))} per forward pass
                <br /><span className="rf-bound" style={{ color: boundColor }}><i />{boundWord}</span>
              </div>
            </>
          ) : (
            <>
              <div className="v">{compact(sim.sysTokPerSec)}<span style={{ fontSize: '14px', opacity: .6 }}> tok/s</span></div>
              <div className="s">
                {compact(sim.perGpuTokPerSec)} per GPU · {(sim.util * 100).toFixed(0)}% duty
                <br /><span className="rf-bound" style={{ color: boundColor }}><i />{boundWord}</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ---------------- plot + breakdown ---------------- */}
      <div className="rf-split">
        <div className="rf-panel">
          <h4>Where this configuration sits</h4>
          <p className="cap">
            The roof is {g.short} at {cfg.precision.toUpperCase()}; the two faint lines are the other
            accelerators for scale. Anything left of the ridge is waiting on memory no matter how good the kernel is.
          </p>
          <Roofline cfg={{ ...cfg, seq }} sim={sim} />
        </div>
        <div className="rf-panel">
          <h4>{isDit ? 'Where a denoise step goes' : 'Where a token’s time goes'}</h4>
          <p className="cap">{isDit
            ? `Attention runs over all ${num(sim.video?.tokens ?? 0)} video tokens at once, in both directions, at every layer — so the cost grows with the square of the clip. This is one of the few inference workloads that genuinely saturates the tensor cores.`
            : serving.blurb}</p>
          <div className="rf-bar" role="img" aria-label={`Time breakdown, ${boundWord}`}>
            {segs.filter((s) => s.v > 0).map((s) => (
              <div key={s.k} className="rf-seg"
                style={{ flexGrow: s.v / segTotal, background: s.c, opacity: s.fade ? .55 : 1 }} />
            ))}
          </div>
          <div className="rf-legend">
            {segs.map((s) => (
              <div key={s.k} className={`rf-leg${s.v > 0 ? '' : ' off'}`}>
                <i style={{ background: s.c, opacity: s.fade ? .55 : 1 }} />
                <span>{s.k}</span>
                <span className="n">{s.v > 0 ? secs(s.v) : '—'}</span>
              </div>
            ))}
          </div>
          {!isDit && (
            <div className="rf-serve-stats">
              <div><span>Time to first token</span><b>{secs(sim.ttft)}</b></div>
              <div><span>Worst gap between tokens</span><b>{secs(sim.itlP99)}</b></div>
              <div><span>Duty cycle</span><b>{(sim.util * 100).toFixed(0)}%</b></div>
            </div>
          )}
        </div>
      </div>

      {/* ---------------- field notes ---------------- */}
      {shown.length > 0 && (
        <div className="rf-notes">
          {shown.map((n) => (
            <div key={n.t} className={`rf-note ${n.k}`}><b>{n.t}</b>{n.body}</div>
          ))}
        </div>
      )}

      {/* ---------------- spec sheet ---------------- */}
      <div className="rf-sheet">
        <div className="rf-sheet-h">
          <h4>The napkin</h4>
          <span>{model.name} · {cfg.precision.toUpperCase()} · {sim.nGpu} × {g.short}</span>
        </div>
        <table className="rf-tbl">
          <thead>
            <tr><th>Quantity</th><th>Size</th><th>Time</th></tr>
          </thead>
          <tbody>
            <tr className="grp"><td colSpan={3}>What has to be resident</td></tr>
            <Row label={`Weights — ${compact(model.params)} params × ${sim.bytesPerWeight} byte${sim.bytesPerWeight === 1 ? '' : 's'}`} size={bytes(sim.weightBytes)} />
            <Row label={isDit ? 'KV cache — every step re-attends from scratch' : `KV cache — batch ${cfg.batch} × ${num(seq)} tokens`}
              size={isDit ? 'none — DiT is bidirectional' : bytes(sim.kvBytes)}
              time={isDit ? undefined : `${bytes(sim.kvPerToken)}/tok`} dim />
            <Row label="Activations, workspace, fragmentation" size={bytes(sim.actBytes)} />
            <Row label="Total against usable HBM" size={bytes(sim.memNeed)}
              time={`of ${bytes(sim.nGpu * g.hbm * 1e9 * 0.9)}`} dim />

            <tr className="grp"><td colSpan={3}>{isDit ? 'Per denoise step' : 'Per output token'}</td></tr>
            {sim.video ? (
              <>
                <Row label="Weights + activations through HBM"
                  size={bytes((sim.weightBytes + sim.actBytes) / sim.nGpu)} time={secs(sim.video.tMem)} />
                <Row label="Tensor-core work, one step"
                  size={`${compact(sim.video.ach * sim.video.perStep)} FLOP`} time={secs(sim.video.tCompute)} />
                <Row label={`All-reduce × ${model.layers * 2}`}
                  size={sim.nGpu > 1 ? bytes(sim.commBytes) : '—'} time={sim.nGpu > 1 ? secs(sim.tComm) : '—'} />
                <Row label={`Whole clip — ${cfg.steps} steps × 2 for CFG`} size="—" time={secs(sim.video.total)} dim />
              </>
            ) : (
              <>
                <Row label="Weight bytes pulled through HBM" size={bytes((model.dense + model.routed * sim.moeFrac) * sim.bytesPerWeight / sim.nGpu)} time={secs(sim.tWeights)} />
                <Row label="KV bytes pulled through HBM" size={bytes(sim.kvBytes / (model.attn === 'mla' ? 1 : sim.nGpu))} time={secs(sim.tKv)} />
                <Row label="Tensor-core work" size={`${compact(sim.achDecode * sim.tpot)} FLOP`} time={secs(sim.tCompute)} />
                <Row label={`All-reduce × ${model.layers * 2}`} size={sim.nGpu > 1 ? bytes(sim.commBytes) : '—'} time={sim.nGpu > 1 ? secs(sim.tComm) : '—'} />
                {sim.tSsd > 0 && <Row label="Weights streamed off NVMe" size={bytes(sim.spill)} time={secs(sim.tSsd)} />}
                <Row label="Kernel launch + scheduler" size="—" time={secs(1.2e-4)} dim />
              </>
            )}

            <tr className="grp"><td colSpan={3}>Roofline</td></tr>
            <Row label={`Peak, ${sim.nGpu} × ${g.short} at ${cfg.precision.toUpperCase()}`}
              size={`${(sim.aggFlops / 1e15).toFixed(2)} PFLOP/s`} time={sim.native ? 'native' : 'emulated'} dim />
            <Row label="Aggregate HBM bandwidth" size={`${(sim.aggBw / 1e12).toFixed(1)} TB/s`} />
            <Row label="Ridge point — where the roof bends" size={`${Math.round(sim.ridge)} FLOP/byte`} />
            <Row label={isDit ? 'This denoise step' : 'This decode step'}
              size={`${(sim.video ? sim.video.ai : sim.aiDecode).toFixed(1)} FLOP/byte`}
              time={`${pctOfPeak < 1 ? pctOfPeak.toFixed(2) : pctOfPeak.toFixed(1)}% of peak`} />
          </tbody>
        </table>
      </div>

      {/* ---------------- ladder ---------------- */}
      <div className="rf-ladder">
        <div className="rf-ladder-h">
          <h4>…and where that lands on the ladder</h4>
          <p>
            Jeff Dean's latency table, unchanged, on a log scale — with the 2026 accelerator
            equivalents and your simulated numbers dropped in beside them. Reading 1 MB of memory
            got 24× faster since 2010. An SSD random read and a datacenter round trip got nothing.
            That gap is the whole reason a decode step looks the way it does.
          </p>
          <div className="rf-key">
            <span><i style={{ background: 'var(--line-2)' }} />2010 table</span>
            <span><i style={{ background: 'var(--net)' }} />2026 accelerator</span>
            <span><i style={{ background: 'var(--mem)' }} />this configuration</span>
          </div>
        </div>
        <div className="rf-rungs">
          {rungs.all.map((r, i) => {
            const w = Math.max(1.5, (Math.log10(Math.max(r.ns, 1)) / rungs.top) * 100);
            const c = r.kind === 'live' ? 'var(--mem)' : r.kind === 'modern' ? 'var(--net)' : 'var(--line-2)';
            return (
              <div key={`${r.label}-${i}`} className={`rf-rung${r.kind === 'live' ? ' live' : ''}`}>
                <span className="t">{r.label}</span>
                <span className="track"><span className="fill" style={{ width: `${w}%`, background: c }} /></span>
                <span className="v">{secs(r.ns / 1e9)}</span>
              </div>
            );
          })}
        </div>
        <div className="rf-punch">
          {isDit && sim.video ? (
            <>One denoise step is <span className="mono">{secs(sim.video.perStep)}</span>, and a clip needs {cfg.steps * 2} of
            them. Video generation is the rare inference workload that is genuinely <b>compute bound</b> —
            attention over {num(sim.video.tokens)} tokens is quadratic, so the tensor cores finally have
            something to do.</>
          ) : (
            <>One decode step is <span className="mono">{secs(sim.tpot)}</span> — about{' '}
              <b>{compact(tpotNs)}</b> L1 cache references, or{' '}
              <b>{dcTrips >= 1 ? `${num(dcTrips, dcTrips < 10 ? 1 : 0)} round trips` : `${(dcTrips * 100).toFixed(0)}%`} across a datacenter</b>
              {sim.bound === 'memory' && <>, spent almost entirely waiting for {bytes((model.dense + model.routed * sim.moeFrac) * sim.bytesPerWeight)} of weights to come back from HBM</>}
              {sim.bound === 'interconnect' && <>, most of it in {model.layers * 2} all-reduces rather than in arithmetic</>}
              {sim.bound === 'storage' && <>, and {((sim.tSsd / sim.tpot) * 100).toFixed(0)}% of it is the NVMe drive</>}.
              {' '}Five hundred more of them — about {secs(sim.tpot * 500)} — and you have an answer.</>
          )}
        </div>
      </div>

      <p className="rf-foot">
        Order-of-magnitude only. Dense (non-sparse) tensor-core rates, HBM read at 80% of spec,
        MFU {`${55}`}% for prefill, one CUDA-graph replay per step, ring all-reduce, NDR InfiniBand at
        50 GB/s per GPU, one PCIe Gen5 NVMe drive at 14 GB/s. It does not model paged-attention
        fragmentation, prefix-cache hits, speculative decoding, or expert-parallel all-to-all at scale —
        so treat MoE at batch 1 as optimistic and everything else as ±2×.
      </p>
    </div>
  );
}

function Row({ label, size, time, dim }: { label: string; size: string; time?: string; dim?: boolean }) {
  return (
    <tr>
      <td>{label}</td>
      <td className={`n${dim ? ' dim' : ''}`}>{size}</td>
      <td className={`n${dim ? ' dim' : ''}`}>{time ?? ''}</td>
    </tr>
  );
}
