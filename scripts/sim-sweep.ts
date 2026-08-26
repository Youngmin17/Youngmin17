/* Exhaustive invariant sweep over the simulator engine.
   Run: npx tsx scripts/sim-sweep.ts            (summary)
        npx tsx scripts/sim-sweep.ts --json     (machine-readable) */
import { BATCH_STEPS, CTX_STEPS, GPUS, LINK, MODELS, PRECISIONS, SERVING, VIDEO_SHAPES } from '../src/components/sim/data';
import { simulate, type Cfg } from '../src/components/sim/engine';

type Fail = { rule: string; cfg: string; detail: string };
const fails: Fail[] = [];
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const tag = (c: Cfg) =>
  `${c.modelId}/${c.precision}/${c.gpuId}/b${c.batch}/s${c.seq}/n${c.gpuOverride ?? 'auto'}/${c.serving}${c.offload ? '/ssd' : ''}${c.videoId !== '480p5' ? '/' + c.videoId : ''}`;
const bad = (rule: string, c: Cfg, detail: string) => fails.push({ rule, cfg: tag(c), detail });

const NUMERIC_FIELDS = [
  'nGpu', 'nAuto', 'nodes', 'aggFlops', 'aggBw', 'aggHbm', 'weightBytes', 'kvBytes', 'kvPerToken',
  'actBytes', 'memNeed', 'capacity', 'spill', 'moeFrac', 'tWeights', 'tKv', 'tCompute', 'tComm',
  'tSsd', 'tpot', 'tokPerSec', 'sysTokPerSec', 'perGpuTokPerSec', 'ttft', 'itlP99', 'kvXfer',
  'util', 'aiDecode', 'aiPrefill', 'achDecode', 'achPrefill', 'ridge', 'commBytes', 'prefillTokPerSec',
  'kvShards',
] as const;

function check(c: Cfg) {
  const s = simulate(c);
  const m = s.model, g = s.gpu;

  for (const f of NUMERIC_FIELDS) {
    const v = (s as any)[f] as number;
    if (!Number.isFinite(v)) bad('finite', c, `${f}=${v}`);
    else if (v < 0) bad('non-negative', c, `${f}=${v}`);
  }
  if (s.video) for (const f of ['perStep', 'total', 'ai', 'ach', 'tCompute', 'tMem', 'tComm', 'commBytes'] as const) {
    if (!Number.isFinite(s.video[f]) || s.video[f] < 0) bad('finite', c, `video.${f}=${s.video[f]}`);
  }

  // --- structural
  if (s.nGpu < 1 || (s.nGpu & (s.nGpu - 1)) !== 0) bad('gpu-count-pow2', c, `nGpu=${s.nGpu}`);
  if (s.nodes !== Math.ceil(s.nGpu / g.perNode)) bad('nodes', c, `${s.nodes}`);
  if (s.crossNode !== s.nGpu > g.perNode) bad('crossnode-flag', c, `${s.crossNode}`);
  if ((m.kind === 'dit') !== (s.video !== null)) bad('video-iff-dit', c, `${s.video !== null}`);

  // --- footprint identities
  if (!near(s.memNeed, s.weightBytes + s.kvBytes + s.actBytes, 1e-9)) bad('memNeed-sum', c, `${s.memNeed}`);
  if (!near(s.weightBytes, m.params * s.bytesPerWeight, 1e-9)) bad('weight-bytes', c, `${s.weightBytes}`);
  if (!near(s.capacity, s.nGpu * g.hbm * 1e9 * 0.9, 1e-9)) bad('capacity', c, `${s.capacity}`);
  if (!near(s.spill, Math.max(0, s.memNeed - s.capacity), 1e-6)) bad('spill', c, `${s.spill}`);
  if (s.fits !== (s.spill <= 0)) bad('fits-flag', c, `${s.fits}`);
  if (m.kind === 'dit' && s.kvBytes !== 0) bad('dit-no-kv', c, `${s.kvBytes}`);
  if (m.kind !== 'dit' && !near(s.kvBytes, c.batch * c.seq * s.kvPerToken, 1e-9)) bad('kv-bytes', c, `${s.kvBytes}`);

  // --- aggregate identities
  if (!near(s.aggBw, s.nGpu * g.bw * 1e12, 1e-9)) bad('agg-bw', c, `${s.aggBw}`);
  if (!near(s.aggHbm, s.nGpu * g.hbm, 1e-9)) bad('agg-hbm', c, `${s.aggHbm}`);
  if (!near(s.aggFlops, s.nGpu * s.peak, 1e-9)) bad('agg-flops', c, `${s.aggFlops}`);
  if (!near(s.ridge, s.peak / (g.bw * 1e12), 1e-9)) bad('ridge', c, `${s.ridge}`);
  if (s.native !== (g.flops[c.precision] != null)) bad('native-flag', c, `${s.native}`);

  // --- timing identities
  const floor = Math.max(s.tWeights + s.tKv, s.tCompute);
  if (s.tpot < floor - 1e-12) bad('tpot-floor', c, `tpot=${s.tpot} < max(mem,compute)=${floor}`);
  if (s.tpot < 1.2e-4 - 1e-12) bad('tpot-launch', c, `tpot=${s.tpot} below launch overhead`);
  if (!near(s.tokPerSec, 1 / s.tpot, 1e-9)) bad('tok-per-sec', c, `${s.tokPerSec}`);
  if (!near(s.sysTokPerSec, (c.batch / s.tpot) * s.util, 1e-9)) bad('sys-tok', c, `${s.sysTokPerSec}`);
  if (!near(s.perGpuTokPerSec, s.sysTokPerSec / s.nGpu, 1e-9)) bad('per-gpu-tok', c, `${s.perGpuTokPerSec}`);
  if (s.itlP99 < s.tpot - 1e-12) bad('itl-ge-tpot', c, `itl=${s.itlP99} tpot=${s.tpot}`);
  if (s.nGpu === 1 && s.tComm !== 0) bad('no-comm-single-gpu', c, `${s.tComm}`);
  if (!c.offload && s.tSsd !== 0) bad('no-ssd-when-off', c, `${s.tSsd}`);
  if (c.offload && s.spill === 0 && s.tSsd !== 0) bad('no-ssd-when-fits', c, `${s.tSsd}`);

  // --- roofline: the achieved point must sit under BOTH ceilings
  const perGpuBytes = s.tpot > 0 ? s.aiDecode > 0 ? (s.achDecode / s.aiDecode) : 0 : 0;
  if (s.achDecode > s.peak * (1 + 1e-9)) bad('under-compute-roof', c, `ach=${s.achDecode} peak=${s.peak}`);
  if (perGpuBytes > g.bw * 1e12 * (1 + 1e-9)) bad('under-bw-roof', c, `effBw=${perGpuBytes} spec=${g.bw * 1e12}`);
  if (s.video) {
    if (s.video.ach > s.peak * (1 + 1e-9)) bad('dit-under-compute-roof', c, `${s.video.ach}`);
    const vb = s.video.ai > 0 ? s.video.ach / s.video.ai : 0;
    if (vb > g.bw * 1e12 * (1 + 1e-9)) bad('dit-under-bw-roof', c, `${vb}`);
  }

  // --- MoE
  if (m.routed > 0) {
    const want = 1 - Math.pow(1 - m.topk / m.experts, c.batch);
    if (!near(s.moeFrac, want, 1e-9)) bad('moe-frac', c, `${s.moeFrac} vs ${want}`);
    if (s.moeFrac <= 0 || s.moeFrac > 1) bad('moe-frac-range', c, `${s.moeFrac}`);
  } else if (s.moeFrac !== 1) bad('moe-frac-dense', c, `${s.moeFrac}`);

  // --- KV sharding: head-parallel cannot split past the kv-head count; MLA runs data-parallel
  if (m.kind !== 'dit') {
    const wantShards = m.attn === 'mla' ? s.nGpu : Math.min(s.nGpu, m.kvHeads);
    if (s.kvShards !== wantShards) bad('kv-shards', c, `${s.kvShards} vs ${wantShards}`);
    if (s.kvShards > s.nGpu) bad('kv-shards-le-ngpu', c, `${s.kvShards} > ${s.nGpu}`);
    if (!near(s.tKv, s.kvBytes / s.kvShards / (g.bw * 1e12 * 0.8), 1e-9)) bad('tkv-from-shards', c, `${s.tKv}`);
  }

  // --- offload evicts weights first; the model must never stream more than the shortfall
  if (c.offload && s.spill > 0) {
    const streamed = s.tSsd - LINK.ssdLat;
    if (streamed <= 0) bad('ssd-positive', c, `${s.tSsd}`);
    const bytesStreamed = streamed * s.nGpu * LINK.ssdBw;
    if (bytesStreamed > s.spill * (1 + 1e-6) + s.weightBytes) bad('ssd-over-stream', c, `${bytesStreamed} vs spill ${s.spill}`);
  }

  // --- a denoise step all-reduces the whole clip, not one token
  if (s.video && s.nGpu > 1) {
    const want = (2 * (s.nGpu - 1) / s.nGpu) * c.batch * s.video.tokens * m.d * 2;
    if (!near(s.video.commBytes, want, 1e-9)) bad('dit-comm-bytes', c, `${s.video.commBytes} vs ${want}`);
    if (s.video.commBytes <= s.commBytes) bad('dit-comm-gt-decode', c, `${s.video.commBytes} <= ${s.commBytes}`);
    if (!near(s.video.perStep, Math.max(s.video.tCompute, s.video.tMem) + s.video.tComm, 1e-9))
      bad('dit-perstep-sum', c, `${s.video.perStep}`);
  }
  if (s.video && s.nGpu === 1 && (s.video.tComm !== 0 || s.video.commBytes !== 0)) bad('dit-comm-single', c, `${s.video.tComm}`);

  // --- KV per token by attention family
  if (m.kind !== 'dit') {
    const want = (m.attn === 'mla' ? m.mlaDim! : 2 * m.kvHeads * m.headDim) * m.layers * s.bytesPerWeight;
    if (!near(s.kvPerToken, want, 1e-9)) bad('kv-per-token', c, `${s.kvPerToken} vs ${want}`);
  }
  return s;
}

/* ---------------- sweep ---------------- */
const base: Cfg = {
  modelId: 'l70b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 4096,
  videoId: '480p5', steps: 50, gpuOverride: null, serving: 'sarathi', offload: false,
};
let n = 0;
for (const m of MODELS) for (const p of PRECISIONS) for (const g of GPUS)
  for (const b of BATCH_STEPS) for (const sv of SERVING) for (const off of [false, true])
    for (const ov of [null, 1, 8, 32] as (number | null)[]) {
      const ctx = CTX_STEPS.filter((x) => x <= m.maxCtx);
      for (const seq of [ctx[0], ctx[Math.floor(ctx.length / 2)], ctx[ctx.length - 1]]) {
        for (const vid of m.kind === 'dit' ? VIDEO_SHAPES.map((v) => v.id) : ['480p5']) {
          check({ ...base, modelId: m.id, precision: p.id, gpuId: g.id, batch: b, seq, serving: sv.id, offload: off, gpuOverride: ov, videoId: vid });
          n++;
        }
      }
    }

/* ---------------- monotonicity / scaling ---------------- */
const mono = (label: string, cfgs: Cfg[], get: (s: ReturnType<typeof simulate>) => number, dir: 'up' | 'down' | 'flat') => {
  const vs = cfgs.map((c) => get(simulate(c)));
  for (let i = 1; i < vs.length; i++) {
    const ok = dir === 'up' ? vs[i] >= vs[i - 1] * (1 - 1e-9)
      : dir === 'down' ? vs[i] <= vs[i - 1] * (1 + 1e-9)
      : near(vs[i], vs[i - 1], 1e-9);
    if (!ok) fails.push({ rule: `monotonic:${label}`, cfg: tag(cfgs[i]), detail: `${vs.map((v) => v.toPrecision(4)).join(' → ')}` });
  }
};

for (const m of MODELS.filter((x) => x.kind === 'llm')) {
  const c = (o: Partial<Cfg>): Cfg => ({ ...base, modelId: m.id, gpuOverride: 8, ...o });
  // Step time rises with batch for a dense model. It need not for a mixture of experts: once the
  // batch is wide enough to touch every expert, more tokens add almost no bytes but do fill the
  // per-expert GEMM tiles, so the step can get cheaper. Throughput is the invariant that holds.
  if (m.routed === 0) mono(`${m.id}:batch↑ tpot↑`, BATCH_STEPS.map((b) => c({ batch: b })), (s) => s.tpot, 'up');
  else mono(`${m.id}:batch↑ moeEff↑`, BATCH_STEPS.map((b) => c({ batch: b })), (s) => s.moeEff, 'up');
  mono(`${m.id}:batch↑ sysTok↑`, BATCH_STEPS.map((b) => c({ batch: b })), (s) => s.sysTokPerSec, 'up');
  mono(`${m.id}:ctx↑ kv↑`, CTX_STEPS.filter((x) => x <= m.maxCtx).map((seq) => c({ seq })), (s) => s.kvBytes, 'up');
  mono(`${m.id}:ctx↑ ttft↑`, CTX_STEPS.filter((x) => x <= m.maxCtx).map((seq) => c({ seq })), (s) => s.ttft, 'up');
  mono(`${m.id}:precision↓ weights↓`, (['bf16', 'fp8', 'fp4'] as const).map((precision) => c({ precision })), (s) => s.weightBytes, 'down');
  mono(`${m.id}:precision↓ kv↓`, (['bf16', 'fp8', 'fp4'] as const).map((precision) => c({ precision })), (s) => s.kvBytes, 'down');
  mono(`${m.id}:gpus↑ nAuto flat`, [1, 2, 4].map((batch) => c({ batch })), (s) => s.ridge, 'flat');
  // adding GPUs must never make the weight read slower
  mono(`${m.id}:gpus↑ tWeights↓`, [1, 2, 4, 8, 16, 32].map((gpuOverride) => c({ gpuOverride })), (s) => s.tWeights, 'down');
  // exact byte scaling with precision
  const [b16, b8, b4] = (['bf16', 'fp8', 'fp4'] as const).map((precision) => simulate(c({ precision })));
  if (!near(b16.weightBytes, b8.weightBytes * 2, 1e-9) || !near(b8.weightBytes, b4.weightBytes * 2, 1e-9))
    fails.push({ rule: 'precision-byte-scaling', cfg: m.id, detail: `${b16.weightBytes}/${b8.weightBytes}/${b4.weightBytes}` });
}

/* ---------------- reference table ----------------
   Printed, not asserted: `expect` is what real serving stacks MEASURE for that shape, and the
   simulator is an upper bound on it, so these columns are meant to differ. scripts/sim-verify.ts
   is where the relationship between the two is actually checked. */
const REF: { label: string; cfg: Partial<Cfg>; expect: string }[] = [
  { label: 'Llama 70B BF16, 1 seq, 4K', cfg: { modelId: 'l70b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 4096 }, expect: '2xH200 measures ~40-50 tok/s' },
  { label: 'Llama 70B FP8, 1 seq, 4K', cfg: { modelId: 'l70b', precision: 'fp8', gpuId: 'h200', batch: 1, seq: 4096 }, expect: '1xH200 measures ~45-55 tok/s' },
  { label: 'Llama 8B BF16, 1 seq, 4K', cfg: { modelId: 'l8b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 4096 }, expect: '1xH200 measures ~150-250 tok/s' },
  { label: 'Llama 405B FP8, 32 seq, 8K', cfg: { modelId: 'l405b', precision: 'fp8', gpuId: 'h200', batch: 32, seq: 8192 }, expect: '8xH200 measures ~25-35 tok/s per user' },
  { label: 'DeepSeek-V3 FP8, 64 seq, 4K', cfg: { modelId: 'dsv3', precision: 'fp8', gpuId: 'h200', batch: 64, seq: 4096 }, expect: '8xH200 measures ~15-25 tok/s per user' },
  { label: 'Wan2.1 14B BF16 480p/5s', cfg: { modelId: 'wan14b', precision: 'bf16', gpuId: 'h200', batch: 1, videoId: '480p5' }, expect: 'one H100/H200 measures ~4-10 min' },
  { label: 'Llama 8B FP16, 1 seq, 4K', cfg: { modelId: 'l8b', precision: 'bf16', gpuId: 'v100', batch: 1, seq: 4096 }, expect: 'bound ~45; V100 measures ~25-35' },
  { label: 'Llama 70B BF16, 1 seq, H100', cfg: { modelId: 'l70b', precision: 'bf16', gpuId: 'h100', batch: 1, seq: 4096 }, expect: 'bound ~75; TP=4 H100 measures ~35-45' },
  { label: 'Llama 70B FP8, 32 seq, H100', cfg: { modelId: 'l70b', precision: 'fp8', gpuId: 'h100', batch: 32, seq: 8192 }, expect: 'bound ~46; batched H100 measures ~30-40' },
  { label: 'Llama 70B BF16 TTFT 8K', cfg: { modelId: 'l70b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 8192 }, expect: 'TTFT ~1-2 s' },
];

if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ configs: n, fails, ref: REF.map((r) => ({ ...r, out: summarize(simulate({ ...base, ...r.cfg })) })) }, null, 1));
} else {
  console.log(`configs checked: ${n}`);
  console.log(`invariant failures: ${fails.length}`);
  const byRule = new Map<string, Fail[]>();
  for (const f of fails) { if (!byRule.has(f.rule)) byRule.set(f.rule, []); byRule.get(f.rule)!.push(f); }
  for (const [rule, list] of [...byRule].sort((a, b) => b[1].length - a[1].length)) {
    console.log(`\n  ✗ ${rule}  (${list.length})`);
    for (const f of list.slice(0, 4)) console.log(`      ${f.cfg}  →  ${f.detail}`);
    if (list.length > 4) console.log(`      … ${list.length - 4} more`);
  }
  console.log('\n--- reference configurations ---');
  for (const r of REF) {
    const s = simulate({ ...base, ...r.cfg });
    console.log(`  ${r.label.padEnd(32)} n=${String(s.nGpu).padStart(2)}  ${summarize(s).padEnd(46)} expect ${r.expect}`);
  }
}

function summarize(s: ReturnType<typeof simulate>): string {
  if (s.video) return `step=${s.video.perStep.toFixed(2)}s clip=${(s.video.total / 60).toFixed(1)}min ${s.bound}`;
  return `tpot=${(s.tpot * 1e3).toFixed(1)}ms ${(1 / s.tpot).toFixed(0)}tok/s ttft=${(s.ttft * 1e3).toFixed(0)}ms ${s.bound}`;
}
