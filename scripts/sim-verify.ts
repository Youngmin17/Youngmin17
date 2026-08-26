/* Ground-truth verification for the roofline simulator.
   Run: npx tsx scripts/sim-verify.ts

   sim-sweep.ts checks that the engine is consistent with itself: it re-derives each output from
   the same formula that produced it, so it passes for any set of constants, right or wrong. This
   file checks the other direction — that the constants and the formulas agree with something
   outside the repository:

     A  every GPU field against the published spec sheet for that part
     B  every parameter count against the arithmetic of the model's own architecture
     C  every video token count against the VAE compression ratio and the DiT patch size
     D  a decode step and a prefill recomputed by hand, from the method description, in one place
     E  headline outputs against measurements published for real serving stacks
     F  every derate the engine applies against the list the Method section shows the reader
*/
import { readFileSync } from 'node:fs';
import { BYTES, CLASSIC, GPUS, LINK, MODELS, PRECISIONS, VIDEO_SHAPES } from '../src/components/sim/data';
import { EMULATION_TAX, simulate, type Cfg } from '../src/components/sim/engine';

type Fail = { area: string; rule: string; detail: string };
const fails: Fail[] = [];
const bad = (area: string, rule: string, detail: string) => fails.push({ area, rule, detail });
const eq = (got: number, want: number, tol: number, area: string, rule: string, what: string) => {
  const off = Math.abs(got - want) / Math.max(1e-30, Math.abs(want));
  if (off > tol) bad(area, rule, `${what}: ${got} vs ${want} (${(off * 100).toFixed(1)}% off, tol ${(tol * 100).toFixed(0)}%)`);
};
const notes: string[] = [];

/* The configuration every check varies from. */
const BASE: Cfg = {
  modelId: 'l70b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 4096,
  videoId: '480p5', steps: 50, gpuOverride: null, serving: 'sarathi', offload: false,
};

/* ---------- A. GPU spec sheets -------------------------------------------------------------
   Transcribed from the vendor datasheet for each part. Tensor-core rates are the DENSE numbers;
   the "with sparsity" figure is twice these and is the one marketing quotes. */
const SPEC: Record<string, { hbm: number; bw: number; bf16: number; fp8: number | null; fp4: number | null; nvlink: number; year: number; mem: string }> = {
  v100: { hbm: 32, bw: 0.9, bf16: 125, fp8: null, fp4: null, nvlink: 300, year: 2017, mem: 'HBM2' },
  a100: { hbm: 80, bw: 2.039, bf16: 312, fp8: null, fp4: null, nvlink: 600, year: 2020, mem: 'HBM2e' },
  h100: { hbm: 80, bw: 3.35, bf16: 989, fp8: 1979, fp4: null, nvlink: 900, year: 2022, mem: 'HBM3' },
  h200: { hbm: 141, bw: 4.8, bf16: 989, fp8: 1979, fp4: null, nvlink: 900, year: 2024, mem: 'HBM3e' },
  b200: { hbm: 180, bw: 8.0, bf16: 2250, fp8: 4500, fp4: 9000, nvlink: 1800, year: 2025, mem: 'HBM3e' },
};
for (const g of GPUS) {
  const s = SPEC[g.id];
  if (!s) { bad('gpu', 'part-not-in-spec-table', g.id); continue; }
  eq(g.hbm, s.hbm, 0.001, 'gpu', 'hbm', g.id);
  eq(g.bw, s.bw, 0.005, 'gpu', 'bandwidth', g.id);
  eq(g.nvlink, s.nvlink, 0.001, 'gpu', 'nvlink', g.id);
  if (g.year !== s.year) bad('gpu', 'year', `${g.id}: ${g.year} vs ${s.year}`);
  if (g.mem !== s.mem) bad('gpu', 'memory-type', `${g.id}: ${g.mem} vs ${s.mem}`);
  for (const p of ['bf16', 'fp8', 'fp4'] as const) {
    const got = g.flops[p], want = s[p];
    if ((got == null) !== (want == null)) bad('gpu', 'tensor-core-presence', `${g.id} ${p}: ${got} vs ${want}`);
    else if (got != null && want != null) eq(got, want, 0.005, 'gpu', 'tensor-core-rate', `${g.id} ${p}`);
  }
  // A part with no bfloat16 in its ISA occupies the 16-bit slot but must not be labelled BF16.
  if (g.id === 'v100' && g.half !== 'FP16') bad('gpu', 'volta-half-label', 'V100 tensor cores are FP16, not BF16');
}
// 400 Gb/s NDR is 50 GB/s; a Gen5 x4 NVMe reads ~14 GB/s sequentially.
eq(LINK.ibBw, 400e9 / 8, 0.001, 'link', 'infiniband-ndr', '400 Gb/s per GPU');
eq(LINK.ssdBw, 14e9, 0.001, 'link', 'nvme-gen5', 'sequential read');
// QD1 4K random read on NAND is tens of microseconds — the media, not the bus. Anything under
// 20 us is 3D XPoint, and quoting it as a Gen5 NVMe figure overstates how much storage improved.
if (LINK.ssdLat < 40e-6 || LINK.ssdLat > 120e-6) bad('link', 'nvme-latency-not-nand', `${LINK.ssdLat * 1e6} us is outside the 40-120 us NAND band`);

/* ---------- B. parameter counts from the architecture ----------------------------------------
   Recomputed from each model's own config: embeddings + per-layer attention and MLP + norms.
   Nothing here reads MODELS.params, so a wrong param count cannot hide behind a wrong layer count. */
const CONFIG: Record<string, { vocab: number; ffn: number; tied: boolean }> = {
  l8b: { vocab: 128256, ffn: 14336, tied: false },
  l13b: { vocab: 32000, ffn: 13824, tied: false },
  l70b: { vocab: 128256, ffn: 28672, tied: false },
  l405b: { vocab: 128256, ffn: 53248, tied: false },
};
for (const m of MODELS) {
  const c = CONFIG[m.id];
  if (!c) continue;                     // MoE and DiT counts are published totals, checked below
  const attn = m.d * m.heads * m.headDim          // q
    + 2 * m.d * m.kvHeads * m.headDim             // k, v
    + m.heads * m.headDim * m.d;                  // o
  const mlp = 3 * m.d * c.ffn;                    // gate, up, down — SwiGLU
  const perLayer = attn + mlp + 2 * m.d;          // two RMSNorms
  const derived = c.vocab * m.d * (c.tied ? 1 : 2) + m.layers * perLayer + m.d;
  eq(m.params, derived, 0.01, 'model', 'param-count', `${m.id} (${m.layers}L d=${m.d} ffn=${c.ffn})`);
}
// DeepSeek-V3 publishes 671B total and 37B activated per token; the engine has to reproduce both.
{
  const m = MODELS.find((x) => x.id === 'dsv3')!;
  eq(m.dense + m.routed, 671e9, 0.005, 'model', 'dsv3-total', 'dense + routed');
  eq(m.dense + m.routed * (m.topk / m.experts), 37e9, 0.03, 'model', 'dsv3-activated', 'per-token active params');
  // MLA caches kv_lora_rank 512 plus qk_rope_head_dim 64.
  eq(m.mlaDim!, 512 + 64, 0.001, 'model', 'mla-latent-width', 'kv_lora_rank + rope dim');
}
// KV bytes per token, straight from the attention shape.
for (const m of MODELS.filter((x) => x.kind === 'llm')) {
  const s = simulate({ ...BASE, modelId: m.id, precision: 'bf16', seq: 1024, batch: 1 });
  const want = (m.attn === 'mla' ? m.mlaDim! : 2 * m.kvHeads * m.headDim) * m.layers * 2;
  eq(s.kvPerToken, want, 1e-9, 'model', 'kv-per-token', m.id);
}

/* ---------- C. video token geometry ----------------------------------------------------------
   Wan's VAE compresses 4x in time (with the first frame kept whole) and 8x in each spatial axis;
   the DiT then patchifies at (1, 2, 2). Token count follows from the pixel shape and nothing else. */
for (const v of VIDEO_SHAPES) {
  const [w, h] = v.px.split('x').map(Number);
  const t = (v.frames - 1) / 4 + 1;
  const derived = t * (h / 8 / 2) * (w / 8 / 2);
  if (!Number.isInteger(t)) bad('video', 'frame-count-not-4n1', `${v.id}: ${v.frames} frames`);
  eq(v.tokens, derived, 1e-9, 'video', 'token-count', `${v.id} ${v.px} x ${v.frames}f`);
}

/* ---------- D. a step recomputed by hand -----------------------------------------------------
   One decode step and one prefill, written out here from the Method paragraph rather than from
   engine.ts, so that a refactor that changes the arithmetic has to be noticed. */
function byHand(cfg: Cfg) {
  const m = MODELS.find((x) => x.id === cfg.modelId)!;
  const g = GPUS.find((x) => x.id === cfg.gpuId)!;
  const bpw = BYTES[cfg.precision];
  const n = cfg.gpuOverride!;
  const bw = g.bw * 1e12 * 0.8;                       // 80% of the spec sheet
  const peak = g.flops[cfg.precision]! * 1e12;
  const dAttn = m.heads * m.headDim;

  const tWeights = m.params * bpw / (n * bw);
  const kvPerTok = 2 * m.kvHeads * m.headDim * m.layers * bpw;
  const shards = Math.min(n, m.kvHeads);
  const tKv = cfg.batch * cfg.seq * kvPerTok / shards / bw;
  const flops = 2 * m.params * cfg.batch + 4 * m.layers * cfg.batch * cfg.seq * dAttn;
  const tCompute = flops / (n * peak * 0.75);
  const linkBw = n > g.perNode ? LINK.ibBw : g.nvlink / 2 * 1e9;
  const linkLat = n > g.perNode ? LINK.ibLat : LINK.nvLat;
  const tComm = n > 1 ? m.layers * 2 * ((2 * (n - 1) / n) * cfg.batch * m.d * 2 / linkBw + linkLat) : 0;
  const tpot = Math.max(tWeights + tKv, tCompute) + tComm + 1.2e-4;

  const preFlops = 2 * m.params * cfg.seq + 2 * m.layers * cfg.seq * cfg.seq * dAttn;
  const preCompute = preFlops / (n * peak * 0.55);
  const preMem = (m.params * bpw / n + cfg.seq * kvPerTok / shards) / bw;
  const preComm = n > 1
    ? m.layers * 2 * ((2 * (n - 1) / n) * cfg.seq * m.d * 2 / linkBw + linkLat) : 0;
  return { tpot, ttft: Math.max(preCompute, preMem) + preComm, tWeights, tKv, tCompute, tComm };
}
const HAND: Cfg[] = [
  { ...BASE, gpuOverride: 2 },
  { ...BASE, gpuOverride: 8, batch: 32, seq: 8192, precision: 'fp8' as const },
  { ...BASE, modelId: 'l8b', gpuOverride: 1, batch: 4, seq: 16384 },
  { ...BASE, modelId: 'l13b', gpuOverride: 4, batch: 16, seq: 4096, gpuId: 'a100' },
  // past the kv-head count and off the node: exercises the sharding ceiling and the InfiniBand leg
  { ...BASE, gpuOverride: 32, batch: 64, seq: 32768 },
];
for (const cfg of HAND) {
  const s = simulate(cfg), h = byHand(cfg);
  const at = `${cfg.modelId}/${cfg.precision}/${cfg.gpuId}/b${cfg.batch}/s${cfg.seq}/n${cfg.gpuOverride}`;
  for (const k of ['tWeights', 'tKv', 'tCompute', 'tComm', 'tpot'] as const) eq((s as any)[k], (h as any)[k], 1e-9, 'hand', k, at);
  eq(s.ttft, h.ttft * (cfg.serving === 'sarathi' ? 1.15 : 1), 1e-9, 'hand', 'ttft', at);
}

/* ---------- E. against published measurements ------------------------------------------------
   The simulator assumes a perfect kernel, a perfect scheduler and no Python, so it generally reads
   ABOVE a real stack. It is not a strict upper bound, though, and claiming one here would be false:
   the mixture-of-experts bandwidth floor is pessimistic at large batch, which is why DeepSeek-V3
   comes out near the bottom of its measured range, while the moeFrac term makes batch-1 MoE
   optimistic — as the page says. So the assertion is the one the page actually makes to the
   reader — "within about a factor of two of published measurements" — applied on both sides.

   `lo`/`hi` are the range published for vLLM- and TensorRT-LLM-class serving of that shape. */
const TOLERANCE = 2;
const MEASURED: { what: string; kind: 'rate' | 'time'; cfg: Partial<Cfg>; lo: number; hi: number; get: (s: ReturnType<typeof simulate>) => number }[] = [
  { what: 'Llama 8B BF16 b=1 4K, 1xH100 — decode tok/s', kind: 'rate', lo: 90, hi: 150,
    cfg: { modelId: 'l8b', precision: 'bf16', gpuId: 'h100', batch: 1, seq: 4096, gpuOverride: 1 }, get: (s) => 1 / s.tpot },
  { what: 'Llama 70B BF16 b=1 4K, 4xH100 — decode tok/s', kind: 'rate', lo: 35, hi: 45,
    cfg: { modelId: 'l70b', precision: 'bf16', gpuId: 'h100', batch: 1, seq: 4096, gpuOverride: 4 }, get: (s) => 1 / s.tpot },
  { what: 'Llama 70B FP8 b=32 8K, 2xH100 — decode tok/s per user', kind: 'rate', lo: 30, hi: 40,
    cfg: { modelId: 'l70b', precision: 'fp8', gpuId: 'h100', batch: 32, seq: 8192, gpuOverride: 2 }, get: (s) => 1 / s.tpot },
  { what: 'Llama 405B FP8 b=32 8K, 8xH200 — decode tok/s per user', kind: 'rate', lo: 25, hi: 35,
    cfg: { modelId: 'l405b', precision: 'fp8', gpuId: 'h200', batch: 32, seq: 8192, gpuOverride: 8 }, get: (s) => 1 / s.tpot },
  { what: 'DeepSeek-V3 FP8 b=64 4K, 8xH200 — decode tok/s per user', kind: 'rate', lo: 15, hi: 25,
    cfg: { modelId: 'dsv3', precision: 'fp8', gpuId: 'h200', batch: 64, seq: 4096, gpuOverride: 8 }, get: (s) => 1 / s.tpot },
  { what: 'Llama 70B BF16 8K prefill, 2xH200 — TTFT seconds', kind: 'time', lo: 1, hi: 2,
    cfg: { modelId: 'l70b', precision: 'bf16', gpuId: 'h200', batch: 1, seq: 8192, gpuOverride: 2 }, get: (s) => s.ttft },
  { what: 'Wan2.1 14B 480p/5s, 50 steps, 1xH200 — clip minutes', kind: 'time', lo: 4, hi: 10,
    cfg: { modelId: 'wan14b', precision: 'bf16', gpuId: 'h200', batch: 1, gpuOverride: 1, videoId: '480p5' }, get: (s) => s.video!.total / 60 },
];
for (const r of MEASURED) {
  const v = r.get(simulate({ ...BASE, ...r.cfg }));
  const floor = r.lo / TOLERANCE, ceil = r.hi * TOLERANCE;
  if (v < floor || v > ceil) {
    const off = v < floor ? r.lo / v : v / r.hi;
    bad('measured', 'outside-2x-of-measurement',
      `${r.what}: ${v.toFixed(2)} vs measured ${r.lo}–${r.hi} (${off.toFixed(1)}x outside)`);
  }
}

/* ---------- F. every derate is disclosed ------------------------------------------------------
   A constant that silently scales a headline number is the one a reader most needs to see. */
const engineSrc = readFileSync(new URL('../src/components/sim/engine.ts', import.meta.url), 'utf8');
const methodSrc = readFileSync(new URL('../src/pages/simulator.astro', import.meta.url), 'utf8');
const DERATES: [string, RegExp, RegExp][] = [
  ['BW_EFF 0.8', /BW_EFF = 0\.8\b/, /80% of the spec sheet/],
  ['HBM_USABLE 0.9', /HBM_USABLE = 0\.9\b/, /90% of capacity/],
  ['MFU 0.55 / 0.75', /MFU_PREFILL = 0\.55[\s\S]*MFU_DECODE = 0\.75/, /55% prefill, 75% decode/],
  ['LAUNCH 120 µs', /LAUNCH = 1\.2e-4\b/, /120 µs per step/],
  ['CHUNK 512', /CHUNK = 512\b/, /512-token chunks/],
  ['MOE_FLOOR 0.3', /MOE_FLOOR = 0\.3\b/, /floor 30%/],
  ['duty cycle', /util = 0\.45[\s\S]*util = 0\.92/, /45% static, 85% continuous, 90% chunked, 92% split/],
  ['NVMe 14 GB/s', /ssdBw: 14e9/, /14 GB\/s per GPU/],
  ['chunked-prefill TTFT +15%', /ttft = ttftRaw \* 1\.15/, /\+15% for the chunking itself/],
  ['static-batching queue', /ttft = ttftRaw \+ tpot \* 64/, /one batch of 64 steps ahead of you/],
  ['activations + workspace', /0\.06 \* \(model\.params \* 2\) \+ 1\.5e9/, /6% of the BF16 weights, plus 1\.5 GB fixed/],
  ['emulation tax 0.85', /EMULATION_TAX = 0\.85\b/, /85% of the part's 16-bit rate/],
  ['DiT activations', /batch \* seq \* model\.d \* 2 \* 8/, /eight residual-sized tensors per layer/],
];
for (const [name, inEngine, onPage] of DERATES) {
  if (!inEngine.test(engineSrc) && !inEngine.test(readFileSync(new URL('../src/components/sim/data.ts', import.meta.url), 'utf8'))) {
    bad('disclosure', 'derate-moved', `${name} no longer matches the engine — the Method entry may now be stale`);
  } else if (!onPage.test(methodSrc)) {
    bad('disclosure', 'derate-undisclosed', `${name} scales the output but is not in the Method list`);
  }
}
// The format picker will not offer a format the SELECTED part lacks, but the roofline draws the
// other four parts too, and any of those that lacks the format gets its ghost roof taxed. So the
// derate is on screen, and the plot must apply the same constant the engine does rather than its
// own copy of it.
{
  const plot = readFileSync(new URL('../src/components/sim/Simulator.tsx', import.meta.url), 'utf8');
  const emulatedRoofs = GPUS.some((g) => PRECISIONS.some((p) => g.flops[p.id] == null));
  if (emulatedRoofs && !/\* \(nat \? 1 : EMULATION_TAX\)/.test(plot)) {
    bad('disclosure', 'emulation-tax-duplicated', 'the roofline taxes ghost roofs with its own literal instead of EMULATION_TAX');
  }
  if (!emulatedRoofs) notes.push('every part now has every format — no ghost roof is emulated.');
  else notes.push(`EMULATION_TAX ${EMULATION_TAX} reaches the reader through the ghost roofs, and is listed under Method.`);
}

/* ---------- G. the latency ladder is what it says it is ---------------------------------------
   These fourteen rows are the classic block of Jonas Bonér's gist 2841832, fetched from
   https://gist.githubusercontent.com/jboner/2841832/raw/latency.txt — whose own header reads
   "Latency Comparison Numbers (~2012)". They are NOT Jeff Dean's 2010 slide, which has twelve rows,
   "Send 2K bytes over 1 Gbps network 20,000 ns", and no SSD rows at all. The page says ~2012 for
   that reason; if the column ever drifts to present-day values the comparison stops meaning
   anything, and if the label drifts back to 2010 it is simply false. */
const CANON_GIST_2012: [string, number][] = [
  ['L1 cache reference', 0.5], ['Branch mispredict', 5], ['L2 cache reference', 7],
  ['Mutex lock/unlock', 25], ['Main memory reference', 100], ['Compress 1K bytes with Zippy', 3_000],
  ['Send 1K bytes over 1 Gbps network', 10_000], ['Read 4K randomly from SSD', 150_000],
  ['Read 1 MB sequentially from memory', 250_000], ['Round trip within same datacenter', 500_000],
  ['Read 1 MB sequentially from SSD', 1_000_000], ['Disk seek', 10_000_000],
  ['Read 1 MB sequentially from disk', 20_000_000], ['Send packet CA→Netherlands→CA', 150_000_000],
];
for (const [label, ns] of CANON_GIST_2012) {
  const row = CLASSIC.find((r) => r.label === label);
  if (!row) bad('ladder', 'row-missing', label);
  else eq(row.ns, ns, 1e-9, 'ladder', 'not-the-2012-gist-value', label);
}
for (const r of CLASSIC) if (!CANON_GIST_2012.some(([l]) => l === r.label)) bad('ladder', 'row-unlisted', r.label);
/* And the copy must not call it the 2010 table again. */
{
  const plot = readFileSync(new URL('../src/components/sim/Simulator.tsx', import.meta.url), 'utf8');
  const data = readFileSync(new URL('../src/components/sim/data.ts', import.meta.url), 'utf8');
  for (const [where, src] of [['Simulator.tsx', plot], ['data.ts', data]] as const) {
    const claim = src.match(/[^\n]*\b2010\b[^\n]*/g) ?? [];
    for (const line of claim) {
      if (!/not Dean's own 2010 slide|NOT Jeff Dean's 2010|twelve rows/i.test(line)) {
        bad('ladder', 'misattributed-to-2010', `${where}: ${line.trim().slice(0, 90)}`);
      }
    }
  }
}

/* The ladder sentence divides one table by the other, looking rows up by their exact label. A
   renamed row makes gain() return 0 and the page renders "0x faster" rather than failing. */
{
  const plot = readFileSync(new URL('../src/components/sim/Simulator.tsx', import.meta.url), 'utf8');
  for (const m of plot.matchAll(/gain\('([^']+)',\s*'([^']+)'\)/g)) {
    const [, classic, modern] = m;
    if (!CLASSIC.some((r) => r.label === classic)) bad('ladder', 'gain-label-unresolved', `CLASSIC has no "${classic}"`);
    const modernRow = plot.match(new RegExp(`label: '${modern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}'`));
    if (!modernRow) bad('ladder', 'gain-label-unresolved', `MODERN has no "${modern}"`);
  }
}

/* ---------- report --------------------------------------------------------------------------- */
console.log(`GPUs verified against spec sheets : ${GPUS.length}`);
console.log(`models with derivable param counts: ${MODELS.filter((m) => CONFIG[m.id]).length} of ${MODELS.length}`);
console.log(`video shapes re-derived           : ${VIDEO_SHAPES.length}`);
console.log(`configs recomputed by hand        : ${HAND.length}`);
console.log(`published bands checked           : ${MEASURED.length}`);
console.log(`failures                          : ${fails.length}`);
const byRule = new Map<string, Fail[]>();
for (const f of fails) { if (!byRule.has(f.rule)) byRule.set(f.rule, []); byRule.get(f.rule)!.push(f); }
for (const [rule, list] of [...byRule].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`\n  x ${rule}  (${list.length})`);
  for (const f of list.slice(0, 6)) console.log(`      ${f.area}: ${f.detail}`);
  if (list.length > 6) console.log(`      … ${list.length - 6} more`);
}
if (notes.length) { console.log(''); for (const n of notes) console.log(`  note: ${n}`); }
if (!fails.length) console.log('\nEvery constant traces to a spec sheet, an architecture, or a published measurement.');
process.exit(fails.length ? 1 : 0);
