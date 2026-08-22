import {
  BYTES, GPUS, LINK, MODELS, VIDEO_SHAPES,
  type Model, type Gpu, type Precision, type Serving,
} from './data';

export interface Cfg {
  modelId: string;
  precision: Precision;
  gpuId: string;
  batch: number;
  seq: number;
  videoId: string;
  steps: number;
  gpuOverride: number | null;
  serving: Serving;
  offload: boolean;
}

/* Utilization ceilings. Decode never gets near peak FLOP/s and is not supposed to —
   it is bandwidth-bound, and max(memory, compute) picks the real limiter. */
const MFU_PREFILL = 0.55;
const MFU_DECODE = 0.75;
const BW_EFF = 0.8;    // a good decode kernel reads HBM at ~80% of the spec sheet
const MOE_FLOOR = 0.3; // scattered per-expert GEMMs at tiny batch, before they fill up
const EMULATION_TAX = 0.85; // dequantize-in-kernel when the format has no native path
const HBM_USABLE = 0.9;
const LAUNCH = 1.2e-4; // one CUDA-graph replay plus scheduler, per step
const CHUNK = 512;

const pow2Ceil = (x: number) => Math.pow(2, Math.ceil(Math.log2(Math.max(1, x))));

export interface Sim {
  model: Model; gpu: Gpu; native: boolean; peak: number; bytesPerWeight: number;
  nGpu: number; nAuto: number; crossNode: boolean; nodes: number;
  aggFlops: number; aggBw: number; aggHbm: number;
  weightBytes: number; kvBytes: number; kvPerToken: number; actBytes: number;
  memNeed: number; capacity: number; spill: number; fits: boolean;
  moeFrac: number;
  tWeights: number; tKv: number; tCompute: number; tComm: number; tSsd: number;
  tpot: number; tokPerSec: number; sysTokPerSec: number; perGpuTokPerSec: number;
  ttft: number; itlP99: number; kvXfer: number; util: number;
  aiDecode: number; aiPrefill: number; achDecode: number; achPrefill: number; ridge: number;
  bound: 'memory' | 'compute' | 'interconnect' | 'storage';
  moeEff: number;
  commBytes: number; prefillTokPerSec: number;
  video: {
    tokens: number; label: string; px: string; perStep: number; total: number;
    ai: number; ach: number; tCompute: number; tMem: number;
  } | null;
}

export function simulate(cfg: Cfg): Sim {
  const model = MODELS.find((m) => m.id === cfg.modelId)!;
  const gpu = GPUS.find((g) => g.id === cfg.gpuId)!;
  const isDit = model.kind === 'dit';
  const shape = VIDEO_SHAPES.find((v) => v.id === cfg.videoId)!;

  const bpw = BYTES[cfg.precision];
  const native = gpu.flops[cfg.precision] != null;
  const peak = (gpu.flops[cfg.precision] ?? gpu.flops.bf16!) * 1e12 * (native ? 1 : EMULATION_TAX);
  const bw = gpu.bw * 1e12;

  const seq = isDit ? shape.tokens : cfg.seq;
  const batch = cfg.batch;
  const dAttn = model.heads * model.headDim;

  /* ---- footprint -------------------------------------------------------- */
  const weightBytes = model.params * bpw;
  // MLA caches one latent vector plus a shared rope key; MHA/GQA cache K and V per kv-head.
  const kvPerToken = isDit
    ? 0
    : (model.attn === 'mla' ? model.mlaDim! : 2 * model.kvHeads * model.headDim) * model.layers * bpw;
  const kvBytes = batch * seq * kvPerToken;
  const actBytes = isDit
    ? batch * seq * model.d * 2 * 8
    : 0.06 * weightBytes + 1.5e9;
  const memNeed = weightBytes + kvBytes + actBytes;

  const usable = gpu.hbm * 1e9 * HBM_USABLE;
  const nAuto = Math.min(1024, pow2Ceil(memNeed / usable));
  const nGpu = cfg.gpuOverride ?? nAuto;
  const capacity = nGpu * usable;
  const spill = Math.max(0, memNeed - capacity);
  const fits = spill <= 0;

  const nodes = Math.ceil(nGpu / gpu.perNode);
  const crossNode = nGpu > gpu.perNode;
  const linkBw = crossNode ? LINK.ibBw : (gpu.nvlink / 2) * 1e9;
  const linkLat = crossNode ? LINK.ibLat : LINK.nvLat;

  /* ---- decode step ------------------------------------------------------ */
  // An expert is read if any of the batch's tokens routes to it.
  const moeFrac = model.routed > 0 ? 1 - Math.pow(1 - model.topk / model.experts, batch) : 1;
  const wRead = (model.dense + model.routed * moeFrac) * bpw;
  // K/V shards with tensor parallelism because it is head-parallel; the MLA latent is
  // replicated on every rank, which is why DeepSeek runs attention data-parallel instead.
  const kvRead = model.attn === 'mla' ? kvBytes : kvBytes / nGpu;

  const spillShare = memNeed > 0 ? spill / memNeed : 0;
  const tSsd = cfg.offload && spill > 0
    ? (wRead * spillShare) / (nGpu * LINK.ssdBw) + LINK.ssdLat
    : 0;
  // Each expert sees batch * topk / experts tokens; below a full tile the GEMM is inefficient.
  const perExpert = model.routed > 0 ? (batch * model.topk) / model.experts : Infinity;
  const moeEff = model.routed > 0
    ? Math.min(1, MOE_FLOOR + (1 - MOE_FLOOR) * Math.min(1, perExpert / 96))
    : 1;
  const effBw = bw * BW_EFF * moeEff;
  const tWeights = (wRead * (1 - (cfg.offload ? spillShare : 0))) / (nGpu * effBw);
  const tKv = kvRead / (bw * BW_EFF);

  const activeParams = model.routed > 0
    ? model.dense + model.routed * (model.topk / model.experts)
    : model.dense;
  const flopsDecode = 2 * activeParams * batch + 4 * model.layers * batch * seq * dAttn;
  const tCompute = flopsDecode / (nGpu * peak * MFU_DECODE);

  // Two all-reduces per layer, one after attention and one after the MLP.
  const commBytes = nGpu > 1 ? (2 * (nGpu - 1) / nGpu) * batch * model.d * 2 : 0;
  const a2a = model.routed > 0 && nGpu > 1
    ? model.layers * 2 * (batch * model.d * 2 * model.topk / linkBw + linkLat)
    : 0;
  const tComm = (nGpu > 1 ? model.layers * 2 * (commBytes / linkBw + linkLat) : 0) + a2a;

  const tMem = tWeights + tKv;
  const tpot = Math.max(tMem, tCompute) + tComm + LAUNCH + tSsd;

  /* ---- prefill ---------------------------------------------------------- */
  // One arriving request, not the whole batch: under continuous batching a new sequence is
  // prefilled on its own (or in chunks) while everything already resident keeps decoding.
  const prefillFlops = 2 * activeParams * seq
    + 2 * model.layers * seq * seq * dAttn; // causal, so half the full attention matrix
  const tPreCompute = prefillFlops / (nGpu * peak * MFU_PREFILL);
  const tPreMem = (wRead / nGpu + kvBytes / Math.max(1, batch) / nGpu) / (bw * BW_EFF);
  const preComm = nGpu > 1
    ? model.layers * 2 * ((2 * (nGpu - 1) / nGpu) * seq * model.d * 2 / linkBw + linkLat)
    : 0;
  const ttftRaw = Math.max(tPreCompute, tPreMem) + preComm;
  const chunkFrac = Math.min(1, CHUNK / Math.max(1, seq));
  const tChunk = ttftRaw * chunkFrac;
  const kvXfer = (kvBytes / Math.max(1, batch)) / (crossNode ? LINK.ibBw : (gpu.nvlink / 2) * 1e9);

  /* ---- serving discipline ----------------------------------------------- */
  let util = 0.45, ttft = ttftRaw, itlP99 = tpot;
  if (cfg.serving === 'orca') { util = 0.85; itlP99 = tpot + ttftRaw; }
  else if (cfg.serving === 'sarathi') { util = 0.9; itlP99 = tpot + tChunk; ttft = ttftRaw * 1.15; }
  else if (cfg.serving === 'pd') { util = 0.92; itlP99 = tpot; ttft = ttftRaw + kvXfer; }
  else { ttft = ttftRaw + tpot * 64; } // static batching queues behind the running batch

  /* ---- roofline --------------------------------------------------------- */
  const decodeBytes = wRead / nGpu + kvRead;
  const aiDecode = decodeBytes > 0 ? (flopsDecode / nGpu) / decodeBytes : 0;
  const prefillBytes = wRead / nGpu + kvBytes / Math.max(1, batch) / nGpu;
  const aiPrefill = prefillBytes > 0 ? (prefillFlops / nGpu) / prefillBytes : 0;
  const achDecode = flopsDecode / tpot / nGpu;
  const achPrefill = prefillFlops / Math.max(ttftRaw, 1e-9) / nGpu;
  const ridge = peak / bw;
  const moeEfficiency = moeEff;

  let bound: Sim['bound'] = tMem >= tCompute ? 'memory' : 'compute';
  if (tComm > Math.max(tMem, tCompute)) bound = 'interconnect';
  if (tSsd > Math.max(tMem, tCompute)) bound = 'storage';

  /* ---- video ------------------------------------------------------------ */
  let video: Sim['video'] = null;
  if (isDit) {
    const stepFlops = 2 * model.params * seq * batch
      + 4 * model.layers * batch * seq * seq * dAttn; // bidirectional, so the full matrix
    const stepMem = (weightBytes + actBytes) / (nGpu * bw * BW_EFF);
    const stepCompute = stepFlops / (nGpu * peak * MFU_PREFILL);
    const perStep = Math.max(stepCompute, stepMem) + tComm;
    video = {
      tokens: seq, label: shape.label, px: shape.px, perStep,
      total: perStep * cfg.steps * 2,
      ai: (stepFlops / nGpu) / ((weightBytes + actBytes) / nGpu),
      ach: stepFlops / perStep / nGpu,
      tCompute: stepCompute, tMem: stepMem,
    };
    // A denoise step is its own workload; the decode-path limiter says nothing about it.
    bound = stepCompute >= stepMem ? 'compute' : 'memory';
    if (tComm > Math.max(stepCompute, stepMem)) bound = 'interconnect';
  }

  const tokPerSec = 1 / tpot;
  const sysTokPerSec = (batch / tpot) * util;

  return {
    model, gpu, native, peak, bytesPerWeight: bpw,
    nGpu, nAuto, crossNode, nodes,
    aggFlops: nGpu * peak, aggBw: nGpu * bw, aggHbm: nGpu * gpu.hbm,
    weightBytes, kvBytes, kvPerToken, actBytes, memNeed, capacity, spill, fits,
    moeFrac,
    tWeights, tKv, tCompute, tComm, tSsd,
    tpot, tokPerSec, sysTokPerSec, perGpuTokPerSec: sysTokPerSec / nGpu,
    ttft, itlP99, kvXfer, util,
    aiDecode, aiPrefill, achDecode, achPrefill, ridge,
    bound, moeEff: moeEfficiency, commBytes, prefillTokPerSec: seq / Math.max(ttftRaw, 1e-9),
    video,
  };
}

/* ---- formatting --------------------------------------------------------- */
export const bytes = (b: number): string => {
  if (!isFinite(b) || b <= 0) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log10(b) / 3));
  const v = b / Math.pow(1000, i);
  return `${v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
};

export const secs = (s: number): string => {
  if (!isFinite(s) || s <= 0) return '—';
  if (s < 1e-6) return `${(s * 1e9).toFixed(0)} ns`;
  if (s < 1e-3) return `${(s * 1e6).toFixed(s * 1e6 < 10 ? 1 : 0)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(s * 1e3 < 10 ? 2 : 1)} ms`;
  if (s < 90) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  if (s < 5400) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${(s / 3600).toFixed(1)} h`;
};

export const num = (n: number, d = 0): string =>
  isFinite(n) ? n.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }) : '—';

export const compact = (n: number): string => {
  if (!isFinite(n)) return '—';
  if (n >= 1e18) return `${(n / 1e18).toFixed(2)}E`;
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)}P`;
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n / 1e9 < 10 ? 2 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(n >= 10 ? 0 : n >= 1 ? 1 : 2);
};
