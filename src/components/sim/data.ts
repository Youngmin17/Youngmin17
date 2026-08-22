/* Hardware, model, and interconnect constants for the toy roofline simulator.
   Every number here is a published spec or a straightforward derivation from one.
   Dense (non-sparse) tensor-core rates are used throughout. */

export type Precision = 'bf16' | 'fp8' | 'fp4';

export const BYTES: Record<Precision, number> = { bf16: 2, fp8: 1, fp4: 0.5 };

export const PRECISIONS: { id: Precision; label: string; note: string }[] = [
  { id: 'bf16', label: 'BF16', note: '2 bytes / weight' },
  { id: 'fp8', label: 'FP8', note: '1 byte / weight' },
  { id: 'fp4', label: 'FP4', note: '0.5 bytes / weight' },
];

export interface Gpu {
  id: string;
  name: string;
  short: string;
  hbm: number; // GB
  bw: number; // TB/s
  flops: Record<Precision, number | null>; // dense TFLOP/s, null = no native path
  nvlink: number; // GB/s, bidirectional, per GPU
  perNode: number;
  year: number;
  mem: string;
}

export const GPUS: Gpu[] = [
  {
    id: 'a100', name: 'A100 80GB SXM', short: 'A100', hbm: 80, bw: 2.039,
    flops: { bf16: 312, fp8: null, fp4: null },
    nvlink: 600, perNode: 8, year: 2020, mem: 'HBM2e',
  },
  {
    id: 'h200', name: 'H200 SXM', short: 'H200', hbm: 141, bw: 4.8,
    flops: { bf16: 989, fp8: 1979, fp4: null },
    nvlink: 900, perNode: 8, year: 2024, mem: 'HBM3e',
  },
  {
    id: 'b200', name: 'B200 SXM', short: 'B200', hbm: 192, bw: 8.0,
    flops: { bf16: 2250, fp8: 4500, fp4: 9000 },
    nvlink: 1800, perNode: 8, year: 2025, mem: 'HBM3e',
  },
];

/* Interconnect and storage, in bytes/s and seconds. */
export const LINK = {
  ibBw: 50e9,        // NDR InfiniBand, 400 Gb/s per GPU
  ibLat: 9e-6,
  nvLat: 3e-6,
  ssdBw: 14e9,       // one PCIe Gen5 x4 NVMe drive, sequential read
  ssdLat: 16e-6,     // random read — unchanged since 2015
};

export type Attn = 'mha' | 'gqa' | 'mla' | 'dit';

export interface Model {
  id: string;
  name: string;
  family: string;
  attn: Attn;
  kind: 'llm' | 'dit';
  params: number;      // total
  dense: number;       // always-resident, always-read params
  routed: number;      // MoE expert params
  experts: number;
  topk: number;
  layers: number;
  d: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  mlaDim?: number;     // cached elements per token per layer for MLA
  maxCtx: number;
  tag: string;
}

const llm = (o: Partial<Model> & Pick<Model, 'id' | 'name' | 'family' | 'attn' | 'params' | 'layers' | 'd' | 'heads' | 'kvHeads' | 'headDim' | 'maxCtx' | 'tag'>): Model => ({
  kind: 'llm', dense: o.params, routed: 0, experts: 0, topk: 0, ...o,
} as Model);

export const MODELS: Model[] = [
  llm({
    id: 'l8b', name: 'Llama 3.1 8B', family: 'dense · GQA 8 kv-heads', attn: 'gqa',
    params: 8.03e9, layers: 32, d: 4096, heads: 32, kvHeads: 8, headDim: 128,
    maxCtx: 131072, tag: 'GQA',
  }),
  llm({
    id: 'l13b', name: 'Llama 2 13B', family: 'dense · MHA, one kv-head per query head', attn: 'mha',
    params: 13.0e9, layers: 40, d: 5120, heads: 40, kvHeads: 40, headDim: 128,
    maxCtx: 4096, tag: 'MHA',
  }),
  llm({
    id: 'l70b', name: 'Llama 3.3 70B', family: 'dense · GQA 8 kv-heads', attn: 'gqa',
    params: 70.6e9, layers: 80, d: 8192, heads: 64, kvHeads: 8, headDim: 128,
    maxCtx: 131072, tag: 'GQA',
  }),
  llm({
    id: 'l405b', name: 'Llama 3.1 405B', family: 'dense · GQA 8 kv-heads', attn: 'gqa',
    params: 405e9, layers: 126, d: 16384, heads: 128, kvHeads: 8, headDim: 128,
    maxCtx: 131072, tag: 'GQA',
  }),
  {
    id: 'dsv3', name: 'DeepSeek-V3 671B', family: 'MoE 256+1 experts, top-8 · MLA latent kv',
    attn: 'mla', kind: 'llm',
    params: 671e9, dense: 17e9, routed: 654e9, experts: 256, topk: 8,
    layers: 61, d: 7168, heads: 128, kvHeads: 128, headDim: 128,
    mlaDim: 576, maxCtx: 131072, tag: 'MLA + MoE',
  },
  {
    id: 'wan14b', name: 'Wan 2.1 T2V 14B', family: 'video DiT · full bidirectional attention',
    attn: 'dit', kind: 'dit',
    params: 14.3e9, dense: 14.3e9, routed: 0, experts: 0, topk: 0,
    layers: 40, d: 5120, heads: 40, kvHeads: 40, headDim: 128,
    maxCtx: 171360, tag: 'DiT',
  },
];

/* Video shapes: Wan VAE compresses 4x in time and 8x in space, DiT patch size is (1,2,2). */
export const VIDEO_SHAPES = [
  { id: '480p5', label: '480p · 5s', px: '832x480', tokens: 32760, frames: 81 },
  { id: '720p5', label: '720p · 5s', px: '1280x720', tokens: 75600, frames: 81 },
  { id: '720p10', label: '720p · 10s', px: '1280x720', tokens: 147600, frames: 161 },
  { id: '1080p5', label: '1080p · 5s', px: '1920x1088', tokens: 171360, frames: 81 },
];

export const CTX_STEPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
export const BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

export type Serving = 'static' | 'orca' | 'sarathi' | 'pd';

export const SERVING: { id: Serving; label: string; sub: string; blurb: string }[] = [
  {
    id: 'static', label: 'Static batching', sub: 'the baseline',
    blurb: 'A batch runs to completion before the next one starts, so every finished sequence sits idle waiting for the longest one. Latency is clean; most of the GPU is not.',
  },
  {
    id: 'orca', label: 'Continuous batching', sub: 'Orca, OSDI 22',
    blurb: 'Finished sequences leave the batch and new ones join at every iteration. Throughput roughly doubles, but an arriving prefill still runs alone and stalls every decoder behind it.',
  },
  {
    id: 'sarathi', label: 'Chunked prefill', sub: 'Sarathi-Serve, OSDI 24',
    blurb: 'Prefill is cut into 512-token chunks and rides along with the decode batch. The tail stops spiking because no single stall is ever longer than one chunk.',
  },
  {
    id: 'pd', label: 'Prefill / decode split', sub: 'disaggregated',
    blurb: 'Prefill and decode run on separate GPU pools, so neither interferes with the other. The bill is one KV cache transfer across the fabric per request.',
  },
];

/* The 2010 table, verbatim. */
export const CLASSIC: { label: string; ns: number }[] = [
  { label: 'L1 cache reference', ns: 1 },
  { label: 'Branch mispredict', ns: 3 },
  { label: 'L2 cache reference', ns: 4 },
  { label: 'Mutex lock/unlock', ns: 17 },
  { label: 'Main memory reference', ns: 100 },
  { label: 'Compress 1K bytes with Zippy', ns: 2_000 },
  { label: 'Read 1 MB sequentially from memory', ns: 3_000 },
  { label: 'SSD random read', ns: 16_000 },
  { label: 'Round trip within same datacenter', ns: 500_000 },
  { label: 'Read 1 MB sequentially from disk', ns: 825_000 },
  { label: 'Disk seek', ns: 2_000_000 },
  { label: 'Send packet CA→Netherlands→CA', ns: 150_000_000 },
];
