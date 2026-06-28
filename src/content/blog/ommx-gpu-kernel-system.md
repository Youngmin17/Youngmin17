---
title: "OMMX GPU kernels: the serving-stack path"
date: 2026-06-24
author: Youngmin Cho
tags: [GPU, OMMX, vLLM, CUDA]
subtitle: "How OMMX INT2+FP4 kernels integrate into a paged-attention serving stack — and where low-bit KV actually pays off."
---

A note on the GPU side of **OMMX** — CUDA / Triton kernels for INT2 + FP4 outlier
quantization, integrated into a vLLM v1 backend.

## What ships

- **Paged decode attention** — INT2/FP4 KV cache with a recent-token window, fused dequant.
- **vLLM integration** — a custom attention backend + packed-KV capacity mode.
- **Benchmarks** — TTFT / TPOT measured against FP16 and Triton baselines.

## Honest take on TPOT

Low-bit KV quantization wins decisively on **accuracy and KV capacity**, but on GPUs the
decode-step **TPOT** gain is limited (and can be negative) at small batch sizes — the dequant
cost is not amortized when each step loads only a few tokens. The real lever is **memory
capacity** (more sequences / longer context), not raw decode latency.

*Detailed benchmarks coming soon.*
