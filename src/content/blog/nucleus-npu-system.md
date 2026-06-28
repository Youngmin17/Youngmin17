---
title: "Nucleus NPU: a low-bit inference accelerator"
date: 2026-06-20
author: Youngmin Cho
tags: [NPU, Nucleus, Silicon]
subtitle: "A system note on Nucleus — the HF→ISA compiler, cycle-accurate model, and RTL for the OMMX low-bit format."
---

A short system note on **Nucleus**, a multi-core inference accelerator co-designed with the
OMMX low-bit format (INT2 dense + FP4 outlier).

## The stack

- **Compiler** — lowers a HuggingFace config into a high-level op-ISA (GEMM / GEMV / GQA /
  RoPE / SwiGLU), with quantization metadata baked into each instruction.
- **Cycle-accurate model** — an ISA-level simulator with an integrated DRAM model
  (Ramulator), cross-validated against RTL simulation.
- **RTL** — a multi-core datapath with mixed-precision MAC units and a streaming DMA engine
  that decompresses OMMX bundles on the fly.

> Goal: one quantization format that runs on both GPU kernels and custom silicon, with
> bit-exact parity between them.

*Full write-up coming soon.*
