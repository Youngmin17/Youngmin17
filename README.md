# Hi, I'm Youngmin 👋

Systems engineer working on **efficient LLM inference** — from ultra-low-bit
quantization and GPU kernels up to custom NPU hardware.

> 효율적인 LLM 추론을 연구·구현합니다 — 초저비트 양자화부터 GPU 커널, 자체 NPU
> 하드웨어까지.

### What I work on
- **Low-bit LLM serving** — INT2 + FP4 outlier quantization (weights & KV cache),
  accuracy-preserving formats, and serving integration (vLLM / SGLang).
- **GPU kernels** — CUDA / Triton kernels for quantized attention and GEMM;
  paged-attention decode, outlier-aware dequant, CUDA-graph capture.
- **NPU hardware** — a multi-core inference accelerator (RTL + compiler +
  cycle-accurate model) for the same low-bit format.
- **Edge AI** — on-device speech / kiosk demos.

### Featured
- 🎙️ [ai-voice-kiosk-edge-demo](https://github.com/Youngmin17/ai-voice-kiosk-edge-demo)
  — edge AI voice kiosk for resident-center & convenience-store scenarios.

### Toolbox
`PyTorch` · `CUDA` · `Triton` · `vLLM` · `SGLang` · `Verilog/SystemVerilog` ·
`HuggingFace` · `lm-eval`

---
<sub>Some research repositories are private while in progress.</sub>
