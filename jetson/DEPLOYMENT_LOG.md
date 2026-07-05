# Jetson Orin Nano — Deployment Log & Findings

Field log of deploying YOLO-Master-EsMoE-N to a **Jetson Orin Nano (4 GB)** via TensorRT.
Captures the working numbers, the memory constraints of the 4 GB variant, and a reproducible
**TensorRT builder bug** (KTM / FP16) with its workaround — kept for a future PR / bug report.

## 1. Platform

| | |
|---|---|
| Device | Jetson **Orin Nano 4 GB** (`Orin`, Compute Capability **8.7 / sm87**, 4 SMs @ 0.624 GHz) |
| OS / JetPack | Ubuntu **24.04**, JetPack (CUDA **13.2**, driver **595.78**) |
| TensorRT | **10.16.2** (`trtexec` at `/usr/src/tensorrt/bin/trtexec`) |
| GPU memory visible to trtexec | ~3459 MiB (shared 4 GB, minus OS/GUI) |
| Model | `esmoe_n_visdrone_sim.onnx` — opset 12, `images[1,3,640,640] → output0[1,14,8400]` |

## 2. Result (deployment number)

Engine built + benchmarked with `trtexec` (synthetic input, batch 1):

| Precision | GPU compute | Throughput | Latency (e2e) | Engine | Build time |
|---|---|---|---|---|---|
| **FP16** ¹ | **31.36 ms** | **31.6 FPS** | 32.0 ms (H2D 0.6 + compute 31.4 + D2H 0.05) | 5.2 MB | 648 s @ OPT=2 |
| INT8 (calibrated QDQ) | *pending* | *pending* | | | |

¹ Measured via the `--int8 --fp16` run, which **fell back to FP16** (no calibration data → activations
can't be quantized → FP16 kernels). So this is the true FP16 speed. The pure-`--fp16` build failed — see §4.

**Context (full benchmark, VisDrone val, per-frame):** x86 CPU (ORT) 40 ms / 25 FPS · **Orin Nano FP16 31.4 ms / 31.6 FPS** · H200 CUDA (C++) 7.8 ms / ~128 FPS. ~31 FPS on a 4 GB edge module is real-time for tiny-object aerial detection.

## 3. 4 GB memory constraints (build, not runtime)

The TensorRT *builder* profiles many tactics, each wanting 100s of MB. On the 4 GB Nano with the desktop
GUI running, only **~216 MB** was free → the builder skipped every fast tactic:

```
[W] [TRT] Tactic Device request: 414MB Available: 216MB. Device memory is insufficient to use tactic.
[W] [TRT] UNSUPPORTED_STATE: Skipping tactic … due to insufficient memory
```
(repeated for every layer → build thrashed for 6+ min, risking outright failure).

**Fixes (all required on 4 GB):**
1. **Headless:** `sudo systemctl isolate multi-user.target` (frees ~1–1.5 GB; brings free RAM from 216 MB → ~1.3 GB). Restore GUI: `sudo systemctl isolate graphical.target`.
2. **Small workspace:** `--memPoolSize=workspace:256`.
3. **Low opt level:** `--builderOptimizationLevel=2` (fewer tactics profiled → faster build). See §5.
4. Kill stray builds between attempts: `sudo pkill -f trtexec`.

Runtime inference is *not* constrained — the engine is 5 MB and needs ~14 MB activation memory.

## 4. BUG (reproducible): pure-FP16 build fails with a KTM / sm80-shader assertion

### Reproduction
```
Platform: Jetson Orin Nano 4GB — JetPack Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.2, GPU sm87
Command:  trtexec --onnx=esmoe_n_visdrone_sim.onnx --fp16 \
            --saveEngine=fp16.engine --memPoolSize=workspace:256 --builderOptimizationLevel=2
```
### Output (fails, empty engine, ~9 s)
```
Linkable shader sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x64_
  stage3_warpsize4x1x1_g1_tensor16x8x16_linkable doesn't have base shader
  sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x64_stage3_warpsize4x1x1_g1_tensor16x8x16
[E] Error[1]: Unexpected exception KTM assertion failure:
    /_src/externals/ktm/src/caskTimingModels/convolutionTimingModel.cpp:65  shader != nullptr
[I] Created engine with size: 0 MiB
[E] Assertion failure: false && "Attempting to access an empty engine!"
```
### Root cause (hypothesis)
At `builderOptimizationLevel <= 2`, TensorRT uses the **KTM (Kernel Timing Model)** — a heuristic that
*estimates* kernel latencies without running them (this is what makes low opt levels fast). On TRT 10.16.2
the FP16 convolution timing model references an **`sm80` (A100) shader that has no base shader for the
device's `sm87` (Orin)**, tripping the assertion and producing an empty engine.

### Workarounds (both verified paths)
- **`--builderOptimizationLevel=3`** (or higher): TensorRT *profiles* tactics on-device instead of using
  the KTM estimate, bypassing the buggy model. Costs a longer build (~10 min) but succeeds.
- **`--int8 --fp16`**: took a different tactic-selection path and built successfully at OPT=2 (this is how
  the §2 number was obtained; it FP16-fell-back for lack of calibration).

### Takeaway for the kit
`OPT=2` is safe for the INT8/mixed path but **pure FP16 needs `OPT>=3` on Orin/TRT 10.16.2**.

## 4b. GOTCHA (reproducible): QDQ INT8 parse fails on asymmetric zero-point

### Reproduction
An INT8 ONNX exported by `onnxruntime.quantization` (QDQ) → `trtexec --int8 --fp16`:
```
[E] [TRT] ModelImporter.cpp:149: ERROR: onnxOpImporters.cpp:1738 In function QuantDequantLinearHelper:
    [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.
    Please set kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA to enable asymmetric quantization … on DLA.
[E] Failed to parse onnx file
```
### Root cause
TensorRT GPU INT8 requires **symmetric** quantization (`zero_point = 0`). ONNXRuntime's static
quantization defaults to **asymmetric activations** (non-zero zero point) to maximize dynamic-range use,
which TensorRT's ONNX parser rejects at import.

### Fix
Re-quantize with symmetric activations + weights. In `onnxruntime.quantization.quantize_static`:
```python
extra_options={"ActivationSymmetric": True, "WeightSymmetric": True}
```
(exposed as `scripts/quantize_int8.py --symmetric`). Symmetric may cost a hair of accuracy vs asymmetric,
but it's mandatory for the GPU INT8 path.

### Note
The **QOperator** INT8 model used for the CPU/ORT accuracy check (−0.84% mAP) is a *different* format and
does not go through this parser; only the **QDQ-for-TensorRT** export needs `--symmetric`.

## 4c. GOTCHA (reproducible): QDQ INT8 parse fails on int32-quantized bias

### Reproduction
After fixing §4b (symmetric), `trtexec --int8 --fp16` on the QDQ model:
```
[E] Error[3]: IDequantizeLayer::setPrecision: A DequantizeLayer can only run in
    kINT8/kFP8/kFP4/kINT4 precision
[E] While parsing DequantizeLinear -> "model.0.conv.bias"  (input: bias_quantized, int32)
[E] Failed to parse onnx file
```
### Root cause
ONNXRuntime quantizes conv **biases to INT32** (standard: `bias_scale = input_scale × weight_scale`) and
emits an int32 `DequantizeLinear`. TensorRT's `DequantizeLayer` accepts only INT8/FP8/FP4/INT4 — and TRT
handles conv bias **internally**, so it wants *no* Q/DQ on biases at all.

### Fix (two options)
- **At quantization:** `extra_options={"QuantizeBias": False}` (exposed via `--symmetric` in
  `scripts/quantize_int8.py`, which now sets it) — biases stay FP32, no int32 DQ.
- **Post-hoc surgery (exact, no re-calibration):** drop each int32-bias `DequantizeLinear` and inline the
  reconstructed FP32 bias `= int32_bias × scale` (zp=0). Verified `max|Δ| = 0` vs the pre-surgery model —
  it's numerically identical, just TRT-parseable. (Used here to avoid a 4th 15-min calibration.)

### The three ORT→TensorRT QDQ requirements (summary)
For an `onnxruntime.quantization` QDQ model to build in TensorRT: **(1)** symmetric activations
(§4b), **(2)** no int32-bias DQ (§4c), **(3)** opset ≥ 13 for per-channel DQ. All handled by
`quantize_int8.py --symmetric` (+ the opset upgrade it already does).

## 5. `--builderOptimizationLevel` tradeoff

Controls how hard TensorRT searches for the fastest per-layer kernel **at build time**. Higher = profiles
more tactics = slower build, marginally faster engine. **Accuracy is identical at every level** (same math,
different kernel). For this small model OPT=2 ≈ OPT=3 in inference speed; the only real effect is the KTM
bug above (which forces OPT≥3 for pure FP16).

## 6. Repro scripts

`jetson/{00_setup.sh, 10_trt_bench.sh, 20_build_runner.sh}` + `README.md`. Models via `mdb:/tmp` or the
GitHub Release. Power: `nvpmodel -m 0` + `jetson_clocks` (re-run after reboot).

## 7. Open

- **Calibrated INT8 (QDQ):** `esmoe_n_visdrone_int8_qdq.onnx` (mixed precision — head/attn/router FP32),
  regenerated on the server; build with `--int8 --fp16` (+ `OPT=3` if it hits the KTM bug on FP32-fallback
  layers). Expected to beat FP16 via the Orin's INT8 tensor cores — closes the deferred INT8-perf validation.
- **On-device mAP:** run the engine over `visdrone_val_bench.tgz` (on `mdb:/tmp`) to confirm FP16/INT8
  accuracy matches the <0.5% / <1% targets on-device.
