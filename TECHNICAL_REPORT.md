# Edge Deployment of YOLO-Master-EsMoE-N: ONNX / NCNN / MNN, INT8, a cross-platform C++ runtime, and full mAP + latency benchmarks

This write-up documents an end-to-end edge-deployment of **YOLO-Master-EsMoE-N** on **VisDrone**: three export formats, INT8 quantization, a universal C++ inference runtime, cross-platform builds (Linux x86_64 + Windows 11), and a rigorous accuracy/latency comparison against the PyTorch original.

**Deployment repo (source, models, scripts, prebuilt bundles):** https://github.com/skywalker-lt/yolo-master-edge

## TL;DR

| Item | Result |
|---|---|
| Formats exported | ONNX + NCNN + MNN |
| mAP50-95 vs PyTorch (548 val, FP32) | **−0.02%** (target < 0.5%) |
| mAP50-95 vs PyTorch (548 val, INT8) | **−0.84%** (target < 1.0%) |
| Platforms built & run | Linux x86_64 (+CUDA) and Windows 11 x64 |
| Fastest backend (x86 CPU) | ONNX/ORT, ~25 FPS; CUDA C++ ~128 FPS |

---

## 1. Model & checkpoint

Base model: **EsMoE-N** (`ES_MOE` sparse-inference variant) trained on **VisDrone** (10 classes), checkpoint reused directly (`runs/baseline/EsMoE-N_VisDrone/weights/best.pt`). PyTorch reference metrics on the 548-image val split (ultralytics `val`): **mAP50 = 0.3504, mAP50-95 = 0.2036**.

Note on MoE export: `ES_MOE.forward` forces the **dense** expert path under `torch.onnx.is_in_onnx_export()`, so the exported graph is a static `Conv/Pool/Softmax/Mul/Add` unroll with no dynamic routing — the dense path is also the numerically correct one, which is why export parity is essentially exact (below).

## 2. Export to three formats

- **ONNX** — exported at **opset 12**, simplified with **onnxsim**, ultralytics metadata (names/imgsz/stride/task) embedded. Static shapes: input `images [1,3,640,640]`, output `output0 [1,14,8400]` (4 box + 10 class), 628 nodes. Opset-12 compatibility verified by loading/running under ONNXRuntime 1.18 / 1.20 / 1.27 and by clean downstream conversion to both NCNN and MNN.
- **NCNN** — converted via **pnnx** (ONNX → pnnx → ncnn). Param file validated: magic `7767517`, **561 layers / 665 blobs**, input blob `in0`, sigmoid-terminated detection head; pnnx `model_ncnn.py` companion emitted. A `metadata.yaml` sidecar carries class names + imgsz.
- **MNN** — converted with `mnnconvert` (ONNX → MNN, 10.8 MB), same graph as ONNX/ncnn.

## 3. INT8 quantization (≥ 300 calibration images)

Static **QOperator per-channel INT8** via `onnxruntime.quantization`, **MinMax** calibration on **300 VisDrone *train* images** (no val leakage; images letterboxed to match inference).

Naive full-INT8 **collapses** (the classification head's logits quantize into a range where sigmoid → 0, zero detections). The fix is **mixed precision** — keep the three quant-hostile blocks in FP32 and INT8 everything else:

| Config | mAP50-95 | Δ vs PyTorch |
|---|---|---|
| Full INT8 | 0.0000 | collapse (cls head → 0) |
| + head (`/model.25/`) FP32 | 0.1924 | −1.12% |
| **+ attention (`/attn/`) + ES-MoE `routing` FP32** | **0.1952** | **−0.84% ✅ < 1.0%** |

Final INT8 model: **10.9 MB → 5.4 MB (2.0×)**, 289 nodes kept FP32. (INT8 gives no *speedup* on CPU — that payoff is on INT8 tensor cores via TensorRT on-device — but the **accuracy budget is met**.)

## 4. Edge inference runtime (C++)

A single universal binary (`yolomaster_edge`) with **ONNXRuntime** and **NCNN** backends, auto-detecting backend, class names, and input size from the model. Vertical-domain preprocessing: **aspect-ratio-preserving letterbox** (min-side scale, 114 padding) → RGB `/255` NCHW. Versatile `--source`: image / directory / video / `dataset.yaml`. 16/16 robustness tests pass.

**Post-processing / NMS tuning for the vertical domain:** low default `conf` for VisDrone's small/dense objects, tunable `--conf`/`--iou`, **per-class NMS** (class-offset trick, matches ultralytics `agnostic=False`), and a `--multi-label` mode (one detection per class > conf per anchor) that reproduces ultralytics `val` exactly for apples-to-apples mAP.

## 5. Accuracy: mAP50-95 vs PyTorch (548 val images, > 500 requirement)

One consistent metric harness for **every** model (conf 0.001, NMS iou 0.7, multi-label, cap 300; ultralytics `DetMetrics`). ONNX/ncnn predictions dumped by the C++ runner; MNN by a Python runner replicating the identical decode.

| Model | mAP50 | mAP50-95 | Δ mAP50-95 vs PyTorch |
|---|---|---|---|
| **PyTorch (original)** | 0.3504 | 0.2036 | — |
| ONNX | 0.3495 | 0.2034 | **−0.02%** |
| NCNN | 0.3495 | 0.2034 | **−0.02%** |
| MNN  | 0.3495 | 0.2034 | **−0.02%** |
| INT8 (mixed) | 0.3377 | 0.1952 | **−0.84%** |

All three FP32 export formats reproduce PyTorch to **−0.02%** (25× inside the 0.5% target) and land on *identical* mAP; INT8 is **−0.84%** (inside 1.0%).

## 6. Latency / throughput benchmark (VisDrone val, per-frame inference)

| Platform | Backend | infer (ms) | FPS |
|---|---|---|---|
| Windows 11 CPU | ONNX (ORT) | 37.6 | **25.4** |
| Windows 11 CPU | NCNN | 80.1 | 12.2 |
| Linux CPU (4-thr) | ONNX (ORT) | 40.0 | 25.0 |
| Linux CPU (4-thr) | **MNN** | 74.0 | 13.5 |
| Linux CPU (4-thr) | NCNN | ~80 | ~12.5 |
| Linux H200 | **ONNX CUDA (C++)** | 7.8 | **~128** |

Consistent picture: **ORT is fastest on x86** (heavily x86-tuned), while **MNN ≈ NCNN at ~half the speed** — both are mobile/ARM-focused, so they're expected to lead on the Orin. CUDA gives a ~5× jump over CPU.

## 7. Numerical parity analysis

Even though errors stayed within target, per-graph parity was checked directly. Feeding **identical letterboxed inputs** to MNN vs the source ONNX over 100 val images: **max|Δ| = 0.096, mean|Δ| = 9.7e-05** on the raw `[1,14,8400]` output (the max is a single box-coordinate LSB; the mean is negligible). Detection counts across formats over the full set are effectively equal (ONNX 157,464 vs ncnn 157,465 at conf 0.001) — functional format equivalence, not just close mAP.

## 8. Cross-platform builds & portable bundles

Cross-platform **CMake**; built and run on **two platforms**:
- **Linux x86_64** (+ CUDA via the ONNXRuntime CUDA EP) — C++ CUDA parity vs PyTorch also < 0.5%.
- **Windows 11 x64** (VS 2026 / MSVC) — CPU, ~25 FPS.

Both are packaged as **self-contained, relocatable bundles** (Linux: `$ORIGIN`-rpath'd, 10 bundled libs, 35 MB, verified running with no `LD_LIBRARY_PATH`; Windows: MSVC runtime bundled, no VC++ Redist needed). To keep the Linux bundle lean, image I/O uses `stb_image` and NMS is hand-written, dropping OpenCV `imgcodecs`(→GDAL) and `dnn`(→protobuf) — **231 libs/129 MB → 10 libs/35 MB**.

## 9. Requirements checklist

| Requirement | Status |
|---|---|
| Train/finetune EsMoE-N on VisDrone (reuse checkpoint) | ✅ |
| Export ≥ 2 formats (ONNX + NCNN / MNN) | ✅ (all three) |
| onnxsim + opset compat; pnnx + param validation; INT8 ≥ 300 calib | ✅ |
| Edge inference code + vertical preprocessing (letterbox) | ✅ |
| Vertical NMS tuning (low conf, per-class) | ✅ |
| CMake, ≥ 2 platforms build+run | ✅ (Linux x86_64 + Windows 11) |
| mAP50-95 vs PyTorch ≥ 500 imgs, < 0.5% / < 1.0% INT8 | ✅ (−0.02% / −0.84%) |
| Layer/visual diff analysis if error exceeds | ✅ (numerical parity provided) |
| Latency + FPS, ONNX vs NCNN vs MNN | ✅ |

## Future work

- **NVIDIA Jetson Orin (aarch64) + TensorRT.** The same CMake builds natively on aarch64 with no changes; the next step is a TensorRT FP16/INT8 engine built on-device, where INT8's *speed* payoff (INT8 tensor cores) and NCNN/MNN's ARM advantage materialize.
- **Production drone platform — DJI Manifold 3.** VisDrone is aerial/drone imagery, so the natural production target is a real onboard drone computer. [DJI Manifold 3](https://enterprise.dji.com/manifold-3) is an **NVIDIA Orin NX-based** enterprise edge computer purpose-built for drones — the exact aarch64 + TensorRT path above deploys directly onto it. Testing this pipeline on the Manifold 3 would validate **real-time on-drone inference in operational scenarios** (aerial surveillance, infrastructure inspection, search-and-rescue), closing the loop from VisDrone training to production drone edge deployment.

---

*Reproducibility:* all scripts (`quantize_int8.py`, `eval_map.py`, `mnn_val.py`, `mnn_parity.py`, `package_linux.sh`), models, and the C++ runtime are in the repo above.
