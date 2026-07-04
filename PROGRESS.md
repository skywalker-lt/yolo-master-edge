# YOLO-Master-EsMoE-N Edge Deployment — Progress Log

Workspace: `/data/yolo-master-edge/` (standalone; kept out of the YOLO-Master git tree).
Branch for any in-repo changes: `dev-edge` (off `dev-tph`).
Mission: export EsMoE-N to ONNX + NCNN, validate parity (<0.5% mAP50-95), build C++
edge runners, cross-platform CMake, pre-built binaries (x86_64 + Jetson), benchmark,
writeup. Scoping decisions: **Jetson available**, **2nd format = NCNN (pnnx)**,
**FP32/FP16 first (INT8 deferred)**.

---

## Status board
| Phase | State |
|---|---|
| 0. Tooling | ✅ done |
| 1. Export (ONNX + NCNN) | ✅ done |
| 2. Parity validation | 🟡 CPU + CUDA verified on subset; full 500-img pending |
| 2b. CUDA inference (ORT CUDA-EP) | ✅ verified (mAP == CPU, −0.02 pt; 271 FPS H200) |
| 2c. CUDA C++ binary vs PyTorch (548 imgs) | ✅ mAP50-95 −0.03% AND mAP50 −0.10%, both < 0.5% |
| 3. C++ edge runners (ORT + NCNN) | ✅ v2 versatile runner; 16/16 robustness tests pass (x86_64) |
| 4. CMake + pre-built binaries | 🟡 Linux x86_64 ✅ + **Windows 11 (VS2026/MSVC) ✅** ; Jetson aarch64 ⬜ (adapter shipping) |
| — Deployment repo | ✅ github.com/skywalker-lt/yolo-master-edge (cross-platform CMake, models, scripts) |
| 5. Benchmark (latency/FPS) | ✅ ONNX vs NCNN on Windows CPU + H200 CUDA (table below) |
| 6. Writeup + deployment repo + optional examples/ PR | 🟡 repo ✅; writeup ⬜ |
| (stretch) INT8, MNN | ⬜ INT8 findings logged (Orin TRT); MNN ⬜ |

### Benchmark — format comparison (VisDrone val, 548 imgs, per-frame inference)
| Platform | Backend | pre | infer | post | FPS | total dets |
|---|---|---|---|---|---|---|
| Windows 11 CPU | ONNX (ORT) | 1.6 ms | **37.6 ms** | 0.2 ms | **25.4** | 21934 |
| Windows 11 CPU | NCNN (CPU) | 1.6 ms | **80.1 ms** | 0.2 ms | **12.2** | 21934 |
| Linux H200 | ONNX CUDA (C++) | — | **7.8 ms** | — | **~128** | — |
- ONNX **2.1x faster than ncnn on x86 CPU** (ORT is x86-tuned; ncnn's edge is ARM -> expect the flip on Orin).
- Both formats emit **identical detections (21934 == 21934)** over the full set -> functional format parity, not just close mAP.
- Practical no-Jetson deployment: **~25 FPS** (Windows ONNX CPU).

### mAP parity — ALL formats vs PyTorch (548 val imgs, >500 requirement)
Consistent methodology for every format: conf 0.001, NMS iou 0.7, multi-label, cap 300
(ultralytics val settings) -> `eval_map.py` (ultralytics DetMetrics). Preds: ONNX/ncnn via the
C++ runner `--save-txt`, MNN via `scripts/mnn_val.py` (identical decode).

| Model | mAP50 | mAP50-95 | delta mAP50-95 vs PyTorch |
|---|---|---|---|
| PyTorch (original) | 0.3504 | 0.2036 | - (ref) |
| ONNX | 0.3495 | 0.2034 | **-0.02%** |
| NCNN | 0.3495 | 0.2034 | **-0.02%** |
| MNN  | 0.3495 | 0.2034 | **-0.02%** |

All three export formats -> **identical mAP** (0.2034/0.3495), all **-0.02%** vs PyTorch, 25x inside
the <0.5% non-quant target. Format det-counts: ONNX 157464, ncnn 157465, MNN ~identical.

### 3rd format: MNN (ultralytics has no ONNX->MNN val, so: mnnconvert + numerical parity)
Converted the exact `sim.onnx` -> `esmoe_n_visdrone.mnn` (mnnconvert, 10.8 MB). Verified vs ONNX on
100 val imgs (identical letterboxed inputs): **max|delta|=0.096, mean|delta|=9.7e-05** -> same graph,
same mAP (already <0.5%). MNN pip has no py3.14 wheel -> ran in a minimal py3.11 side-env
(`conda env mnn`, MNN 3.6). Fair same-box CPU (H200, 4 threads): **MNN 74.0 ms / 13.5 FPS** vs
**ONNX 40.0 ms / 25.0 FPS** -> MNN ~1.85x slower on x86 (sits next to ncnn; both mobile/ARM-tuned).
Script: `scripts/mnn_parity.py`. No new prebuilt binary (per scope).

---

## Key findings (the risk retirement)
1. **MoE export is a non-issue — designed for it.** `ES_MOE.forward` (`ultralytics/nn/modules/moe/modules.py:602`) forces the **dense** path under `torch.onnx.is_in_onnx_export()`; `_dense_forward` (line 627) is a static unrolled loop over a fixed expert list → standard `Conv/Pool/Softmax/Mul/Add`. No dynamic routing on the export path. Dense is also the numerically-correct path (sidesteps the sparse-collapse bug).
2. **Reuse, don't rebuild.** Ultralytics `exporter.py` already does ONNX (+simplify via onnxslim) and NCNN via **pnnx**; `YOLO(model).val()` runs mAP on any backend; `benchmark()` does latency+mAP across formats. Only wrote the mission-required **onnxsim** post-step and a parity probe. C++ runners will *adapt* `examples/YOLOv8-ONNXRuntime-CPP`, not rewrite.
3. **NCNN attention fallback is benign.** pnnx emitted `fallback batch axis` on the `A2C2f` self-attention, but NCNN mAP == ONNX mAP exactly → no drift.

---

## Phase 0 — Tooling (done)
Installed into the `yolo_master` conda env (CPU, so GPU-0 training untouched):
`onnx 1.22.0`, `onnxsim 0.6.5`, `onnxruntime 1.27.0`. pnnx + ncnn are auto-fetched
by the ultralytics NCNN exporter.

## Phase 1 — Export (done)
Source checkpoint: `/data/YOLO-Master/scripts/reproduce/results/result-esmoen-visdrone/weights/best.pt`
(VisDrone EsMoE-N, imgsz 640, reproduction run; full-val ~0.350/0.203). Model:
231 layers, 2.65M params, 8.5 GFLOPs, output `(1, 14, 8400)` = 4 bbox + 10 classes × 8400 anchors (P3/P4/P5 @640).

- **ONNX** (opset 12, CPU export, `simplify=False`) → **onnxsim** → validated.
  - node count 833 → **628** after onnxsim; opset `{ai.onnx: 12}`.
  - raw-tensor parity PyTorch vs ORT: **mean|Δ| 6.5e-6, max|Δ| 1.6e-3** (normal FP32 kernel variance).
  - Benign ORT shape-inference warning on `A2C2f` Transpose (functional; parity confirms OK).
- **NCNN** via built-in pnnx path (`model.export(format="ncnn")`), CPU.
  - valid `.param` (magic `7767517`, **561 layers / 665 blobs**), `.bin` 10.7 MB, 9.6 GFLOPs.
  - pnnx `fallback batch axis` warnings on attention — later shown benign.

## Phase 2 (subset) — CPU parity on 50 VisDrone val images (passed)
Subset: `/data/yolo-master-edge/visdrone50/` (50 img + labels symlinked; `visdrone50.yaml`).
Restored ultralytics metadata (names/task/imgsz) onto the onnxsim model so it carries class names.

| Backend | mAP50 | mAP50-95 | Δ vs PyTorch |
|---|---|---|---|
| PyTorch (.pt) | 0.3764 | 0.2496 | — |
| ONNX-sim (ORT) | 0.3703 | 0.2477 | **−0.186 pts** |
| NCNN (pnnx) | 0.3703 | 0.2477 | **−0.186 pts** |

Both exported backends: **within −0.19 mAP50-95 pts of PyTorch (< 0.5% target)**, and
identical to each other. (Subset baseline 0.2496 ≠ full-val 0.203 only because it's a
different image slice — the Δ is what matters.) CPU inference times were config-artifacts
(single-threaded ORT/ncnn vs 96-core torch), NOT deployment latency — measured later on target HW.

---

## Artifacts
```
/data/yolo-master-edge/
├── models/
│   ├── esmoe_n_visdrone.onnx           # raw ONNX (opset 12)
│   ├── esmoe_n_visdrone_sim.onnx       # onnxsim'd + metadata restored (DEPLOY)
│   └── esmoe_n_visdrone_ncnn/          # model.ncnn.param / .bin (DEPLOY)
├── visdrone50/                          # 50-img parity subset + yaml
├── scripts/_subset_val.py               # reusable subset-val (pt/onnx/ncnn)
└── PROGRESS.md                          # this file
```
Ultralytics also wrote `best.onnx` / `best_ncnn_model` into the checkpoint's `weights/`
dir (git-ignored under `scripts/**/results/`).

---

## Phase 3 — C++ edge runners (DONE on x86_64)
Deployment `cpp/` — one binary `yolomaster_edge` over BOTH backends, shared decode.
```
cpp/
├── CMakeLists.txt            # OpenCV + ORT + ncnn; SDK roots overridable (-DONNXRUNTIME_ROOT/-DNCNN_ROOT) for Jetson
├── include/{yolomaster,ort_backend,ncnn_backend}.hpp
└── src/{common,ort_backend,ncnn_backend,main}.cpp
```
- `common.cpp`: centered **letterbox** (aspect-preserve, pad 114), YOLOv8 **decode** (channel-major
  [14×8400], post-sigmoid class scores) + `cv::dnn::NMSBoxes`, VisDrone/SKU class tables, draw.
- `ort_backend`: ONNXRuntime CPU EP; preprocess via `cv::dnn::blobFromImage`.
- `ncnn_backend`: ncnn `Net` (`in0`/`out0`), copies Mat → channel-major buffer → same decode.
- `main.cpp`: CLI `--backend onnx|ncnn --model --image --classes visdrone|sku --conf --iou --threads --repeat --out`.
- Build deps: `apt cmake libopencv-dev` (OpenCV 4.5.4); SDKs in `third_party/` (ORT 1.18.1, ncnn 20260526 shared).
  NOTE: extract SDK tarballs with `tar --no-same-owner` (MooseFS chown bug).

**Validation (x86_64, image `0000001_02999_d_0000005.jpg`, conf 0.25, 8 threads):**
| Backend | dets | infer | total | FPS |
|---|---|---|---|---|
| ONNX (ORT) | 40 | 48 ms | 57 ms | 17.5 |
| NCNN (pnnx) | 40 | 106 ms | 110 ms | 9.1 |

Both backends produce **bit-identical detections** (`car 0.869898` vs `0.869899`, ~1e-6). C++ decode
path correct; pnnx conversion faithful. (ORT>ncnn on x86 CPU is expected; ncnn wins on ARM/Jetson.)

### v2 — versatile/adaptive runner (current)
Rebuilt as a **universal, model-agnostic** tool (weights loaded at runtime, never baked in):
- **CLI11** arg parser (`-m/--model -s/--source -b/--backend --classes --imgsz --conf --iou --threads --limit --out --no-save --quiet`).
- **Backend auto-detect** from the model path (`.onnx`→ORT, dir/`.param`→ncnn).
- **Auto class names + nc + imgsz from model metadata** — ONNX `metadata_props` and ncnn
  `metadata.yaml`; `--classes auto` needs no manual table. Static-input models auto-align imgsz.
- **Versatile `--source`**: image / directory / `dataset.yaml` (resolves the val split) / video.
- Per-frame try/catch (one bad frame can't crash a run); annotated outputs + timing/FPS summary.
- Binary 574 KB (CLI11 templates; still weight-free/light). Reusable battery: `cpp/run_tests.sh`.

**Robustness battery `cpp/run_tests.sh`: 16/16 PASS** — 4 source types, backend/classes/imgsz
auto-detection, ONNX==ncnn parity, overrides, and error paths (missing model/source, unknown
extension, corrupt image skipped, static-imgsz mismatch auto-handled, missing-arg CLI error).

## CUDA inference verification (done, x86_64 H200)
- Gotcha: default `onnxruntime-gpu 1.27` targets **CUDA 13** (`libcudart.so.13`), box is CUDA 12.8.
  Fixed: install CUDA-12 ORT build (index `onnxruntime-cuda-12`) + `LD_LIBRARY_PATH` = torch's
  bundled `site-packages/nvidia/*/lib`. (On Orin/JetPack, match ORT to the JetPack CUDA/TRT.)
- Diff is box-coord FP32 variance only (max 1.76 px; class scores match 1e-5).
- Functional mAP (50-img): CUDA-EP 0.2475 vs CPU-EP 0.2477 (−0.02 pt) vs PyTorch 0.2496 (−0.21 pt) — < 0.5% target.
- Latency 3.68 ms/frame (271 FPS) on H200. `TensorrtExecutionProvider` also present.

## CUDA C++ binary + 548-image parity (done, x86_64 H200)
- C++ ORT backend gained a CUDA EP (`--device cuda`, graceful CPU fallback) + `--save-txt`
  + per-class NMS (class-offset trick, matches ultralytics agnostic=False) + `--max-det`.
  Built against the ORT-GPU 1.20.1 C++ SDK (CUDA 12; also ships TensorRT provider).
  Run libs: torch's `site-packages/nvidia/*/lib` + `third_party/onnxruntime-gpu/lib` on LD_LIBRARY_PATH.
- CUDA latency: **7.8 ms/frame infer (~128 FPS), 96.7 FPS end-to-end** on H200 (548 imgs).
- mAP eval: `scripts/eval_map.py` reuses ultralytics `match_predictions` + `DetMetrics`
  (dumped preds vs VisDrone GT) -> directly comparable to `.val()`.
- **Parity (548 val imgs, conf 0.001 / iou 0.7 / max_det 300, --multi-label):**
  | metric | PyTorch | ONNX-CUDA | C++ CUDA | Δ |
  |---|---|---|---|---|
  | mAP50 | 0.3504 | 0.3494 | **0.3494** | **−0.10%** ✅ |
  | mAP50-95 | 0.2036 | 0.2032 | **0.2033** | **−0.03%** ✅ |
  C++ postproc is bit-faithful to ultralytics (mAP50 0.3494 == ONNX-CUDA 0.3494). Two fixes:
  float boxes (cv::Rect2f/Rect2d, no int rounding) + **multi-label decode** (`--multi-label`, one
  det per class>conf per anchor = ultralytics `multi_label=True`; the real driver). Multi-label is
  opt-in (post 0.7->3.9ms) for eval-parity; deploy default stays single-argmax/fast (= ultralytics predict).

## INT8 quantization (attempted x86 CPU, DEFERRED to Orin TensorRT)
No ultralytics ONNX-INT8 path -> hand-rolled `scripts/quantize_int8.py` (onnxruntime.quantization,
static QDQ/QOperator, per-channel, opset->17, `--exclude` for mixed precision). Calibrate on VisDrone
train (500 imgs, no leakage). Findings on 548 val imgs (mAP50-95, ref PyTorch 0.2036):
  - full INT8 -> **0.0000** (cls head collapses: quantized logits -> sigmoid -> 0).
  - head (`/model.25/`) kept FP32 -> **0.1924 (−1.12%)** — just over target.
  - + attn (`/attn/`) + router (`routing`) FP32 -> expected < 1%, but UNCONFIRMED: CPU INT8 is
    untenable here (1.7x SLOWER than FP32; ORT calibration augments+collects all activations over
    500 imgs = 20-60 min/attempt; each val ~45 min).
**Decision:** defer INT8 to the Orin via `model.export(format="engine", int8=True, data=...)`
(ultralytics built-in, fast on-GPU calibration, real tensor-core speedup) with the **cls head pinned
to FP32** (finding #1). CPU INT8 gives no speedup and is the wrong platform.

## Next
- **TensorRT verify (x86)**: the ORT-GPU SDK ships libonnxruntime_providers_tensorrt.so; try TRT EP to de-risk the Orin fast-path.
- **Full 500-img mAP** across pt/onnx/ncnn/cuda.
- **Export + validate the new EsMoE-N-P2** checkpoint (+3% mAP50 / +2.2% mAP50-95 over baseline).
- **C++ `--device cuda`**: add CUDA EP to OrtBackend (needs ORT-GPU C++ SDK + rpath to CUDA libs).
- **Jetson aarch64 build**: cross/native build with aarch64 ORT + ncnn SDKs (`-DONNXRUNTIME_ROOT/-DNCNN_ROOT`); on-device latency/FPS (ncnn should lead there).
- **Benchmark table** ONNX vs NCNN (x86_64 + Jetson).
- **Writeup** (GitHub Discussion) + `git init` the deployment repo + optional `examples/` PR.
- SKU-110K variant (export its checkpoint; higher `--imgsz`; `--classes sku`).
