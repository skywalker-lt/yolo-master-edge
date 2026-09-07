# YOLO-Master Android runtime (ncnn)

The Android backend runtime for the cross-platform edge example. It runs YOLO-Master
detection and segmentation on-device with **ncnn**, reusing the shared C++ core in
`../cpp` verbatim (the same letterbox -> ncnn -> decode -> NMS path as the Linux, Windows,
Jetson, and macOS runners) behind a small JNI bridge.

This milestone is the **runtime + JNI + on-device test harness only** — there is no app UI
yet. A future app module depends on `:runtime`.

## What it provides

- `libyolomaster_ncnn.so` (arm64-v8a, x86_64) wrapping the ncnn backend.
- A thin Kotlin API, `dev.yolomaster.ncnn.YoloMasterNcnn`:
  - `init(modelDir, useVulkan, threads, precision)` / `initBest(modelDir, probe, threads, precision)`
    (precision-with-parity-fallback, then Vulkan-with-parity-fallback)
  - `setConfig(conf, iou, maxDet)` (cheap retune, reuses the cached forward)
  - `infer(bitmap) -> List<Detection>`, `inferSeg(bitmap) -> SegResult`
  - `activeBackend` (the REAL resolved precision: "ncnn-CPU-fp32" | "ncnn-CPU-fp16" |
    "ncnn-CPU-int8+fp32" | "ncnn-CPU-int8+fp16" | "ncnn-Vulkan" | "ncnn-Vulkan-fp32"),
    `backendNote` (why a requested precision was downgraded), `lastError`, `close()`
  - `Precision { AUTO, FP32, FP16, INT8 }` - see "Precision modes and mixed-INT8" below
- An instrumented parity/robustness harness (`ParityTest`).

## Robustness model (why this is not a naive ncnn wrapper)

Two ARM-only hazards silently break these models on phones; both are handled:

1. **fp16 underflow (ARM CPU) - handled PER MODEL.** ncnn enables fp16 CPU kernels on
   armv8.2 (every modern phone). The emulated mixture-of-experts router uses `1e-9` mask
   nudges and `1e30` expert masks that are unrepresentable in fp16 (1e-9 flushes to zero
   under ARM FZ16, 1e30 overflows fp16's 65504 max) and return **no detections**. The
   runtime reads that fingerprint from the `.param` itself (`meta::scan_ncnn_param` in
   `cpp/src/common.cpp`, applied in `cpp/src/ncnn_backend.cpp`): mixture models are pinned
   fp32 (with the reason in `backendNote`), while the dense models (v0.1, EsMoE-N, whose
   graphs carry no such constants) run **fp16** on armv8.2 - about -45% latency vs fp32
   (Orin measurement). An explicit `Precision.FP16` request on a mixture model is
   downgraded and explained, never run as a silent zero-detection model.
2. **Unregistered router ops.** ncnn has no `TopK/Gather/Where`, so raw gated-router MoE
   models will not load. The models are exported with `scripts/export_ncnn_mixture.py`,
   which rewrites the router into stock ncnn ops (census-gated). The Android runtime loads
   the resulting `.param/.bin` unchanged.

Vulkan (fp16, ~4x faster) is opt-in: it is used only if a GPU is actually present
(`ncnn::get_gpu_count()`), otherwise the runtime transparently falls back to the CPU
choice. The same per-model rule applies on the GPU: a mixture model runs Vulkan with the
fp16 flags off (`ncnn-Vulkan-fp32`). INT8 never uses Vulkan (no ncnn Vulkan int8 kernels).
`initBest()` additionally verifies the GPU agrees with the CPU reference on a probe image
before trusting it, so per-device driver differences cannot ship silent garbage.

## Supported models (validated ncnn set)

v0.1 (all sizes incl. the `v0.1-seg-N` default), EsMoE-N, and the four supported mixture
models (`moa-n`, `mot-n`, `moa-mot-n`, `molora-merged`). Out of scope for now: UoMoE-N
(no ncnn export exists), `molora-routed`, and yolo26 end2end-on-ncnn.

## Build prerequisites

1. **ncnn Android SDK**, tag **20260526**, `-android-vulkan` variant (matches the desktop
   SDK so `.param/.bin` are byte-parity). Download from the ncnn releases, or cross-build
   with the NDK mirroring `../jetson/24_build_ncnn.sh`.
2. **opencv-mobile** Android SDK (core + imgproc).
3. **Android NDK** r26+ and a JDK 17.

Keep both SDKs out of git (they are large). Then:

```bash
cd android
cp sdk-paths.example.properties sdk-paths.properties
$EDITOR sdk-paths.properties     # set NCNN_ANDROID_ROOT and OPENCV_ANDROID_ROOT
./gradlew :runtime:assembleRelease
```

## Run the on-device harness

The models are not committed. Stage your local ncnn model dirs and a probe image, then run
the instrumented tests on a connected arm64 device:

```bash
./scripts/stage_models.sh                 # copies models/*_ncnn -> runtime assets (gitignored)
# put any scene image at runtime/src/main/assets/probe.jpg
./gradlew :runtime:connectedAndroidTest
```

The harness (`ParityTest`) asserts:
- the `v0.1-seg-N` default with explicit `FP32` reports **ncnn-CPU-fp32** and detects (the
  original fp16-underflow regression guard), and with `AUTO` reports **ncnn-CPU-fp16** on
  arm64 (fp32 on the x86_64 emulator, where ncnn's fp16 kernels do not exist) and still
  detects within +/-1 of the fp32 count - the fp16 guard for a dense model;
- an emulated-router mixture model (`moa-n`) is pinned **ncnn-CPU-fp32** by fingerprint
  (`backendNote` says why), detects on ARM, and an explicit `FP16` request is refused;
- Vulkan agrees with CPU within +/-1 detection, or falls back to the CPU choice cleanly;
- a missing model fails with an error, not a crash; `INT8` with no `-int8_ncnn` sibling
  fails with an error naming int8 (never a silent float fallback).

Latencies are logged under the `ParityTest` / `YMNcnn` tags (`adb logcat`).

## Precision modes and mixed-INT8

`Precision.AUTO` is the default and the right choice for apps: the precision is derived from
the model (see hazard 1 above). `FP32` / `FP16` are explicit overrides (downgraded with a
`backendNote` when the model cannot honour them). `INT8` loads a **pre-quantized sibling
directory** named `<name>-int8_ncnn` next to the float dir - the same three files - produced
by `scripts/quantize_ncnn_int8.py` (mixed per-layer int8: the conv trunk is int8, the stem
pair, DFL and any router layers stay float; see its `metadata.yaml` `quant:` block for the
exact provenance). INT8 is CPU-only; its float remainder follows the same per-model rule
(fp16 for dense models, fp32 for mixture models). ncnn loads a mixed int8/float graph
natively - there is no runtime switch to flip; the mode simply selects which directory is
opened. Stage the siblings with `scripts/stage_models.sh` (they are optional: a missing one
only skips the INT8 rows).

`LatencyBenchTest` sweeps `(model, precision) x threads {1, 2, 4, big}` and logs one parseable
line per configuration (warmup 10, 50 timed end-to-end infers on the real probe):

```
YM_LAT model=.. precision=.. backend=.. abi=.. threads=.. n=.. median_ms=.. p90_ms=.. dets=.. fp32_dets=.. note=".."
```

Capture with `adb logcat -s ParityTest | grep YM_LAT`. `cpp/tools/ncnn_bench` gives the
kernel-only number for the same variants. Speed verdicts come **only** from an arm64 device:
x86 cannot run ncnn's fp16 kernels at all, and its int8 path (AVX512-VNNI) says nothing about
ARM `sdot`/`i8mm`. INT8 accuracy, on the other hand, is certified on Linux first
(`tests/certify_ncnn_int8.py`, gate: mAP50-95 within 1.0 point of fp32) before any speed
claim is made.

## Layout

```
android/
  settings.gradle.kts  build.gradle.kts  gradle.properties
  sdk-paths.example.properties            # template (real one gitignored)
  scripts/stage_models.sh
  runtime/                                # com.android.library = the runtime
    build.gradle.kts
    src/main/cpp/{CMakeLists.txt, jni_bridge.cpp}
    src/main/kotlin/dev/yolomaster/ncnn/{YoloMasterNcnn,Types}.kt
    src/main/assets/models/.gitkeep       # staged models land here (payloads gitignored)
    src/androidTest/kotlin/.../ParityTest.kt
```

The `cpp/` core is not duplicated; `runtime/src/main/cpp/CMakeLists.txt` compiles
`../../../../../cpp/src/{common,ncnn_backend,stb_impl}.cpp` directly, the same way the iOS
app reuses `mac/Sources/YOLOMasterKit`.

## Not in this milestone

App UI, CameraX/live video, UoMoE-N ncnn export, AAR publishing, upstream PR.
