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
  - `init(modelDir, useVulkan, threads)` / `initBest(modelDir, probe)` (Vulkan-with-parity-fallback)
  - `setConfig(conf, iou, maxDet)` (cheap retune, reuses the cached forward)
  - `infer(bitmap) -> List<Detection>`, `inferSeg(bitmap) -> SegResult`
  - `activeBackend` ("ncnn-CPU-fp32" | "ncnn-Vulkan"), `lastError`, `close()`
- An instrumented parity/robustness harness (`ParityTest`).

## Robustness model (why this is not a naive ncnn wrapper)

Two ARM-only hazards silently break these models on phones; both are handled:

1. **fp16 underflow (ARM CPU).** ncnn enables fp16 CPU kernels on armv8.2 (every modern
   phone). The mixture-of-experts routing uses ~1e-7 / 1e-9 epsilons that underflow to
   zero in fp16 and return **no detections**. The CPU path is pinned to fp32 in
   `cpp/src/ncnn_backend.cpp:23-33` and inherited here. Never force fp16 on CPU.
2. **Unregistered router ops.** ncnn has no `TopK/Gather/Where`, so raw gated-router MoE
   models will not load. The models are exported with `scripts/export_ncnn_mixture.py`,
   which rewrites the router into stock ncnn ops (census-gated). The Android runtime loads
   the resulting `.param/.bin` unchanged.

Vulkan (fp16, ~4x faster) is opt-in: it is used only if a GPU is actually present
(`ncnn::get_gpu_count()`), otherwise the runtime transparently falls back to CPU-fp32.
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

The harness asserts:
- the `v0.1-seg-N` default loads, reports **ncnn-CPU-fp32**, and produces detections
  (the fp16-underflow regression guard);
- an emulated-router mixture model (`moa-n`) detects on ARM (both hazards handled on-device);
- Vulkan agrees with CPU within +/-1 detection, or falls back to CPU-fp32 cleanly;
- a missing model fails with an error, not a crash.

Latencies are logged under the `ParityTest` / `YMNcnn` tags (`adb logcat`).

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
