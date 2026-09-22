# YOLO-Master CoreML Runner 1.2.0

The macOS runner catches up with the v1.2.0 Linux runtime: it now measures itself the same way
(benchmark mode with the shared `yolomaster-bench/v1` document and an on-device mAP), tracks
objects across video frames (BoT-SORT / ByteTrack) and builds the model input on the GPU. The
numeric parts are not ports: the mAP scorer, the tracker and the benchmark statistics are the
Linux runtime's own C++ code, compiled into the Swift package as the portable core
(`Sources/YOLOMasterCore`) and called through a C header, so a number produced here is the number
the Linux CLI would produce for the same detections.

## New: Bench (sidebar section) and `--bench` / `--accuracy` (CLI)

- **Cold sweep**: warm-up then timed Core ML predictions on a gray-114 probe at the model's input
  size (`inferOnly`, the "model-only" latency), reported as floor-rank median / p90 / p95 / p99,
  the convention the phone Bench tabs and the Linux CLI already use.
- **Sustained**: a timed loop (0.5 to 10 minutes) with one median per second, the throttle
  percentage (slowest-quarter median vs the cold median) and the thermal state sampled once per
  second (`sustained.thermal`, macOS / iOS only).
- **Accuracy**: choose the `images` folder of a labelled set (labels next to it in `labels/`, the
  ultralytics layout, or the images -> labels rule per file); the runner runs the val protocol
  (conf 0.001, IoU 0.7, max_det 300, multi-label) and prints mAP50 / mAP50-95 and the per-class
  table. The first Core ML mAP number of this project.
- **Save JSON**: the document (`tool: "macos"`, model card with compute unit and preprocessing
  device, environment from `sysctl` and Metal, protocol, cold / sustained / dataset / accuracy
  blocks) validates with `scripts/bench_schema_check.py` like every other producer's.
- CLI: `--bench cold|sustained`, `--bench-iters`, `--bench-warmup`, `--bench-minutes`,
  `--bench-json`, `--accuracy auto|LABELS_DIR`, `--limit N`; the Linux `[summary]`, `[bench]` and
  `[accuracy]` lines. The old `--benchmark [--iters N]` still works as an alias of `--bench cold`.

## New: `--save-txt` (CLI)

One file per image or video frame, `class conf x1 y1 x2 y2 [track_id]` in original-image
pixels, numbers printed exactly like the C++ CLI (`%g`, six significant digits). The dumps score
with `scripts/eval_map.py`, `yolomaster_score` and the parity tools like a Linux dump; the
in-process accuracy is computed on the same rounded values, so `[accuracy]` equals scoring the
`--save-txt` dump.

## New: Tracking (Detection section, video) and `--track` (CLI)

- **ByteTrack** (Kalman on xyah, two-stage IoU association) and **BoT-SORT** (Kalman on xywh
  plus camera-motion compensation), ultralytics `bytetrack.yaml` / `botsort.yaml` defaults, no
  appearance model. Ids are drawn as `#id` with a per-id colour and carried into the exported
  video; `--save-txt` gains the 7th column.
- In the app, tracking is a pure function of the cached candidates: the camera motion is
  recorded once while the video is inferred, so switching the tracker on, changing it or moving
  the confidence / IoU sliders recomputes the tracks in a moment without touching the video.
- BoT-SORT's camera motion on Apple platforms comes from Vision's translational image
  registration (the previous frame registered onto the current one, downscaled); the Linux
  runtime uses sparse optical flow plus a RANSAC partial affine. The two agree on pans and
  handheld drift; rotation and zoom between two frames are not compensated here.

## New: GPU preprocessing (Preprocess section, "Device")

A Metal compute kernel does the letterbox (bilinear, the same source-coordinate rule as
`cv::resize` and the Linux CUDA kernel, integer pads), the BGRA / RGBA -> planar RGB conversion
and the / 255 straight into a shared-storage buffer that Core ML reads as its input tensor. Images
upload their decoded bytes as they are; the live camera's pixel buffers are read through the
Metal texture cache without a copy. The Core Graphics + vDSP path stays selectable ("CPU",
`--cpu-preproc`) and the stats panel shows the preprocess and postprocess stages next to the
model time. `--dump-input DIR` writes the raw input tensors; `scripts/preproc_compare.py` checks
them against the Linux reference within 1/255.

## Test battery: `mac/tests/run_mac_tests.sh`

Run on a Mac: build, the core handshake (`--core-selftest`), bench JSON schema, txt format,
in-process accuracy equal to `eval_map_standalone.py` on the dump, tracking persistence on a
synthetic pan clip, the camera-motion axis check on a diagonal pan, Metal tensor parity and the
mAP agreement between the two preprocessing devices.

## Notes

- The iOS app consumes the same Kit and is unchanged: every Kit addition is additive and the
  preprocessing device defaults to CPU inside the Kit (the macOS CLI and app opt into Metal).
- The accuracy pass takes a detection model; the bundled `v0.1-seg-N` runs the bench and
  tracking but not the accuracy pass (segmentation candidates at conf 0.001 are too many).
- Bench and accuracy always run single-pass at the model input size; slicing is not part of
  the protocol.
