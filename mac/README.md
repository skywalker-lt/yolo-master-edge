# YOLO-Master-EsMoE-N — Mac (Core ML) runner

A first Swift runner for **YOLO-Master-EsMoE-N** on the **Mac** via **Core ML**. It loads a
`.mlpackage`, letterboxes an image exactly as the C++/ONNX pipeline does, runs inference on the
Apple Silicon ANE/GPU, and decodes with multi-label + per-class NMS. Target: M-series Mac, macOS 14+.

> The same `.mlpackage` + decode also run on iOS; an iPhone target (Vision/`VNCoreMLRequest`, camera
> capture) is a separate future sibling to this command-line Mac runner.

> **Status:** direct CoreML conversion of EsMoE-N is currently **blocked** — coremltools can't lower an
> in-place op in the custom MoE/attention backbone (ONNX and TensorRT both handle it). Until that's
> resolved, the **working Mac path is ONNX Runtime + the CoreML Execution Provider** (below), which runs
> our already-validated `.onnx` directly and lets ORT offload supported subgraphs to the ANE/GPU. The
> Swift/`MLModel` runner here is correct the day the `.mlpackage` converts.

## Working path today — C++ + ONNX Runtime (CPU + CoreML EP)
```bash
bash mac/build_ort_macos.sh      # Homebrew OpenCV + ORT osx-arm64 (CoreML EP built in), builds the runner
# then A/B on a folder of images:
cpp/build_mac/yolomaster_edge --model esmoe_n_visdrone_sim.onnx --source <dir> --device cpu    --no-save --quiet
cpp/build_mac/yolomaster_edge --model esmoe_n_visdrone_sim.onnx --source <dir> --device coreml --no-save --quiet
```
`--device coreml` appends ORT's CoreML EP (`MLComputeUnits=CPUAndGPU` — the GPU handles this model's graph
fragmentation better than the ANE). Compare the `[summary] model-FPS`. Expect CPU to already be near
real-time on an M-series chip; the CoreML EP may or may not beat it depending on how the graph partitions
(this MoE+attention model fragments, so it's an empirical call — measure, don't assume).


## 1. Export the model (once, in the training env)
```bash
pip install coremltools           # + ultralytics already installed
python export_coreml.py --weights EsMoE-N_VisDrone.pt --imgsz 640 --out EsMoE-N.mlpackage
```
The exporter forces the model's **dense** path during trace (it is gated on
`torch.onnx.is_in_onnx_export()`, which Core ML export does not trigger — otherwise the sparse MoE
routing is captured and conversion fails). It converts with a **float32 tensor input** `[1,3,640,640]`
so the Swift side owns preprocessing, and embeds class names + the output tensor name as metadata.

## 2. Build
```bash
swift build -c release           # or: open Package.swift in Xcode
```

## 3. Run
```bash
.build/release/YOLOMasterCoreML --model EsMoE-N.mlpackage --source img.jpg \
    --conf 0.25 --iou 0.5 --out out.jpg
```
Prints per-detection lines (`class conf [x1 y1 x2 y2]`) and writes an annotated `out.jpg`.
Core ML dispatches across ANE/GPU/CPU automatically (`computeUnits = .all`).

## Notes
- **Preprocessing parity:** aspect-preserving letterbox to 640, 114 gray pad, RGB, `/255`, NCHW —
  the same as the ONNX/TensorRT runners, so detections should match.
- **Decode parity:** output `[1, 4+nc, anchors]`; multi-label (one detection per class above `conf`
  per anchor) + per-class greedy NMS, capped at 300 — mirrors `cpp/src/common.cpp`.
- **UMA:** on Apple Silicon the preprocess/decode buffers and the model share one address space, so
  there's no host↔device copy. For a ~2.7 M-param model this is a latency nicety, not a capacity need.

## First-runner caveats (verify on-device, iterate)
- The letterbox and draw use CoreGraphics with an explicit top-down flip; **eyeball the first `out.jpg`**
  for correct box placement / no vertical mirroring before trusting numbers.
- No on-image text labels yet (boxes only); class names are printed to stdout.
- If the export names the output tensor something unexpected, the runner reads it from model metadata
  (`output`); it also prints the resolved input/output names at startup so mismatches are visible.
- For rigorous accuracy, dump predictions and score with `scripts/eval_map_standalone.py` as on the
  other platforms (a `--save-txt` mode is the natural next addition).
