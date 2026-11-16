# Core ML export

`export_coreml.py` converts an Ultralytics / YOLO-Master `.pt` checkpoint (detector **or**
segmenter) into a Core ML `.mlpackage` carrying the exact metadata the [macOS app](../mac/) reads
(`names`, `imgsz`, `output` tensor, `task`, and — for segmentation — `proto`/`nm`). It emits an
mlprogram and validates the class count against the output shape.

Conversion runs on **Linux** (coremltools; only *prediction* needs macOS).

## Environment

```bash
conda create -n cmlexport python=3.11 -y
pip install torch==2.5.1 torchvision==0.20.1 coremltools==9.0 "numpy<2.3"
```

`numpy 2.4` breaks coremltools 9.0's torch frontend the same way a newer torch does
(`TypeError: only 0-dimensional arrays can be converted to Python scalars` on `aten::Int`): it
no longer converts a one-element array to a scalar. Pin **numpy below 2.3**.

Then install the `ultralytics` build that **matches the checkpoint** — mixing them fails at trace time:

| Checkpoint | ultralytics |
| :--------- | :---------- |
| YOLO-Master (v0.1 / EsMoE / UoMoE, incl. P2, seg) | the fork: `pip install -e /path/to/YOLO-Master --no-deps` |
| stock YOLO (yolo11, …) | `pip install ultralytics` |
| sunsmarterjie/yolov12 | stock ultralytics **+** `--yolov12-aattn` (see below) |

`torch 2.11` breaks coremltools' torch frontend (`aten::Int`) — pin **torch 2.5**.

## Usage

```bash
# detector / segmenter (task auto-detected from the output count)
python export_coreml.py --weights model.pt --imgsz 640 --out model.mlpackage

# yolov12 authors' checkpoints (split qk+v area-attention)
python export_coreml.py --weights yolov12x.pt --imgsz 640 --out yolov12x.mlpackage --yolov12-aattn

# a LoRA-fine-tuned model: merge the trained adapters, then export
python export_coreml.py --weights base.pt --merge-lora-dir lora_adapter/ --imgsz 640 --out ft.mlpackage

# stock ultralytics models for the iOS app (iOS 17 target); a pose model is detected from its head
python export_coreml.py --weights yolo11n.pt --imgsz 640 --target ios17 --out yolo11n.mlpackage
python export_coreml.py --weights yolo26n-pose.pt --imgsz 640 --target ios17 --anchors --out yolo26n-pose.mlpackage

# top-down whole-body pose (DWPose-t, 133 keypoints): a separate script, from the published ONNX
python export_dwpose.py --onnx rtmpose-t_.../end2end.onnx --target ios17 --out dwpose-t-wholebody.mlpackage
```

`export_dwpose.py` covers the top-down SimCC pose models (DWPose / RTMPose). They take one
person crop, not a frame, so the package is a different contract: input `image`
`[1, 3, 256, 192]` (RGB, 0..1), output `keypoints` `[1, 133, 3]` (x, y in crop pixels and a
score), `task=pose-topdown`. Normalisation and SimCC decoding are inside the package; the app
runs a person detector in front of it. The script header has the download link and the three
extra packages it needs (`onnx`, `onnxruntime`, `onnx2torch`).

Any model of that family exports the same way and runs in the app unchanged; the crop size is
read from the package. Their keypoint scores are on different scales, so pass the score to
draw from as `--kpt-score` (stamped as `kpt_score`). Measured on 200 COCO-WholeBody val images
behind stock yolo26n person boxes, in PyTorch: share of keypoints within 20% of the hand's
size (10% of the face's size) of the label, and the score that keeps 95% of the correctly
placed body keypoints. The Kit draws body keypoints from that score and gates the feet, the
face and each hand as whole parts at multiples of it (`Detector.poseKeypoints`).

| ONNX (download.openmmlab.com/mmpose/v1/projects/...) | crop | size | GFLOPs | hands | face | `--kpt-score` |
| :-- | :-- | --: | --: | --: | --: | --: |
| DWPose-t `rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip` | 256x192 | 11 MB | 0.5 | 62% | 97% | 0.3 |
| DWPose-m `rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip` | 256x192 | 36 MB | 2.2 | 74% | 98% | 0.45 |
| DWPose-l `rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip` | 384x288 | 67 MB | 10 | 82% | 99% | 0.45 |
| RTMW-l `rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip` | 384x288 | 115 MB | 17.7 | 85% | 99% | 2.8 |

## What the script handles (and why)

- **YOLO-Master MoE** — forces the dense path (`is_in_onnx_export=True`), constant-folds shapes
  (`jit.freeze` + `run_frozen_optimizations`, fixing the dynamic `aten::Int`), and no-ops EsMoE's
  in-place aux-loss telemetry (fixing `aten::copy_` "No matching select or slice"). All no-ops for plain models.
- **Segmentation** — detects the 2-output signature and writes `task=segment` + `proto`/`nm`.
- **Pose** — a head with a `kpt_shape` writes `task=pose` + `kpt_shape` (`17,3` for COCO). The
  keypoints ride in the detection tensor after the class scores, `[1, 4+nc+17*3, anchors]`,
  already decoded by the head to input pixels (x, y) and a 0..1 visibility. Checked on
  yolo26n-pose: decoding that tensor the way the app does reproduces `predict()` on COCO images
  (same people, box IoU 0.95+, joints within 1-2 px).
- **End-to-end heads (YOLO26, YOLOv10)** — `end2end` is always stamped (`true` / `false`), so the
  app does not have to guess the layout from the shape. `--anchors` exports such a checkpoint
  through its one-to-many head in the classic layout with NMS in the app: the app runs the
  end-to-end layout on the CPU only (its in-graph top-k tail does not compile for the GPU), the
  anchors layout on the GPU and the Neural Engine.
- **Area attention (YOLOv12)** — an eager warmup bakes each layer's concrete spatial dims so the
  area reshapes fold to static shapes; `--yolov12-aattn` swaps in the qk+v AAttn the authors' weights expect.
- **LoRA** — `--merge-lora-dir` merges trained adapters before export (a merged LoRA is a static graph;
  routed MoLoRA cannot be traced). Note: `apply_lora` may re-initialize the detection head — restore it
  from the base before merging if you apply LoRA fresh.

## Precision

`ct.convert(..., convert_to="mlprogram")` emits fp16 weights and fp16 activations (coremltools'
default `compute_precision` for ML programs); the packages here are fp16 already, no separate
step (v0.1-N: 7.55 M parameters, 15.5 MB on disk; yolov12x: 59.4 M, 119 MB).

`quantize_w8a16.py in.mlpackage out.mlpackage` turns an existing package into W8A16 (int8
per-channel weights, fp16 activations). The package halves: yolov12x 119 -> 60 MB, RTMW-l
115 -> 58 MB, yolo26n 4.9 -> 2.5 MB.

Only conv / linear weights are quantized. Until 2026-11-12 this script and the Project03
recipe (`scripts/export_coreml_p03.py`, `mac/quantize_a8_p03.py`) used a global config, which
also rounds the detection head's anchor grid and stride table to int8 with one scale per
table - every box moves, and mAP50-95 falls by about 0.10 (yolo26n 0.403 -> 0.296, v0.1-N
0.425 -> 0.320). A `p03_v01n_coco_w8.mlpackage` built before that date has this fault.

Effect of the fixed recipe on the full COCO val2017 set, simulated in PyTorch (the fused
model's weights rounded exactly as the package holds them, then ultralytics' validation;
Core ML cannot predict on Linux, so the packages themselves were not run):

| model | metric | fp16 | W8A16 | fp16 size | W8A16 size |
| :-- | :-- | --: | --: | --: | --: |
| yolo26n | box mAP50-95 | 0.4034 | 0.4012 | 5.1 MB | 2.7 MB |
| yolo11n | box mAP50-95 | 0.3878 | 0.3872 | 5.5 MB | 2.9 MB |
| yolo12n | box mAP50-95 | 0.4016 | 0.4012 | 5.5 MB | 3.0 MB |
| yolov5nu | box mAP50-95 | 0.3374 | 0.3369 | 5.5 MB | 2.9 MB |
| YOLO-Master v0.1-N | box mAP50-95 | 0.4245 | 0.4232 | 15.5 MB | 8.2 MB |
| yolo26n-pose | pose mAP50-95 | 0.5686 | 0.5649 | 6.2 MB | 3.3 MB |
| YOLO-Master v0.1-seg-N | mask mAP50-95 (500 val images, via its ONNX export) | 0.3605 | 0.3594 | 6.2 MB | 3.4 MB |
| RTMW-l 384x288 | hand keypoints within 20% of the hand's size | 83.1% | 83.0% | 114.9 MB | 57.7 MB |

Sizes are the whole package on disk. Speed is not expected to change: the weights are expanded
to fp16 when the model loads and the arithmetic is the same; that is not measured here.

## Neural Engine and the MoE router (`--no-ane-safe-moe` to opt out)

The stock export of `OptimizedMOEImproved` (the v0.1 family) and `OptimizedMOE` keeps the router's
`torch.topk` and a `torch.gather` over the stacked expert outputs in the graph: MIL `topk`,
`gather_along_axis` and int32 index tensors. The ANE has no kernels for those, so under
`cpuAndNeuralEngine` Core ML runs every router on the CPU with a sync and a copy on both sides
of each MoE block (v0.1-N: 9.2 ms on the ANE against 4.1 ms for the router-free v0.1-seg-N graph
on an M4 Max, while the GPU ran it in 4.3 ms).

By default the exporter now writes the same top-k mixture with arithmetic only:
`rank_i = sum_j step(p_j - p_i)`, `mask_i = step(k - 0.5 - rank_i)`, `w = p * mask / sum(p * mask)`,
`y = shared(x) + sum_i w_i * expert_i(x)`, with `step(z) = clamp(32768 z, 0, 1)`. The graph has no
integer op left (`moe_export: ane_safe_rank` in the metadata); eager fp32 outputs agree with the
stock path to 3e-4 over a 0..637 output range on the v0.1-N checkpoint, and two experts whose
pooled probabilities are within 3e-5 of each other (an fp16 tie) get a proportional blend instead
of an arbitrary winner.

Note what the export cannot change: a static Core ML graph computes EVERY expert and masks the
mixture, so v0.1-N runs its 4 + 8 + 16 experts on every frame. The sparse top-2 saving exists
only in the PyTorch eager path.

`compute_plan.py model.mlpackage --units ane` (macOS 14.4+, coremltools 8+) prints which device
Core ML gives each op and lists the ops the ANE cannot take, with their estimated cost share.
It is the text form of Xcode's performance report and the way to check whether an "ANE" number
from the Bench is the Neural Engine or a CPU fallback.
