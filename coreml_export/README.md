# Core ML export

`export_coreml.py` converts an Ultralytics / YOLO-Master `.pt` checkpoint (detector **or**
segmenter) into a Core ML `.mlpackage` carrying the exact metadata the [macOS app](../mac/) reads
(`names`, `imgsz`, `output` tensor, `task`, and — for segmentation — `proto`/`nm`). It emits an
mlprogram and validates the class count against the output shape.

Conversion runs on **Linux** (coremltools; only *prediction* needs macOS).

## Environment

```bash
conda create -n cmlexport python=3.11 -y
pip install torch==2.5.1 torchvision==0.20.1 coremltools==9.0
```

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
```

## What the script handles (and why)

- **YOLO-Master MoE** — forces the dense path (`is_in_onnx_export=True`), constant-folds shapes
  (`jit.freeze` + `run_frozen_optimizations`, fixing the dynamic `aten::Int`), and no-ops EsMoE's
  in-place aux-loss telemetry (fixing `aten::copy_` "No matching select or slice"). All no-ops for plain models.
- **Segmentation** — detects the 2-output signature and writes `task=segment` + `proto`/`nm`.
- **Area attention (YOLOv12)** — an eager warmup bakes each layer's concrete spatial dims so the
  area reshapes fold to static shapes; `--yolov12-aattn` swaps in the qk+v AAttn the authors' weights expect.
- **LoRA** — `--merge-lora-dir` merges trained adapters before export (a merged LoRA is a static graph;
  routed MoLoRA cannot be traced). Note: `apply_lora` may re-initialize the detection head — restore it
  from the base before merging if you apply LoRA fresh.
