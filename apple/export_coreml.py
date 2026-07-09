#!/usr/bin/env python3
"""Export YOLO-Master-EsMoE-N to Core ML (.mlpackage) for the Swift runner.

Two model-specific points:
- EsMoE-N switches to its DENSE compute path only under `torch.onnx.is_in_onnx_export()`.
  Core ML export traces via torch.jit, which does NOT set that flag, so we patch it to True
  during the trace — otherwise the sparse MoE routing (data-dependent) gets captured and the
  conversion fails or produces a garbage graph.
- We convert with a float32 TENSOR input [1,3,640,640] (not an image input), so the Swift side
  owns the letterbox exactly as the C++/ONNX pipeline does (aspect-preserving, 114 pad, /255, RGB).

Run in the training env (ultralytics + coremltools installed):
    python export_coreml.py --weights EsMoE-N_VisDrone.pt --imgsz 640 --out EsMoE-N.mlpackage
"""
import argparse
from unittest import mock

import torch
import coremltools as ct
from ultralytics import YOLO

VISDRONE = ["pedestrian", "people", "bicycle", "car", "van", "truck",
            "tricycle", "awning-tricycle", "bus", "motor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="EsMoE-N_VisDrone.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="EsMoE-N.mlpackage")
    a = ap.parse_args()

    model = YOLO(a.weights).model.eval()
    # Detect head -> export mode: emit the single concatenated [1, 4+nc, anchors] tensor
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "coreml"

    ex = torch.zeros(1, 3, a.imgsz, a.imgsz)
    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        traced = torch.jit.trace(model, ex, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, a.imgsz, a.imgsz), dtype=float)],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    out_name = mlmodel.get_spec().description.output[0].name
    mlmodel.user_defined_metadata["names"] = ",".join(VISDRONE)
    mlmodel.user_defined_metadata["imgsz"] = str(a.imgsz)
    mlmodel.user_defined_metadata["output"] = out_name
    mlmodel.save(a.out)
    print(f"saved {a.out}")
    print(f"  input : images  [1,3,{a.imgsz},{a.imgsz}]  (feed 0-1 RGB, NCHW)")
    print(f"  output: {out_name}  [1,{4+len(VISDRONE)},anchors]")


if __name__ == "__main__":
    main()
