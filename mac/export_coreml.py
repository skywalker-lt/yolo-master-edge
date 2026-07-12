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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="EsMoE-N_VisDrone.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="EsMoE-N.mlpackage")
    a = ap.parse_args()

    ym = YOLO(a.weights)
    model = ym.model.eval()
    # the model's OWN class names (works for any dataset: VisDrone, COCO, custom, ...)
    names = [ym.names[i] for i in sorted(ym.names)] if getattr(ym, "names", None) else []
    # Detect head -> export mode: emit the single concatenated [1, 4+nc, anchors] tensor
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "coreml"

    # Core ML fix: ES_MOE._compute_load_balancing_loss writes training telemetry buffers in-place
    # (load_balancing_loss/expert_usage_counts .copy_(...)). Those aten::copy_ ops make coremltools'
    # tensor-assignment pass fail ("No matching select or slice"); ONNX/TRT tolerate them. The value is
    # training-only aux loss, unused at inference, so replace it with a no-op for the trace.
    from ultralytics.nn.modules.moe.modules import ES_MOE
    ES_MOE._compute_load_balancing_loss = (
        lambda self, routing_weights, eps=1e-6: torch.zeros((), device=routing_weights.device)
    )

    ex = torch.zeros(1, 3, a.imgsz, a.imgsz)
    # check_trace=False: the Detect head caches its anchor grid on the first pass, so trace's
    # double-run consistency check trips on a benign graph diff. The first-pass graph bakes the
    # correct anchors for the fixed imgsz, which is what we want.
    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        traced = torch.jit.trace(model, ex, strict=False, check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, a.imgsz, a.imgsz), dtype=float)],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    out_name = mlmodel.get_spec().description.output[0].name
    if names:
        mlmodel.user_defined_metadata["names"] = ",".join(names)
    mlmodel.user_defined_metadata["imgsz"] = str(a.imgsz)
    mlmodel.user_defined_metadata["output"] = out_name
    mlmodel.save(a.out)
    print(f"saved {a.out}")
    print(f"  input : images  [1,3,{a.imgsz},{a.imgsz}]  (feed 0-1 RGB, NCHW)")
    print(f"  output: {out_name}  [1,{4 + len(names)},anchors]  ({len(names)} classes: {','.join(names[:6])}{'…' if len(names) > 6 else ''})")


if __name__ == "__main__":
    main()
