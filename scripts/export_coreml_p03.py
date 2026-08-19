#!/usr/bin/env python
"""Project03 iPhone path: pruned v0.1-N COCO -> Core ML .mlpackage, fp16 + INT8-W8.

Runs in /data/tmp/venv-coreml (py3.11, torch 2.5.1, coremltools 9.0) with
PYTHONPATH=/data/YOLO-Master. Linux converts and weight-quantizes; ACTIVATION
quantization (A8) needs a Core ML runtime and therefore runs on the Mac side:
see mac/quantize_a8_p03.py, which consumes the fp16 package produced here.

Deployment context (from the phase-2 study): this model is activation-bound, so
W8 buys size/bandwidth (half the weight bytes, ANE-friendly) rather than big
latency; A8 is the experiment for actual speed. ANE is fp16-native, so fp16 is
the expected latency baseline on iPhone as it is on every GPU we measured.

Outputs (models/p03_coreml/):
  p03_v01n_coco_fp16.mlpackage      baseline, ANE-ready
  p03_v01n_coco_w8.mlpackage        INT8 weights, fp16 activations
  p03_ref.npz                       input blob + torch reference output (parity)

Usage:
  PYTHONPATH=/data/YOLO-Master /data/tmp/venv-coreml/bin/python \
      scripts/export_coreml_p03.py \
      --model runs/project03/pod-prune/v01-n-coco/YOLO-Master-v0.1-N_pruned_t0.10.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from unittest import mock

import numpy as np
import torch
import torch.nn as nn


class DetectOnly(nn.Module):
    """Trace wrapper: single detection tensor out (ONNX-consistent semantics)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models/p03_coreml")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = (ck.get("model") if isinstance(ck, dict) else ck).float().eval()
    head = net.model[-1]
    head.export = True
    head.format = "onnx"          # same single-tensor semantics as every other export

    wrapper = DetectOnly(net).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    # the MoE sparse path uses index_add_ (no MIL mapping); the class ships a
    # dense stack+gather branch gated on is_in_onnx_export - trace under a mock,
    # the same trick export_coreml_mixture.py uses. Exact at k==E (pruned: 2/2).
    with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        ref_out = wrapper(dummy).numpy()
        traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, args.imgsz, args.imgsz))],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    fp16_path = out / "p03_v01n_coco_fp16.mlpackage"
    mlmodel.save(str(fp16_path))
    print(f"[coreml] {fp16_path}")

    w8_cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    w8 = linear_quantize_weights(mlmodel, config=w8_cfg)
    w8_path = out / "p03_v01n_coco_w8.mlpackage"
    w8.save(str(w8_path))
    print(f"[coreml] {w8_path}")

    # parity reference: fixed random image-like blob + torch output
    rng = np.random.default_rng(0)
    blob = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        ref = wrapper(torch.from_numpy(blob)).numpy()
    np.savez_compressed(out / "p03_ref.npz", input=blob, output=ref)
    print(f"[coreml] p03_ref.npz (zero-input ref max |out| {np.abs(ref_out).max():.3f})")


if __name__ == "__main__":
    main()
