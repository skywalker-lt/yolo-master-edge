#!/usr/bin/env python3
"""Export a top-down whole-body pose model (DWPose / RTMPose, SimCC head) to Core ML .mlpackage.

These models do not find people: they take ONE person crop and return that person's
keypoints. The apps run a detection model first and this one once per person
(YOLOMasterKit `Detector.isTopDown`). The package this script writes is what the Kit expects:

    input   "image"      [1, 3, H, W]   RGB, 0..1, one person crop (DWPose-t: H 256, W 192)
    output  "keypoints"  [1, K, 3]      x, y in crop pixels and a score per keypoint
    metadata             task=pose-topdown, kpt_shape="K,3", names=person, kpt_score

The mean / std normalisation and the SimCC decoding live inside the package: x and y are the
argmax of the two classification vectors divided by the split ratio (2 bins per pixel), the
score is the smaller of the two peak values (mmpose `get_simcc_maximum`). The score is NOT a
probability and its scale differs from model to model, so the package carries the score to
draw a keypoint from (`--kpt-score`, metadata `kpt_score`; the Kit assumes 0.3 without it).
Measured on COCO-WholeBody val behind yolo26n person boxes, as the score that keeps 95% of
the correctly placed body keypoints: DWPose-t 0.3, DWPose-m and DWPose-l 0.45, RTMW-l at
384x288 2.8 (README.md has the table).

The source is the ONNX file OpenMMLab publishes for the checkpoint (no mmpose / mmcv install
needed), turned back into a PyTorch graph with onnx2torch and traced for coremltools:

    # DWPose-t, 133 COCO-WholeBody keypoints, 5.5 M parameters
    wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip
    unzip rtmpose-t_*.zip
    python export_dwpose.py --onnx 20230831/rtmpose_onnx/rtmpose-t_*/end2end.onnx \
        --target ios17 --out dwpose-t-wholebody.mlpackage

Needs, on top of the export environment in README.md: `pip install onnx onnxruntime onnx2torch`
(install them with --no-deps, or into a separate directory on PYTHONPATH, so that the pinned
torch and numpy stay as they are). onnxruntime is only used for the check at the end.
"""
import argparse
import sys

import numpy as np
import torch

MEAN = [123.675, 116.28, 103.53]     # mmpose's ImageNet statistics, on 0..255 RGB
STD = [58.395, 57.12, 57.375]
SPLIT_RATIO = 2.0                    # SimCC bins per pixel


def load_core(onnx_path: str):
    """The published ONNX graph as a PyTorch module (input: normalised crop; output: simcc_x, simcc_y)."""
    import onnx
    from onnx2torch import convert
    m = onnx.load(onnx_path)
    for n in m.graph.node:
        # ScaleNorm's clamp(min=eps) is exported as Clip(x, min, "") - an omitted max that
        # onnx2torch reads as a missing constant
        if n.op_type == "Clip":
            while len(n.input) > 1 and n.input[-1] == "":
                n.input.pop()
    for v in list(m.graph.input) + list(m.graph.output):    # a fixed batch of one
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    return convert(m).eval(), shape


class TopDownPose(torch.nn.Module):
    """RGB 0..1 crop -> [1, K, 3] (x, y in crop pixels, score)."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) / 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) / 255.0)

    def forward(self, image):
        sx, sy = self.core((image - self.mean) / self.std)
        mx, ix = sx.max(dim=2)
        my, iy = sy.max(dim=2)
        return torch.stack((ix.to(sx.dtype) / SPLIT_RATIO, iy.to(sy.dtype) / SPLIT_RATIO,
                            torch.minimum(mx, my)), dim=2)


def _deploy_target(name: str):
    import coremltools as ct
    return {"macos13": ct.target.macOS13, "macos14": ct.target.macOS14, "macos15": ct.target.macOS15,
            "ios16": ct.target.iOS16, "ios17": ct.target.iOS17}[name]


def check_against_onnxruntime(model, onnx_path: str, shape) -> None:
    """The wrapper must give the keypoints ONNX Runtime gives for the same crop."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[check] onnxruntime not installed - skipped")
        return
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    rs = np.random.RandomState(0)
    worst_xy, worst_score = 0.0, 0.0
    for _ in range(5):
        img = rs.rand(*shape).astype(np.float32)
        norm = (img * 255 - np.array(MEAN, np.float32).reshape(1, 3, 1, 1)) / np.array(STD, np.float32).reshape(1, 3, 1, 1)
        sx, sy = sess.run(None, {sess.get_inputs()[0].name: norm.astype(np.float32)})
        ref = np.stack((sx.argmax(2) / SPLIT_RATIO, sy.argmax(2) / SPLIT_RATIO,
                        np.minimum(sx.max(2), sy.max(2))), axis=2)
        with torch.no_grad():
            got = model(torch.from_numpy(img)).numpy()
        worst_xy = max(worst_xy, float(np.abs(got[..., :2] - ref[..., :2]).max()))
        worst_score = max(worst_score, float(np.abs(got[..., 2] - ref[..., 2]).max()))
    print(f"[check] PyTorch graph vs ONNX Runtime, 5 random crops: max |dx, dy| = {worst_xy} px, "
          f"max |dscore| = {worst_score:.1e}")
    if worst_xy > 0.5 or worst_score > 1e-3:
        raise RuntimeError("the rebuilt graph does not reproduce the ONNX model")


def export(onnx_path: str, out: str, target: str = "ios17", source: str = "", kpt_score: float = 0.3) -> dict:
    import coremltools as ct
    core, shape = load_core(onnx_path)
    if len(shape) != 4 or shape[1] != 3:
        raise RuntimeError(f"expected a [1, 3, H, W] input, the ONNX declares {shape}")
    model = TopDownPose(core).eval()
    check_against_onnxruntime(model, onnx_path, shape)
    example = torch.rand(*shape)
    with torch.no_grad():
        k = int(model(example).shape[1])
    traced = torch.jit.trace(model, example, strict=False)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=tuple(shape))],
                    outputs=[ct.TensorType(name="keypoints")],
                    minimum_deployment_target=_deploy_target(target), convert_to="mlprogram")
    meta = ml.user_defined_metadata
    meta["task"] = "pose-topdown"
    meta["kpt_shape"] = f"{k},3"
    meta["imgsz"] = f"{shape[2]}x{shape[3]}"       # height x width of the person crop
    meta["precision"] = "fp16"
    meta["names"] = "person"
    meta["kpt_score"] = f"{kpt_score:g}"          # keypoints scoring below this are not drawn
    if source:
        meta["source"] = source
    ml.short_description = (f"Top-down pose: one person crop [1,3,{shape[2]},{shape[3]}] RGB 0..1 in, "
                            f"keypoints [1,{k},3] (x, y in crop pixels, score) out.")
    ml.save(out)
    return {"out": out, "crop": f"{shape[2]}x{shape[3]}", "keypoints": k, "kpt_score": kpt_score}


def main():
    ap = argparse.ArgumentParser(description="Export a top-down SimCC pose model (DWPose / RTMPose ONNX) to Core ML")
    ap.add_argument("--onnx", required=True, help="the model's end2end.onnx (outputs simcc_x, simcc_y)")
    ap.add_argument("--out", required=True, help="output .mlpackage path")
    ap.add_argument("--target", default="ios17", choices=["macos13", "macos14", "macos15", "ios16", "ios17"])
    ap.add_argument("--source", default="", help="free text stamped into the package (which checkpoint this is)")
    ap.add_argument("--kpt-score", type=float, default=0.3,
                    help="keypoint score to draw from, stamped as kpt_score (model-specific, see the header)")
    a = ap.parse_args()
    try:
        r = export(a.onnx, a.out, a.target, a.source, a.kpt_score)
        print(f"OK  {r['out']}  task=pose-topdown  crop={r['crop']}  keypoints={r['keypoints']}  "
              f"kpt_score={r['kpt_score']:g}")
    except Exception as e:
        print(f"FAIL  {a.onnx}  {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
