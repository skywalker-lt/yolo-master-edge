#!/usr/bin/env python3
"""Validate exported mixture models (scripts/export_mixture.py output) against the edge
runtime. Runs in the edge environment (numpy + onnxruntime; no ultralytics needed).

Three levels per model:
  A. torch-vs-pipeline: the eager predict reference (<model>.ref.npz, torch ground truth)
     is IoU-matched against the C++ CLI's --save-txt boxes on the same images. Validates
     the FULL-MODEL export numerics + edge decode + NMS end to end (upstream's own
     evidence stops at block-level roundtrips).
  B. python-vs-C++ decode: onnxruntime here, with a numpy replica of the edge letterbox
     and decode (both layouts: [1,4+nc,anchors] and end2end [1,N,6]), tightly matched
     against the CLI boxes. Isolates the C++ decode path at ~pixel tolerance.
  C. feature smoke via the CLI: metadata nc/imgsz line, sparse slicing line, CW-NMS tag,
     label-export instance count == detection count.

Backends: onnx always; mnn/ncnn when a sibling <stem>.mnn / <stem>_ncnn exists (det-count
parity vs the onnx backend, T6-style).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def letterbox_params(w, h, imgsz):
    r = min(imgsz / w, imgsz / h)
    nw, nh = round(w * r), round(h * r)
    return r, (imgsz - nw) // 2, (imgsz - nh) // 2


def preprocess(img_path, imgsz):
    import cv2
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]
    r, px, py = letterbox_params(w, h, imgsz)
    resized = cv2.resize(bgr, (round(w * r), round(h * r)))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), (r, px, py, w, h)


def decode_np(out, lb, conf):
    """numpy replica of the edge decode for both layouts -> (xyxy, conf, cls)."""
    r, px, py, w, h = lb
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] == 6 and out.shape[0] >= 32:          # end2end [N, 6]
        keep = out[:, 4] >= conf
        d = out[keep]
        xyxy = (d[:, :4] - [px, py, px, py]) / r
        return xyxy, d[:, 4], d[:, 5].astype(int)
    feat, anchors = out.shape                               # [4+nc, anchors]
    scores = out[4:, :]
    cls = scores.argmax(0)
    sc = scores.max(0)
    keep = sc >= conf
    cx, cy, bw, bh = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
    x0 = (cx - bw / 2 - px) / r
    y0 = (cy - bh / 2 - py) / r
    xyxy = np.stack([x0, y0, x0 + bw / r, y0 + bh / r], 1)
    return xyxy, sc[keep], cls[keep]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    ar_a = np.prod(a[:, 2:] - a[:, :2], 1)
    ar_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def match_rate(ref_xyxy, ref_cls, got_xyxy, got_cls, iou_thr=0.8):
    """fraction of ref boxes with a same-class IoU>=thr counterpart (greedy)."""
    if len(ref_xyxy) == 0:
        return 1.0
    m = iou_matrix(ref_xyxy, got_xyxy)
    m[ref_cls[:, None] != got_cls[None, :]] = 0
    matched = 0
    used = set()
    for i in np.argsort(-m.max(1) if m.size else []):
        j_order = np.argsort(-m[i]) if m.size else []
        for j in j_order:
            if m[i, j] < iou_thr:
                break
            if j not in used:
                used.add(j)
                matched += 1
                break
    return matched / len(ref_xyxy)


def read_savetxt(path):
    rows = [l.split() for l in open(path)] if Path(path).exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    a = np.array(rows, dtype=float)
    return a[:, 2:6], a[:, 1], a[:, 0].astype(int)


def run_cli(bin_, model, src, extra, tmp):
    cmd = [bin_, "-m", str(model), "-s", str(src), "--no-save"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/mixture")
    ap.add_argument("--bin", default="cpp/build_pkg-cpu/yolomaster_edge")
    ap.add_argument("--images", default="visdrone50/images/val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    import onnxruntime as ort
    import tempfile

    mdir = Path(args.models)
    bin_ = str(Path(args.bin).resolve())
    onnx_models = sorted(mdir.glob("*.onnx"))
    if args.only:
        keep = set(args.only.split(","))
        onnx_models = [m for m in onnx_models if m.stem in keep]
    failures = []

    for model in onnx_models:
        print(f"\n=== {model.stem} ===")
        sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        det_out = next(o for o in sess.get_outputs() if len(o.shape) == 3)
        e2e = det_out.shape[1] >= 32 and det_out.shape[2] == 6
        print(f"  layout: {'end2end ' + str(det_out.shape) if e2e else 'anchors ' + str(det_out.shape)}")

        ref_npz = mdir / f"{model.stem}.ref.npz"
        ref = np.load(ref_npz) if ref_npz.exists() else None
        stems = sorted({k.rsplit("_", 1)[0] for k in ref.files}) if ref is not None else []

        with tempfile.TemporaryDirectory() as tmp:
            rates_a, rates_b = [], []
            for stem in stems:
                img = next(Path(args.images).glob(stem + ".*"))
                # C++ CLI boxes
                run_cli(bin_, model.resolve(), img.resolve(),
                        ["--conf", str(args.conf), "--iou", "0.7", "--quiet",
                         "--save-txt", tmp], tmp)
                got_xyxy, got_conf, got_cls = read_savetxt(Path(tmp) / f"{stem}.txt")
                # A: torch eager reference vs CLI
                r_xyxy, r_cls = ref[stem + "_xyxy"], ref[stem + "_cls"].astype(int)
                rates_a.append(match_rate(r_xyxy, r_cls, got_xyxy, got_cls))
                # B: python ORT decode vs CLI (tight: same preprocessing math)
                blob, lb = preprocess(img, args.imgsz)
                out = sess.run([det_out.name], {in_name: blob})[0]
                p_xyxy, p_conf, p_cls = decode_np(out, lb, args.conf)
                rates_b.append(match_rate(p_xyxy, p_cls, got_xyxy, got_cls, iou_thr=0.9))
            a, b = (np.mean(rates_a) if rates_a else -1), (np.mean(rates_b) if rates_b else -1)
            print(f"  A torch-eager vs CLI match: {a:.3f}   B py-ORT vs CLI match: {b:.3f}")
            if rates_a and a < 0.8:
                failures.append(f"{model.stem}: eager-vs-CLI match {a:.3f}")
            if rates_b and b < 0.9:
                failures.append(f"{model.stem}: pyORT-vs-CLI match {b:.3f}")

            # C: feature smoke (folder = images dir limited)
            log = run_cli(bin_, model.resolve(), Path(args.images).resolve(),
                          ["--limit", "2", "--quiet", "--slicing", "sparse", "--cw-nms",
                           "--export-labels", tmp + "/lbl"], tmp)
            ok_meta = re.search(r"nc=80", log) or re.search(r"nc=\d+ \(model-metadata\)", log)
            ok_slice = "[slicing]" in log
            ok_cw = "nms=cw" in log
            m_lbl = re.search(r"\[labels\].*instances=(\d+)", log)
            m_dets = re.search(r"total_dets=(\d+)", log)
            ok_lbl = m_lbl and m_dets and m_lbl.group(1) == m_dets.group(1)
            print(f"  C meta:{bool(ok_meta)} slicing:{ok_slice} cw:{ok_cw} labels==dets:{bool(ok_lbl)}")
            for name, ok in [("meta", ok_meta), ("slicing", ok_slice), ("cw", ok_cw), ("labels", ok_lbl)]:
                if not ok:
                    failures.append(f"{model.stem}: feature check '{name}' failed")

            # MNN / ncnn siblings: det-count parity vs onnx backend
            for sib, flag in [(mdir / f"{model.stem}.mnn", "mnn"), (mdir / f"{model.stem}_ncnn", "ncnn")]:
                if not sib.exists():
                    continue
                log_o = run_cli(bin_, model.resolve(), Path(args.images).resolve(),
                                ["--limit", "3", "--quiet", "--conf", "0.25"], tmp)
                log_s = run_cli(bin_, sib.resolve(), Path(args.images).resolve(),
                                ["--limit", "3", "--quiet", "--conf", "0.25"], tmp)
                d_o = re.search(r"total_dets=(\d+)", log_o)
                d_s = re.search(r"total_dets=(\d+)", log_s)
                same = d_o and d_s and abs(int(d_o.group(1)) - int(d_s.group(1))) <= 2
                print(f"  {flag} parity: onnx={d_o and d_o.group(1)} {flag}={d_s and d_s.group(1)} -> {'OK' if same else 'MISMATCH'}")
                if not same:
                    failures.append(f"{model.stem}: {flag} det-count parity")

    print("\n===== validation summary =====")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print(f"  all checks passed for {len(onnx_models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
