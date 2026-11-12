#!/usr/bin/env python3
"""W8A16 for an existing .mlpackage: int8 per-channel linear-symmetric weights, fp16 activations.

    python coreml_export/quantize_w8a16.py yolov12x.mlpackage yolov12x-w8a16.mlpackage

Runs on Linux: weight-only, no calibration data. The dequantize happens on the compute unit at
load / run time (`constexpr_affine_dequantize`), so the package halves on disk and in memory;
the arithmetic stays fp16.

ONLY the weights of convolutions and linear layers are quantized (the constants feeding `conv`,
`conv_transpose`, `linear` and `matmul` ops). A global config - what scripts/export_coreml_p03.py
used, and this script before 2026-11-12 - also rounds every other table of 2048 values or more
in the graph. In a YOLO head those are the anchor grid [1, 2, anchors] and the stride table
[1, anchors], each rounded with ONE scale for the whole table: grid coordinates 0.5 .. 79.5
land on multiples of 0.626 (up to a third of a cell off, i.e. up to 10 px at stride 32) and
stride 8 becomes 8.06. Every box moves; the weights have nothing to do with it. The summary
printed at the end lists what was quantized and the large tables left in fp16.
"""
import collections
import sys

import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

WEIGHT_OPS = ("conv", "conv_transpose", "linear", "matmul")


def summary(path: str) -> str:
    """What the saved package holds: int8 weight tensors and the fp16 tables of 2048 values or more."""
    spec = ct.utils.load_spec(path)
    q_values, q_tensors, kept = 0, collections.Counter(), []
    for fn in spec.mlProgram.functions.values():
        for blk in fn.block_specializations.values():
            for op in blk.operations:
                o = op.outputs[0]
                shape = [d.constant.size for d in o.type.tensorType.dimensions]
                n = 1
                for s in shape:
                    n *= s
                if op.type == "constexpr_affine_dequantize":
                    q_values += n
                    q_tensors[len(shape)] += 1
                elif op.type == "const" and n >= 2048:
                    kept.append((n, o.name, shape))
    kept.sort(reverse=True)
    lines = [f"int8: {q_values / 1e6:.2f} M values in {sum(q_tensors.values())} tensors "
             f"(by rank: {dict(sorted(q_tensors.items()))})",
             f"fp16 tables of 2048+ values left as they are: {len(kept)}, {sum(k[0] for k in kept) / 1e6:.3f} M values"]
    lines += [f"    {name} {shape}" for _, name, shape in kept[:6]]
    return "\n".join(lines)


def main(src: str, dst: str) -> None:
    m = ct.models.MLModel(src, skip_model_load=True)
    op = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    cfg = OptimizationConfig(global_config=None, op_type_configs={t: op for t in WEIGHT_OPS})
    q = linear_quantize_weights(m, config=cfg)
    for k, v in m.user_defined_metadata.items():
        q.user_defined_metadata[k] = v
    q.user_defined_metadata["precision"] = "w8a16"
    q.user_defined_metadata["quantization"] = "int8 conv/linear weights per-channel linear_symmetric, fp16 activations"
    q.short_description = m.short_description or ""
    q.save(dst)
    print(f"OK  {dst}")
    print(summary(dst))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    main(sys.argv[1], sys.argv[2])
