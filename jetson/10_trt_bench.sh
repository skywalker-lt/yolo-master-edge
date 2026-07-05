#!/usr/bin/env bash
# Build TensorRT engines from the ONNX and report GPU throughput on the Orin.
# FP16 = the safe deploy precision; INT8 = the tensor-core speed ceiling (uncalibrated here).
set -e
cd "$(dirname "$0")"
M=models/esmoe_n_visdrone_sim.onnx
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
[ -f "$M" ]        || { echo "missing $M — see 00_setup.sh"; exit 1; }
[ -x "$TRTEXEC" ]  || { echo "trtexec not found"; exit 1; }
mkdir -p models engines

bench() {  # name  extra-flags
  local name="$1"; shift
  echo "==================== $name ===================="
  "$TRTEXEC" --onnx="$M" --saveEngine="engines/esmoe_n_${name}.engine" \
             --memPoolSize=workspace:2048 "$@" 2>&1 \
    | grep -iE "Throughput|GPU Compute Time: .*mean|Latency: .*mean|error|failed" \
    | sed 's/^/  /'
}

bench fp16  --fp16
bench int8  --int8 --fp16          # INT8 kernels where available, FP16 fallback (dynamic ranges)

echo
echo "==================== summary ===================="
echo "  Throughput = frames/sec (qps). GPU Compute mean = per-inference ms."
echo "  Engines saved to engines/.  FP16 is the deploy engine; for accurate INT8 build from a"
echo "  calibrated model with the detection head pinned to FP16 (see TECHNICAL_REPORT.md §3)."
