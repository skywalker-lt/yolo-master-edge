#!/usr/bin/env bash
# Project03 engines on the Orin: pruned v0.1-N COCO, FP16 + mixed-INT8 (explicit
# Q/DQ, calibrate-only, the accuracy-criterion artifact: -0.8 AP vs fp32).
#
# The QDQ ONNX carries its quantization scales in-graph, so INT8 needs NO
# calibrator and no calibration data on-device: trtexec --int8 --fp16 is the
# whole build. Engines are arch-specific and must be built here, not copied.
#
# Model files are fetched from the mdb drop (override with MDB=user@host:path):
#   yolo-master-v01n-coco-pruned.onnx   (plain, for fp32/fp16 engines)
#   yolo-master-v01n-coco-qdq.onnx      (explicit-INT8 quantized)
# Latency methodology: report the "GPU Compute Time" median (model-only). The
# end-to-end number belongs to the runner (yolomaster CLI), not trtexec.
set -e
cd "$(dirname "$0")"
MDB="${MDB:-root@185.76.11.54:/yolotmp/jetson-p03}"
PLAIN=models/yolo-master-v01n-coco-pruned.onnx
QDQ=models/yolo-master-v01n-coco-qdq.onnx

mkdir -p models engines
for f in "$PLAIN" "$QDQ" models/p03-metadata.yaml; do
  [ -f "$f" ] || scp "$MDB/$(basename "$f")" "$f"
done

# locate trtexec across the common JetPack paths (env override wins)
TRTEXEC="${TRTEXEC:-}"
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
  for c in "$(command -v trtexec 2>/dev/null)" \
           /usr/src/tensorrt/bin/trtexec \
           "$(find /usr -name trtexec -type f 2>/dev/null | head -1)"; do
    [ -n "$c" ] && [ -x "$c" ] && { TRTEXEC="$c"; break; }
  done
fi
[ -n "$TRTEXEC" ] && [ -x "$TRTEXEC" ] || {
  echo "trtexec not found. sudo apt install -y tensorrt  (then re-run)"; exit 1; }
echo "using trtexec: $TRTEXEC"

WORKSPACE="${WORKSPACE:-512}"
OPT="${OPT:-3}"      # 3 matches the datacenter ladders; OPT=5 for the last few percent

bench() {  # name  onnx  extra-flags...
  local name="$1" onnx="$2"; shift 2
  local log="engines/trtexec_p03_${name}.log"
  echo "==================== p03 $name ===================="
  "$TRTEXEC" --onnx="$onnx" --saveEngine="engines/p03_${name}.engine" \
             --memPoolSize=workspace:"${WORKSPACE}" \
             --builderOptimizationLevel="${OPT}" "$@" 2>&1 | tee "$log" | \
             grep -E "Engine built|Total Host|error|Error" || true
  # sidecar so the yolomaster runner picks up names/imgsz automatically
  cp models/p03-metadata.yaml "engines/p03_${name}.metadata.yaml"
  echo "-------------------- $name result --------------------"
  grep -iE "Throughput|GPU Compute Time:|error|failed" "$log" | tail -6 | sed 's/^/  /'
  echo
}

bench fp32      "$PLAIN"                       # honest baseline: no TF32 on Orin GPU anyway
bench fp16      "$PLAIN" --fp16
bench int8-qdq  "$QDQ"   --int8 --fp16         # explicit quantization: scales in-graph

echo "==================== summary ===================="
echo "  Engines + metadata sidecars in engines/."
echo "  Model-only latency = 'GPU Compute Time' median above."
echo "  End-to-end via the runner (same binary as the tier-3 battery):"
echo "    ./build/yolomaster --engine engines/p03_fp16.engine --input <img-or-dir> --bench"
echo "    ./build/yolomaster --engine engines/p03_int8-qdq.engine --input <img-or-dir> --bench"
