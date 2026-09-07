#!/usr/bin/env bash
# Stage ncnn model dirs from the repo's (gitignored) models/ into the runtime's test
# assets so the on-device harness can load them. Run before ./gradlew connectedAndroidTest.
#
# Models are NOT committed (they ship as GitHub Release assets); this copies your local
# working-tree copies into assets/ where the instrumented test reads them.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DST="$REPO/android/runtime/src/main/assets/models"
mkdir -p "$DST"

# name-in-assets  <-  path-under-repo/models   (float models)
STAGE=(
  "v0.1-seg-n_ncnn:v0.1-seg-n_ncnn"
  "moa-n_ncnn:mixture/moa-n_ncnn"
  "esmoe_n_visdrone_ncnn:esmoe_n_visdrone_ncnn"
  "p03_v01n_ncnn:p03_v01n_ncnn"
)
# Pre-quantized mixed-INT8 siblings (scripts/quantize_ncnn_int8.py). The runtime resolves
# Precision.INT8 to "<name>-int8_ncnn" next to the float dir, so they MUST be staged under
# exactly that name. Same three files as a float dir; optional - a missing sibling only skips
# the INT8 rows of the harness (and makes Precision.INT8 fail cleanly with lastError).
STAGE_INT8=(
  "v0.1-seg-n-int8_ncnn:v0.1-seg-n-int8_ncnn"
  "esmoe_n_visdrone-int8_ncnn:esmoe_n_visdrone-int8_ncnn"
  "p03_v01n-int8_ncnn:p03_v01n-int8_ncnn"
)

stage_one() {  # name-in-assets  path-under-repo/models  kind
  local name="$1" rel="$2" kind="$3" src="$REPO/models/$2"
  if [ -d "$src" ]; then
    rm -rf "$DST/$name"
    cp -r "$src" "$DST/$name"
    echo "staged  $name  ($kind)"
  elif [ "$kind" = int8 ]; then
    echo "WARN: $src not found - build it with scripts/quantize_ncnn_int8.py (INT8 rows will be skipped)" >&2
  else
    echo "WARN: $src not found - fetch the ncnn models first" >&2
  fi
}
for pair in "${STAGE[@]}";      do stage_one "${pair%%:*}" "${pair##*:}" float; done
for pair in "${STAGE_INT8[@]}"; do stage_one "${pair%%:*}" "${pair##*:}" int8;  done

PROBE="$REPO/android/runtime/src/main/assets/probe.jpg"
[ -f "$PROBE" ] || echo "NOTE: drop a test image at $PROBE (any scene with objects)"
echo "      and a VisDrone-domain image at $REPO/android/runtime/src/main/assets/probe_visdrone.jpg (used for esmoe/mixture rows)"
echo "done. now: cd android && ./gradlew :runtime:connectedAndroidTest"
