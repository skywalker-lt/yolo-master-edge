#!/usr/bin/env bash
# Boots yolomaster_server on a free port with two CPU models and runs tests/server/test_api.py.
# Env: BIN (server binary), CLI (yolomaster_edge), ONNX, NCNN (models), DIR (images), PY (python).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN=${BIN:-$ROOT/cpp/build/server/yolomaster_server}
CLI=${CLI:-$ROOT/cpp/build/yolomaster_edge}
ONNX=${ONNX:-$ROOT/models/esmoe_n_visdrone_sim.onnx}
NCNN=${NCNN:-$ROOT/models/esmoe_n_visdrone_ncnn}
DIR=${DIR:-$ROOT/visdrone50/images/val}
PY=${PY:-python3}
PORT=${PORT:-$(( 20000 + RANDOM % 20000 ))}
LOG=${LOG:-/tmp/yolomaster_server_test_$PORT.log}
[ -x "$BIN" ] || { echo "server binary not found: $BIN (build with -DBUILD_SERVER=ON)"; exit 2; }
"$BIN" -p "$PORT" --loop-threads 2 --max-queue 8 --max-body-mb 4 \
  -m "esmoe=$ONNX,threads=2,workers=2" -m "esmoe-ncnn=$NCNN,threads=2,workers=1" > "$LOG" 2>&1 &
SPID=$!
trap 'kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 0.5; done
curl -sf "localhost:$PORT/readyz" >/dev/null || { echo "server did not become ready; log:"; tail -20 "$LOG"; exit 3; }
export YM_SERVER_URL="http://localhost:$PORT" YM_TEST_MODEL=esmoe YM_TEST_MODEL2=esmoe-ncnn YM_TEST_IMAGES="$DIR" YM_CLI="$CLI" YM_CLI_MODEL="$ONNX"
if $PY -c "import pytest" 2>/dev/null; then $PY -m pytest "$ROOT/tests/server" -q -p no:cacheprovider; RC=$?
else $PY "$ROOT/tests/server/test_api.py"; RC=$?; fi
kill -INT $SPID; wait $SPID; SRC=$?
echo "server exit code after SIGINT: $SRC (log: $LOG)"
[ $RC -eq 0 ] && [ $SRC -eq 0 ]
