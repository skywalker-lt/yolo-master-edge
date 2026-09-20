# Changelog

Two version numbers: the runtime, CLI and repository tag (`VERSION`) and the API server image
(`cpp/server/SERVER_VERSION`), because the server first shipped as image tag 1.2.0 on top of
runtime 1.1.1 and that image stays as published.

## Runtime 1.2.0 / server 1.2.1 (in progress, Linux phase)

Runtime and CLI
- One `preprocess_nchw` for every backend (letterbox, BGR to RGB, /255, NCHW); TensorRT and MNN
  keep their staging buffers; output byte-identical to 1.1.1.
- ncnn: fp32 is pinned per layer (the 1e30 mask and 1e-9 nudge chains of the emulated MoE
  routers, via `Layer::featmask`) instead of refusing fp16 for the whole model; metadata
  `fp16_mixed: false` vetoes.
- MNN: an fp16 request on a routed model (metadata `fp16_safe: false`) is downgraded to fp32 and
  explained in `ep_note` instead of returning zero detections; experimental multi-path session
  behind `YOLOMASTER_MNN_MULTIPATH` (not correct on MNN 3.6.1, see the source comment).
- Bench mode: `--bench cold|sustained`, `--bench-json`, one `yolomaster-bench/v1` schema (floor
  rank percentiles, the phones' convention); `--accuracy` scores the source with an in-process
  mAP that equals `scripts/eval_map*.py` to four decimals; `yolomaster_score` scores txt dumps.
- Multi-object tracking: `--track botsort|bytetrack` on video sources (Kalman, two-stage IoU
  association, BoT-SORT camera motion compensation behind `USE_GMC`), ids in the annotated
  video and as a 7th txt column.
- `scripts/make_coco_subset.py` (coco500) and `scripts/package_eval_sets.sh` for the labelled
  eval sets (release assets).
- `--version` on both binaries; bench JSON carries version, commit and build flags.
- Phone graph: the pruned v0.1-N ships as the dense ncnn export (fused SDPA, native gate
  broadcast; x86 parity with the ONNX path 261/261 detections, the stock export was not) with a
  re-quantized ACIQ INT8 sibling (139/145 layers, 2048 COCO train images): coco500 in-process
  mAP50-95 fp32 0.4309, INT8 0.4205 (-1.04 AP, x86 ncnn); the S26 re-timing is the Android phase.

API server 1.2.1
- `track=` on `/v1/video` and `/v1/stream` (`track_id` per detection), `POST /v1/bench`,
  per-request slicing documented and tested.
- API-key authentication (`api_keys`, `YM_API_KEYS`, `--api-key-file`), token-bucket rate
  limiting (`rate_limit`), inbound `X-Request-Id` and W3C `traceparent` propagation, JSON access
  log (`log_format: json`), `GET /openapi.json` rendered from the route table, counters
  `yolomaster_rate_limited_total` and `yolomaster_auth_failed_total`.
- Image tag from `SERVER_VERSION` (`deploy/docker/version.bzl` written by `stage_bundle.sh`);
  `YM_API_KEYS` passthrough in `deploy/run.sh` and `docker-compose.yml`.
- Tests: `tests/run_server_tests.sh` boots a second, hardened instance for
  `tests/server/test_hardened.py`; CPU-only GitHub Actions workflow.

## 1.1.1

Previous release (see the GitHub release notes).
