// Benchmark mode shared by every host of the core (CLI, server, later the Android / iOS apps):
// one result schema ("yolomaster-bench/v1"), one statistics convention, one probe.
//
// Statistics convention: floor rank. For an ascending sorted vector s of n samples,
// percentile q = s[min(int(q * n), n - 1)]; median is q = 0.5. This is exactly what the Android
// BenchStats and the iOS BenchView already compute, so device numbers stay comparable.
// Sustained: median of the slowest quarter of all samples (the phones' lastQuarterMedian);
// throttle_pct = (sustained - cold) / cold * 100.
#pragma once
#include <atomic>
#include <string>
#include <vector>
#include "json.hpp"
#include "yolomaster.hpp"
#include "map_metrics.hpp"

namespace yolomaster::bench {

constexpr const char* kSchema = "yolomaster-bench/v1";

struct StageStats {
    size_t n = 0;
    double mean = 0, median = 0, p90 = 0, p95 = 0, p99 = 0, min = 0, max = 0;
};
StageStats reduce(std::vector<double> samples);      // sorts a copy; n == 0 -> all zero
nlohmann::json to_json(const StageStats& s);

// Per-frame stage samples collected during a dataset pass.
struct Samples {
    std::vector<double> pre, infer, post, total;
    long frames = 0, total_dets = 0;
    void add(const Backend& be, size_t ndets);
};

struct ModelInfo {
    std::string id, path, backend, runtime, execution_provider, precision, ep_note;
    int nc = 0; bool is_seg = false; int imgsz = 0;
};
struct EnvInfo {
    std::string host, os, cpu_model;
    int cpu_count = 0; long long mem_bytes = 0;
    std::string gpu_name;
    int threads = 0;
    std::string git_commit, build_flags, version;
};
struct Protocol {
    std::string mode = "cold";                // cold | sustained
    float conf = 0, iou = 0; int max_det = 0; bool multi_label = false;
    std::string slicing = "off"; int tile_size = 0;
    int warmup = 0, iters = 0; double minutes = 0;
    std::string probe = "gray114", probe_mode;   // probe_mode: infer_only | full
    std::string dataset; int image_count = 0; std::string image_list_sha256;
};
struct ColdResult { StageStats infer_ms; std::string probe_mode; };
struct SustainedResult {
    StageStats infer_ms;
    double cold_median_ms = 0, sustained_median_ms = 0, throttle_pct = 0, duration_s = 0;
    std::vector<double> sparkline;             // one median per second
    std::string probe_mode;
};
struct DatasetResult {
    long frames = 0, total_dets = 0;
    StageStats pre_ms, infer_ms, post_ms, total_ms;
    double model_fps = 0, wall_s = 0;
};
struct AccuracyResult {
    bool present = false;
    float conf = 0, iou = 0; int max_det = 0; bool multi_label = false;
    std::string labels;
    metrics::MapResult map;
    StageStats infer_ms;
};
struct BenchResult {
    std::string tool = "cli";                 // cli | server | android | ios
    std::string timestamp;
    ModelInfo model; EnvInfo env; Protocol protocol;
    bool has_cold = false; ColdResult cold;
    bool has_sustained = false; SustainedResult sustained;
    bool has_dataset = false; DatasetResult dataset;
    AccuracyResult accuracy;
};

// Gray-114 imgsz probe forwards: `warmup` untimed, `iters` timed. Uses forward_raw(decode=false)
// (pure kernel time) when the backend implements it, else infer() and probe_mode = "full".
ColdResult cold_sweep(Backend& be, const Config& cfg, int warmup, int iters);
// Timed loop for `minutes` after `warmup`; cold baseline = median of the first `cold_iters`.
SustainedResult sustained_loop(Backend& be, const Config& cfg, int warmup, double minutes,
                               int cold_iters = 50, const std::atomic<bool>* cancel = nullptr);
EnvInfo collect_env(const Backend& be, int threads);
ModelInfo model_info(const Backend& be, const std::string& path, const std::string& backend,
                     const std::string& precision, const Config& cfg);
// sha256 over the sorted basenames joined by '\n' (scripts/make_coco_subset.py hashes the same string)
std::string image_list_sha256(const std::vector<std::string>& paths);
std::string sha256_hex(const std::string& data);
std::string timestamp_utc();
nlohmann::json to_json(const BenchResult& r);

} // namespace yolomaster::bench
