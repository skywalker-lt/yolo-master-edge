#include "bench.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <thread>
#ifndef _WIN32
#include <sys/utsname.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace yolomaster::bench {

using clk = std::chrono::steady_clock;

StageStats reduce(std::vector<double> v) {
    StageStats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    auto q = [&](double p) { return v[std::min(static_cast<size_t>(p * n), n - 1)]; };
    s.n = n;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    s.median = q(0.5); s.p90 = q(0.9); s.p95 = q(0.95); s.p99 = q(0.99);
    s.min = v.front(); s.max = v.back();
    return s;
}

nlohmann::json to_json(const StageStats& s) {
    return {{"n", s.n}, {"mean", s.mean}, {"median", s.median}, {"p90", s.p90}, {"p95", s.p95},
            {"p99", s.p99}, {"min", s.min}, {"max", s.max}};
}

void Samples::add(const Backend& be, size_t ndets) {
    pre.push_back(be.pre_ms); infer.push_back(be.infer_ms); post.push_back(be.post_ms);
    total.push_back(be.pre_ms + be.infer_ms + be.post_ms);
    frames++; total_dets += static_cast<long>(ndets);
}

static cv::Mat probe_image(int imgsz) {
    return cv::Mat(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
}

// forward_raw(decode=false) is the phone Bench tab's inferOnly; backends without it (TensorRT,
// MNN) fall back to a full infer() and say so.
static bool probe_once(Backend& be, const cv::Mat& probe, const Config& cfg, bool& infer_only) {
    if (infer_only) {
        try { be.forward_raw(probe, cfg, /*decode=*/false); return true; }
        catch (const std::exception&) { infer_only = false; }
    }
    try { (void)be.infer(probe, cfg); return true; } catch (const std::exception&) { return false; }
}

ColdResult cold_sweep(Backend& be, const Config& cfg, int warmup, int iters) {
    ColdResult r;
    const cv::Mat probe = probe_image(cfg.imgsz);
    bool infer_only = true;
    for (int i = 0; i < warmup; ++i) if (!probe_once(be, probe, cfg, infer_only)) break;
    std::vector<double> v; v.reserve(std::max(iters, 0));
    for (int i = 0; i < iters; ++i) {
        if (!probe_once(be, probe, cfg, infer_only)) break;
        v.push_back(be.infer_ms);
    }
    r.infer_ms = reduce(v);
    r.probe_mode = infer_only ? "infer_only" : "full";
    return r;
}

SustainedResult sustained_loop(Backend& be, const Config& cfg, int warmup, double minutes,
                               int cold_iters, const std::atomic<bool>* cancel) {
    SustainedResult r;
    const cv::Mat probe = probe_image(cfg.imgsz);
    bool infer_only = true;
    for (int i = 0; i < warmup; ++i) if (!probe_once(be, probe, cfg, infer_only)) break;
    std::vector<double> all, second;
    const auto t0 = clk::now();
    auto sec_start = t0;
    const double budget_s = minutes * 60.0;
    while (true) {
        if (cancel && cancel->load()) break;
        const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();
        if (elapsed >= budget_s && static_cast<int>(all.size()) >= cold_iters) break;
        if (!probe_once(be, probe, cfg, infer_only)) break;
        all.push_back(be.infer_ms); second.push_back(be.infer_ms);
        if (std::chrono::duration<double>(clk::now() - sec_start).count() >= 1.0) {
            r.sparkline.push_back(reduce(second).median);
            second.clear(); sec_start = clk::now();
        }
    }
    if (!second.empty()) r.sparkline.push_back(reduce(second).median);
    r.duration_s = std::chrono::duration<double>(clk::now() - t0).count();
    r.infer_ms = reduce(all);
    if (!all.empty()) {
        std::vector<double> head(all.begin(), all.begin() + std::min<size_t>(all.size(), std::max(cold_iters, 1)));
        r.cold_median_ms = reduce(head).median;
        std::vector<double> sorted = all;
        std::sort(sorted.begin(), sorted.end());
        const size_t q = std::max<size_t>(sorted.size() / 4, 1);
        std::vector<double> slowest(sorted.end() - q, sorted.end());
        r.sustained_median_ms = reduce(slowest).median;
        r.throttle_pct = r.cold_median_ms > 0 ? (r.sustained_median_ms - r.cold_median_ms) / r.cold_median_ms * 100.0 : 0;
    }
    r.probe_mode = infer_only ? "infer_only" : "full";
    return r;
}

static std::string read_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
            const auto c = line.find(':');
            if (c != std::string::npos) {
                std::string v = line.substr(c + 1);
                const auto b = v.find_first_not_of(" \t");
                return b == std::string::npos ? "" : v.substr(b);
            }
        }
    }
    return "";
}

static std::string popen_line(const char* cmd) {
    std::string out;
#ifndef _WIN32
    FILE* p = popen(cmd, "r");
    if (!p) return out;
    char buf[256];
    if (fgets(buf, sizeof(buf), p)) out = buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
#endif
    return out;
}

EnvInfo collect_env(const Backend& be, int threads) {
    EnvInfo e;
#ifndef _WIN32
    char h[256] = {0};
    if (gethostname(h, sizeof(h) - 1) == 0) e.host = h;
    struct utsname u{};
    if (uname(&u) == 0) e.os = std::string(u.sysname) + " " + u.release + " " + u.machine;
#endif
    e.cpu_model = read_cpu_model();
    e.cpu_count = static_cast<int>(std::thread::hardware_concurrency());
#ifdef __linux__
    struct sysinfo si{};
    if (sysinfo(&si) == 0) e.mem_bytes = static_cast<long long>(si.totalram) * si.mem_unit;
#endif
    e.gpu_name = be.device_name();
    if (e.gpu_name.empty()) {
        const std::string ep = be.active_ep;
        if (ep.find("CUDA") != std::string::npos || ep.find("cuda") != std::string::npos ||
            ep.find("TRT") != std::string::npos || ep.find("trt") != std::string::npos ||
            ep.find("TensorRT") != std::string::npos)
            e.gpu_name = popen_line("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null");
    }
    e.threads = threads;
#ifdef YM_GIT_COMMIT
    e.git_commit = YM_GIT_COMMIT;
#endif
#ifdef YM_BUILD_FLAGS
    e.build_flags = YM_BUILD_FLAGS;
#endif
#ifdef YM_VERSION
    e.version = YM_VERSION;
#endif
    return e;
}

ModelInfo model_info(const Backend& be, const std::string& path, const std::string& backend,
                     const std::string& precision, const Config& cfg) {
    ModelInfo m;
    std::string p = path;
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    const auto slash = p.find_last_of("/\\");
    std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
    const auto dot = base.find_last_of('.');
    m.id = (dot == std::string::npos || dot == 0) ? base : base.substr(0, dot);
    m.path = path; m.backend = backend; m.runtime = be.runtime_name();
    m.execution_provider = be.active_ep; m.ep_note = be.ep_note; m.precision = precision;
    m.nc = cfg.num_classes(); m.is_seg = be.is_seg(); m.imgsz = cfg.imgsz;
    return m;
}

// ---- compact SHA-256 (FIPS 180-4), enough for list fingerprints ----
namespace {
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
}

std::string sha256_hex(const std::string& data) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::string msg = data;
    const uint64_t bitlen = static_cast<uint64_t>(data.size()) * 8;
    msg.push_back(static_cast<char>(0x80));
    while (msg.size() % 64 != 56) msg.push_back('\0');
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bitlen >> (i * 8)) & 0xff));
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4])) << 24) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 1])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 2])) << 8) |
                   static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
    return std::string(out, 64);
}

std::string image_list_sha256(const std::vector<std::string>& paths) {
    std::vector<std::string> names;
    names.reserve(paths.size());
    for (const auto& p : paths) {
        const auto slash = p.find_last_of("/\\");
        names.push_back(slash == std::string::npos ? p : p.substr(slash + 1));
    }
    std::sort(names.begin(), names.end());
    std::string joined;
    for (size_t i = 0; i < names.size(); ++i) { if (i) joined += '\n'; joined += names[i]; }
    return sha256_hex(joined);
}

std::string timestamp_utc() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

nlohmann::json to_json(const BenchResult& r) {
    nlohmann::json j;
    j["schema_version"] = kSchema;
    j["timestamp"] = r.timestamp.empty() ? timestamp_utc() : r.timestamp;
    j["tool"] = r.tool;
    j["model"] = {{"id", r.model.id}, {"path", r.model.path}, {"backend", r.model.backend},
                  {"runtime", r.model.runtime}, {"execution_provider", r.model.execution_provider},
                  {"ep_note", r.model.ep_note}, {"precision", r.model.precision}, {"nc", r.model.nc},
                  {"is_seg", r.model.is_seg}, {"imgsz", r.model.imgsz}};
    j["environment"] = {{"host", r.env.host}, {"os", r.env.os}, {"cpu_model", r.env.cpu_model},
                        {"cpu_count", r.env.cpu_count}, {"mem_bytes", r.env.mem_bytes}, {"gpu_name", r.env.gpu_name},
                        {"threads", r.env.threads}, {"git_commit", r.env.git_commit},
                        {"build_flags", r.env.build_flags}, {"version", r.env.version}};
    j["protocol"] = {{"mode", r.protocol.mode}, {"conf", r.protocol.conf}, {"iou", r.protocol.iou},
                     {"max_det", r.protocol.max_det}, {"multi_label", r.protocol.multi_label},
                     {"slicing", r.protocol.slicing}, {"tile_size", r.protocol.tile_size},
                     {"warmup", r.protocol.warmup}, {"iters", r.protocol.iters}, {"minutes", r.protocol.minutes},
                     {"probe", r.protocol.probe}, {"probe_mode", r.protocol.probe_mode},
                     {"dataset", r.protocol.dataset}, {"image_count", r.protocol.image_count},
                     {"image_list_sha256", r.protocol.image_list_sha256}};
    j["stats_convention"] = "floor_rank";
    if (r.has_cold) j["cold"] = {{"infer_ms", to_json(r.cold.infer_ms)}, {"probe_mode", r.cold.probe_mode}};
    if (r.has_sustained)
        j["sustained"] = {{"infer_ms", to_json(r.sustained.infer_ms)}, {"cold_median_ms", r.sustained.cold_median_ms},
                          {"sustained_median_ms", r.sustained.sustained_median_ms},
                          {"throttle_pct", r.sustained.throttle_pct}, {"sparkline", r.sustained.sparkline},
                          {"duration_s", r.sustained.duration_s}, {"probe_mode", r.sustained.probe_mode}};
    if (r.has_dataset)
        j["dataset"] = {{"frames", r.dataset.frames}, {"total_dets", r.dataset.total_dets},
                        {"pre_ms", to_json(r.dataset.pre_ms)}, {"infer_ms", to_json(r.dataset.infer_ms)},
                        {"post_ms", to_json(r.dataset.post_ms)}, {"total_ms", to_json(r.dataset.total_ms)},
                        {"model_fps", r.dataset.model_fps}, {"wall_s", r.dataset.wall_s}};
    if (r.accuracy.present) {
        nlohmann::json pc = nlohmann::json::array();
        for (const auto& c : r.accuracy.map.per_class)
            pc.push_back({{"class_id", c.cls}, {"n_gt", c.n_gt}, {"n_pred", c.n_pred},
                          {"ap50", c.ap50()}, {"ap5095", c.ap5095()}});
        j["accuracy"] = {{"protocol", {{"conf", r.accuracy.conf}, {"iou", r.accuracy.iou},
                                       {"max_det", r.accuracy.max_det}, {"multi_label", r.accuracy.multi_label}}},
                         {"labels", r.accuracy.labels}, {"images", r.accuracy.map.images},
                         {"map50", r.accuracy.map.map50}, {"map5095", r.accuracy.map.map5095},
                         {"per_class", pc}, {"timings", {{"infer_ms", to_json(r.accuracy.infer_ms)}}}};
    }
    return j;
}

} // namespace yolomaster::bench
