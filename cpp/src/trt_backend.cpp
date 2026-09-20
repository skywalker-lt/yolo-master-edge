#include "trt_backend.hpp"
#ifdef USE_TRT_ONNXPARSER
#include <NvOnnxParser.h>
#endif
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
    }
};
static TrtLogger g_logger;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); } while (0)

// ---- engine building (nvonnxparser) ----
namespace {
// tiny SHA-1 (public-domain style) for cache keys: content-addressed engines
struct Sha1 {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    uint64_t len = 0; uint8_t buf[64]; size_t blen = 0;
    static uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    void block(const uint8_t* p) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) | (uint32_t(p[4*i+2]) << 8) | p[4*i+3];
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f,k;
            if (i<20){f=(b&c)|(~b&d);k=0x5A827999u;} else if(i<40){f=b^c^d;k=0x6ED9EBA1u;}
            else if(i<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDCu;} else {f=b^c^d;k=0xCA62C1D6u;}
            uint32_t t = rol(a,5)+f+e+k+w[i]; e=d; d=c; c=rol(b,30); b=a; a=t;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
    }
    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n) { size_t k = std::min(n, 64 - blen); memcpy(buf + blen, p, k); blen += k; p += k; n -= k;
                    if (blen == 64) { block(buf); blen = 0; } }
    }
    std::string hex() {
        uint64_t bits = len * 8; uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0; while (blen != 56) update(&z, 1);
        uint8_t lb[8]; for (int i = 0; i < 8; ++i) lb[i] = uint8_t(bits >> (56 - 8*i)); update(lb, 8);
        char out[41]; for (int i = 0; i < 5; ++i) std::snprintf(out + 8*i, 9, "%08x", h[i]); return std::string(out, 40);
    }
};
std::string file_sha1(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    Sha1 s; std::vector<uint8_t> b(1 << 20);
    while (f) { f.read(reinterpret_cast<char*>(b.data()), b.size()); s.update(b.data(), size_t(f.gcount())); }
    return s.hex();
}
std::string gpu_slug() {
    cudaDeviceProp p{}; int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return "gpu";
    std::string s(p.name); std::string o;
    for (char c : s) o += (std::isalnum(static_cast<unsigned char>(c)) ? char(std::tolower(c)) : '_');
    return o;
}
bool ends_with(const std::string& a, const std::string& suf) {
    return a.size() >= suf.size() && a.compare(a.size() - suf.size(), suf.size(), suf) == 0;
}
} // namespace

std::string TrtBackend::cached_engine_path(const std::string& onnx_path, const TrtOptions& opt) {
    namespace fs = std::filesystem;
    const fs::path op(onnx_path);
    const fs::path dir = opt.cache_dir.empty() ? op.parent_path() : fs::path(opt.cache_dir);
    std::ostringstream n;
    n << op.stem().string() << "-" << file_sha1(onnx_path).substr(0, 12) << "-" << gpu_slug()
      << "-trt" << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
      << "-" << (opt.fp16 ? "fp16" : "fp32") << ".engine";
    return (dir / n.str()).string();
}

std::string TrtBackend::build_engine(const std::string& onnx_path, const std::string& engine_path,
                                     const TrtOptions& opt) {
#ifndef USE_TRT_ONNXPARSER
    (void)onnx_path; (void)engine_path; (void)opt;
    throw std::runtime_error("engine building needs nvonnxparser (rebuild with TensorRT's onnx parser); "
                             "pass a prebuilt .engine instead");
#else
    namespace fs = std::filesystem;
    auto t0 = clk::now();
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger));
    if (!builder) throw std::runtime_error("TensorRT builder init failed");
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_logger));
    if (!parser->parseFromFile(onnx_path.c_str(),
                               int(opt.verbose ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING))) {
        std::string msg = "ONNX parse failed: " + onnx_path;
        for (int i = 0; i < parser->getNbErrors(); ++i) msg += std::string("\n  ") + parser->getError(i)->desc();
        throw std::runtime_error(msg);
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(opt.workspace_mb) << 20);
    if (opt.fp16) {
        if (!builder->platformHasFastFp16()) std::cerr << "[trt] warn: platform has no fast fp16; building fp16 anyway\n";
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
    if (!plan) throw std::runtime_error("TensorRT engine build failed for " + onnx_path);
    std::error_code ec;
    fs::create_directories(fs::path(engine_path).parent_path(), ec);
    const std::string tmp = engine_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write engine: " + tmp);
        f.write(static_cast<const char*>(plan->data()), std::streamsize(plan->size()));
    }
    fs::rename(tmp, engine_path, ec);
    if (ec) throw std::runtime_error("cannot move engine into place: " + engine_path);
    std::cerr << "[trt] built " << (opt.fp16 ? "fp16" : "fp32") << " engine in " << ms_since(t0) / 1000.0
              << "s -> " << engine_path << " (" << plan->size() / (1024 * 1024) << " MB)\n";
    return engine_path;
#endif
}

TrtBackend::TrtBackend(const std::string& model_path, const TrtOptions& opt) {
    namespace fs = std::filesystem;
    if (ends_with(model_path, ".onnx")) {
        const std::string ep = cached_engine_path(model_path, opt);
        std::error_code ec;
        if (!fs::exists(ep, ec)) build_engine(model_path, ep, opt);
        else std::cerr << "[trt] cached engine: " << ep << "\n";
        // sidecar metadata for the engine (names/imgsz/end2end) comes from the onnx's siblings:
        // <onnx-minus-ext>.metadata.yaml or metadata.yaml next to the onnx. Copy it next to the
        // engine once so the engine dir is self-describing.
        const fs::path eng(ep);
        const fs::path side = fs::path(eng).replace_extension(".metadata.yaml");
        if (!fs::exists(side, ec)) {
            for (const fs::path& p : { fs::path(model_path).replace_extension(".metadata.yaml"),
                                       fs::path(model_path).parent_path() / "metadata.yaml" }) {
                if (fs::exists(p, ec)) { fs::copy_file(p, side, fs::copy_options::overwrite_existing, ec); break; }
            }
        }
        load_engine(ep);
        active_ep = opt.fp16 ? "TRT-CUDA-fp16" : "TRT-CUDA-fp32";
    } else {
        load_engine(model_path);
    }
}

TrtBackend::TrtBackend(const std::string& engine_path) { load_engine(engine_path); }

void TrtBackend::load_engine(const std::string& engine_path_) {
    const std::string& engine_path = engine_path_;
    this->engine_path = engine_path;
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!runtime_)
        throw std::runtime_error("TensorRT runtime init failed (CUDA/GPU unavailable? "
                                 "check the driver: on Jetson, nvgpu module + reboot; "
                                 "in containers, --runtime nvidia)");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error("failed to deserialize engine (built for a different GPU arch / TRT version?)");
    ctx_.reset(engine_->createExecutionContext());
    if (!ctx_)
        throw std::runtime_error("TensorRT execution context creation failed (out of GPU memory?)");
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // discover I/O tensors (TensorRT 10 named-tensor API): input [1,3,H,W];
    // the rank-3 output is the detection head, a rank-4 output is a seg proto tensor.
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* nm = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(nm);
        if (engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT) {
            in_name_ = nm; in_sz_ = dims.d[2];                                // [1,3,H,W]
        } else if (dims.nbDims == 4) {
            proto_name_ = nm;
            pc_ = dims.d[1]; ph_ = dims.d[2]; pw_ = dims.d[3];                // [1,nm,mh,mw]
        } else {
            out_name_ = nm; feat_dim_ = dims.d[1]; num_anchors_ = dims.d[2];  // [1,feat,anchors]
        }
    }
    if (in_sz_ <= 0 || feat_dim_ <= 0 || num_anchors_ <= 0)
        throw std::runtime_error("unexpected engine I/O shape");
    fixed_imgsz = in_sz_;
    active_ep = "TRT-CUDA";

    // metadata sidecar (engines embed no names/imgsz): <engine-minus-ext>.metadata.yaml,
    // then metadata.yaml next to the engine. Same format as the ncnn/mnn exports, so the
    // parser is shared. --classes on the CLI still overrides.
    {
        namespace fs = std::filesystem;
        const fs::path ep(engine_path);
        for (const fs::path& p : { fs::path(ep).replace_extension(".metadata.yaml"),
                                   ep.parent_path() / "metadata.yaml" }) {
            std::vector<std::string> names; int misz = 0; bool e2e = false;
            std::error_code ec;
            if (fs::exists(p, ec) && meta::read_ncnn_yaml(p.string(), names, misz, e2e)) {
                meta_names = std::move(names);
                meta_imgsz = misz;
                end2end_ = e2e;
                if (misz > 0 && misz != in_sz_)
                    std::cerr << "[trt] warn: sidecar imgsz=" << misz << " but engine input is "
                              << in_sz_ << "px (" << p.string() << ")\n";
                break;
            }
        }
    }
    if (end2end_ || looks_end2end(feat_dim_, num_anchors_))
        std::cerr << "[trt] end2end model: NMS-free [num_det,6] output\n";

    CUDA_CHECK(cudaMalloc(&d_in_,  size_t(3) * in_sz_ * in_sz_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_, size_t(feat_dim_) * num_anchors_ * sizeof(float)));
    h_out_.resize(size_t(feat_dim_) * num_anchors_);
    ctx_->setTensorAddress(in_name_.c_str(),  d_in_);
    ctx_->setTensorAddress(out_name_.c_str(), d_out_);
    if (pc_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_proto_, size_t(pc_) * ph_ * pw_ * sizeof(float)));
        h_proto_.resize(size_t(pc_) * ph_ * pw_);
        ctx_->setTensorAddress(proto_name_.c_str(), d_proto_);
    }
}

TrtBackend::~TrtBackend() {
    if (d_in_)    cudaFree(d_in_);
    if (d_out_)   cudaFree(d_out_);
    if (d_proto_) cudaFree(d_proto_);
    if (stream_)  cudaStreamDestroy(stream_);
}

std::vector<Detection> TrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    auto t0 = clk::now();
    LetterboxInfo lb;
    std::vector<float>& in = h_in_;                    // persistent host staging (no per-frame allocation)
    in.resize(static_cast<size_t>(3) * in_sz_ * in_sz_);
    preprocess_nchw(bgr, in_sz_, cfg.stretch, in.data(), lb);
    pre_ms = ms_since(t0);

    auto t1 = clk::now();
    CUDA_CHECK(cudaMemcpyAsync(d_in_, in.data(), in.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("TRT enqueueV3 failed");
    CUDA_CHECK(cudaMemcpyAsync(h_out_.data(), d_out_, h_out_.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    if (pc_ > 0)
        CUDA_CHECK(cudaMemcpyAsync(h_proto_.data(), d_proto_, h_proto_.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    infer_ms = ms_since(t1);

    // "forward once, tune cheap": cache the pre-NMS candidates + letterbox (+ proto for
    // seg engines) so slicing, cached re-NMS and annotation export work like every other
    // backend (mirrors ort_backend.cpp).
    auto t2 = clk::now();
    if (end2end_ || looks_end2end(feat_dim_, num_anchors_))   // [1, num_det, 6] NMS-free
        candidates = decode_end2end(h_out_.data(), feat_dim_, cfg, lb);
    else
        candidates = decode_candidates(h_out_.data(), feat_dim_, num_anchors_, cfg, lb);
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    if (pc_ > 0) {
        proto = h_proto_;
        proto_c = pc_; proto_h = ph_; proto_w = pw_;
    } else {
        proto.clear();
        proto_c = proto_h = proto_w = 0;
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
