#include "trt_backend.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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

TrtBackend::TrtBackend(const std::string& engine_path) {
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
    cv::Mat padded = preprocess(bgr, in_sz_, cfg.stretch, lb);   // in_sz_ x in_sz_, BGR
    const int sz = in_sz_, hw = sz * sz;
    std::vector<float> in(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;                  // BGR -> RGB /255, NCHW
            const int idx = y * sz + x;
            in[idx]        = px[2] * (1.0f / 255);
            in[hw + idx]   = px[1] * (1.0f / 255);
            in[2 * hw + idx] = px[0] * (1.0f / 255);
        }
    }
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
