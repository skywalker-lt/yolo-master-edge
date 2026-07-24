#include "mnn_backend.hpp"
#include <MNN/MNNForwardType.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

MnnBackend::MnnBackend(const std::string& model_path, int threads, const std::string& forward)
    : threads_(threads) {
    interp_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path.c_str()),
        [](MNN::Interpreter* p) { if (p) MNN::Interpreter::destroy(p); });
    if (!interp_) throw std::runtime_error("MNN: failed to load " + model_path);

    MNN::ScheduleConfig sc;
    sc.numThread = threads;
    sc.type = (forward == "cuda") ? MNN_FORWARD_CUDA : MNN_FORWARD_CPU;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_High;   // FP32
    bc.power     = MNN::BackendConfig::Power_High;
    sc.backendConfig = &bc;

    session_ = interp_->createSession(sc);
    if (!session_) throw std::runtime_error("MNN: createSession failed for " + model_path);
    input_  = interp_->getSessionInput(session_, nullptr);    // first input
    output_ = interp_->getSessionOutput(session_, nullptr);   // first output
    if (!input_ || !output_) throw std::runtime_error("MNN: could not resolve input/output tensor");
    active_ep = (forward == "cuda") ? "MNN-CUDA" : "MNN-CPU";

    // YOLO-Master graphs bake the attention token counts at the training size -> fixed input.
    auto ishape = input_->shape();   // NCHW, e.g. {1,3,640,640}
    if (ishape.size() == 4 && ishape[2] == ishape[3] && ishape[2] > 0) {
        fixed_imgsz = ishape[2];
        meta_imgsz  = ishape[2];
    }
    // MNN has no built-in metadata map -> read an optional class-name sidecar. Prefer a per-model
    // "<model>.metadata.yaml" (so several .mnn can share a dir); fall back to "metadata.yaml".
    const std::filesystem::path mp(model_path);
    const std::string per_model = (mp.parent_path() / (mp.stem().string() + ".metadata.yaml")).string();
    const std::string shared    = (mp.parent_path() / "metadata.yaml").string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(per_model, nm, mi) || meta::read_ncnn_yaml(shared, nm, mi)) {
        meta_names = nm;
        if (mi > 0) { meta_imgsz = mi; if (fixed_imgsz == 0) fixed_imgsz = mi; }
    }
}

MnnBackend::~MnnBackend() {
    if (interp_ && session_) interp_->releaseSession(session_);   // interp_ freed by shared_ptr deleter
}

std::vector<Detection> MnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 (identical to ORT) ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);   // imgsz x imgsz, CV_8UC3 BGR
    const int sz = cfg.imgsz, hw = sz * sz;
    std::vector<float> blob(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;           // BGR
            const int idx = y * sz + x;
            blob[idx]          = px[2] * (1.0f / 255);  // R
            blob[hw + idx]     = px[1] * (1.0f / 255);  // G
            blob[2 * hw + idx] = px[0] * (1.0f / 255);  // B
        }
    }
    // resize the session input if it doesn't already match imgsz (handles fixed & flexible graphs)
    auto ishape = input_->shape();
    if (ishape.size() != 4 || ishape[2] != sz || ishape[3] != sz) {
        interp_->resizeTensor(input_, std::vector<int>{1, 3, sz, sz});
        interp_->resizeSession(session_);
        output_ = interp_->getSessionOutput(session_, nullptr);
    }
    pre_ms = ms_since(t0);

    // ---- inference: copy blob into the input tensor (NCHW/Caffe), run ----
    auto t1 = clk::now();
    {
        MNN::Tensor host(input_, MNN::Tensor::CAFFE);   // NCHW host tensor shaped like input_
        std::memcpy(host.host<float>(), blob.data(), blob.size() * sizeof(float));
        input_->copyFromHostTensor(&host);
    }
    interp_->runSession(session_);
    infer_ms = ms_since(t1);

    // ---- postprocess: detection = rank-3 output [1,feat,anchors]; proto (seg) = rank-4 [1,nm,mh,mw] ----
    auto t2 = clk::now();
    auto all = interp_->getSessionOutputAll(session_);
    MNN::Tensor* detT = nullptr; MNN::Tensor* protoT = nullptr;
    for (auto& kv : all) {
        if (kv.second->shape().size() == 4) protoT = kv.second;
        else                                detT   = kv.second;
    }
    if (!detT) detT = output_;                          // detection-only safety
    MNN::Tensor detHost(detT, MNN::Tensor::CAFFE);
    detT->copyToHostTensor(&detHost);
    const auto os = detHost.shape();
    const float* raw = detHost.host<float>();
    int feat_dim = 0, num_anchors = 0;
    const float* dec = raw;
    std::vector<float> buf;
    if (os.size() == 3) {
        if (os[1] <= os[2]) { feat_dim = os[1]; num_anchors = os[2]; }   // channel-major (expected)
        else {                                                          // [1,anchors,feat] -> transpose
            feat_dim = os[2]; num_anchors = os[1];
            buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
            for (int a = 0; a < num_anchors; ++a)
                for (int f = 0; f < feat_dim; ++f)
                    buf[static_cast<size_t>(f) * num_anchors + a] = raw[static_cast<size_t>(a) * feat_dim + f];
            dec = buf.data();
        }
    }
    candidates = decode_candidates(dec, feat_dim, num_anchors, cfg, lb);
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (protoT) {                                       // segmentation proto
        MNN::Tensor protoHost(protoT, MNN::Tensor::CAFFE);
        protoT->copyToHostTensor(&protoHost);
        const auto ps = protoHost.shape();              // {1, nm, mh, mw}
        proto_c = (int)ps[1]; proto_h = (int)ps[2]; proto_w = (int)ps[3];
        const float* pp = protoHost.host<float>();
        proto.assign(pp, pp + (size_t)proto_c * proto_h * proto_w);
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
