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
    // MNN has no built-in metadata map -> read an optional class-name sidecar next to the model.
    const std::string dir = std::filesystem::path(model_path).parent_path().string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(dir + "/metadata.yaml", nm, mi)) {
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
    cv::Mat padded = letterbox(bgr, cfg.imgsz, lb);   // imgsz x imgsz, CV_8UC3 BGR
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

    // ---- postprocess: pull output to host, present channel-major [feat_dim, num_anchors] ----
    auto t2 = clk::now();
    MNN::Tensor outHost(output_, MNN::Tensor::CAFFE);
    output_->copyToHostTensor(&outHost);
    const auto os = outHost.shape();                    // expect {1, feat, anchors}
    const int feat = 4 + cfg.num_classes();
    const float* raw = outHost.host<float>();
    int feat_dim = feat, num_anchors = 0;
    const float* dec = raw;
    std::vector<float> buf;
    if (os.size() == 3 && os[1] == feat) {              // [1, feat, anchors] channel-major (expected)
        feat_dim = os[1]; num_anchors = os[2];
    } else if (os.size() == 3 && os[2] == feat) {       // [1, anchors, feat] -> transpose
        num_anchors = os[1];
        buf.resize(static_cast<size_t>(feat) * num_anchors);
        for (int a = 0; a < num_anchors; ++a)
            for (int f = 0; f < feat; ++f)
                buf[static_cast<size_t>(f) * num_anchors + a] = raw[static_cast<size_t>(a) * feat + f];
        dec = buf.data();
    } else {                                            // fallback: assume channel-major, infer anchor count
        num_anchors = static_cast<int>(outHost.elementSize() / feat);
    }
    auto dets = decode(dec, feat_dim, num_anchors, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
