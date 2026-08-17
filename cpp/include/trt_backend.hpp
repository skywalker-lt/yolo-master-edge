// TensorRT backend for YOLO-Master - GPU inference from a prebuilt .engine.
// Loads an engine built on-device by trtexec (jetson/10_trt_bench.sh) and runs it on CUDA.
// Detection engines have one output [1,feat,anchors]; segmentation engines add a proto
// output [1,nm,mh,mw] (both discovered by rank). Class names / imgsz come from an
// optional metadata.yaml sidecar (engines embed no metadata):
// <engine-minus-ext>.metadata.yaml, or metadata.yaml next to the engine.
#pragma once
#include "yolomaster.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace yolomaster {

class TrtBackend : public Backend {
public:
    explicit TrtBackend(const std::string& engine_path);
    ~TrtBackend() override;
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    void* d_in_    = nullptr;
    void* d_out_   = nullptr;
    void* d_proto_ = nullptr;              // seg engines only
    std::string in_name_, out_name_, proto_name_;
    int in_sz_ = 0;                        // input H (== W)
    int feat_dim_ = 0, num_anchors_ = 0;   // detection output [1, feat_dim, num_anchors]
    bool end2end_ = false;                 // NMS-free [1, num_det, 6] head (sidecar or shape)
    int pc_ = 0, ph_ = 0, pw_ = 0;         // proto output [1, pc, ph, pw] (0 = detection engine)
    std::vector<float> h_out_, h_proto_;
};

} // namespace yolomaster
