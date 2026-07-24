// Shared backend construction — used by both the CLI (main.cpp) and the GUI so the
// two never drift. Header-only; guarded by the same USE_* defines as CMake sets.
#pragma once
#include <memory>
#include <string>
#include <system_error>
#include <filesystem>
#include "yolomaster.hpp"

#ifdef USE_ORT
#include "ort_backend.hpp"
#endif
#ifdef USE_NCNN
#include "ncnn_backend.hpp"
#endif
#ifdef USE_MNN
#include "mnn_backend.hpp"
#endif
#ifdef USE_TRT
#include "trt_backend.hpp"
#endif

namespace yolomaster {

// Infer backend name from a model path ("" if undecidable).
inline std::string detect_backend(const std::string& model) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto ends = [&](const char* s) {
        const std::string x = s; return model.size() >= x.size() &&
            model.compare(model.size() - x.size(), x.size(), x) == 0;
    };
    if (fs::is_directory(model, ec) || ends(".param")) return "ncnn";
    if (ends(".onnx")) return "onnx";
    if (ends(".mnn"))  return "mnn";
    if (ends(".engine") || ends(".trt")) return "trt";
    return "";
}

// Construct a backend. On failure returns nullptr and fills `err`. `backend` may be
// "auto" (detected from the path). `device` is onnx-only ("cpu|cuda|coreml|trt").
inline std::unique_ptr<Backend> make_backend(std::string model, std::string backend,
                                             int threads, const std::string& device,
                                             std::string& resolved, std::string& err) {
    namespace fs = std::filesystem;
    if (backend == "auto") {
        backend = detect_backend(model);
        if (backend.empty()) { err = "cannot infer backend from '" + model + "'"; return nullptr; }
    }
    resolved = backend;
    // GPU maps to each backend's native accelerator: onnx->CUDA EP, ncnn->Vulkan, mnn->OpenCL.
    const bool want_gpu = (device == "gpu" || device == "cuda" || device == "vulkan" || device == "opencl");
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            std::string ep = want_gpu ? "cuda" : (device.empty() ? "cpu" : device);
            return std::make_unique<OrtBackend>(model, threads, ep);
#else
            err = "built without ONNXRuntime backend"; return nullptr;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            std::string param = model, bin;
            std::error_code ec;
            if (fs::is_directory(model, ec)) {
                param = (fs::path(model) / "model.ncnn.param").string();
                bin   = (fs::path(model) / "model.ncnn.bin").string();
            } else bin = param.substr(0, param.rfind('.')) + ".bin";
            return std::make_unique<NcnnBackend>(param, bin, threads, want_gpu);   // want_gpu = Vulkan
#else
            err = "built without ncnn backend"; return nullptr;
#endif
        } else if (backend == "mnn") {
#ifdef USE_MNN
            std::string fwd = "cpu";
            if (device == "vulkan") fwd = "vulkan";
            else if (device == "cuda") fwd = "cuda";
            else if (want_gpu) fwd = "opencl";
            return std::make_unique<MnnBackend>(model, threads, fwd);
#else
            err = "built without MNN backend (rebuild with -DUSE_MNN=ON)"; return nullptr;
#endif
        } else if (backend == "trt") {
#ifdef USE_TRT
            return std::make_unique<TrtBackend>(model);
#else
            err = "built without TensorRT backend"; return nullptr;
#endif
        }
        err = "unknown backend: " + backend; return nullptr;
    } catch (const std::exception& e) {
        err = std::string("backend init failed: ") + e.what(); return nullptr;
    }
}

} // namespace yolomaster
