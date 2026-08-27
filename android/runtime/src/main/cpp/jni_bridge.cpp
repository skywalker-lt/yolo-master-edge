// JNI bridge for the YOLO-Master ncnn runtime on Android.
//
// It wraps exactly one seam from the shared C++ core: Backend::infer(cv::Mat, Config).
// Nothing from the CLI driver (main.cpp), videoio, or the filesystem source layer is
// pulled in. Robustness rules encoded here:
//   * CPU path is fp32 (inherited from ncnn_backend.cpp:23-33) or ARM fp16 zeroes the
//     mixture routing -> no detections. We never re-enable fp16 on CPU.
//   * Vulkan is verified available (get_gpu_count) before it is requested; otherwise we
//     transparently fall back to CPU-fp32 (reported via activeBackend).
//   * Every entry point is try/catch -> typed error string, never a native crash.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolomaster.hpp"
#include "ncnn_backend.hpp"   // pulls ncnn net.h -> platform.h (defines NCNN_VULKAN)
#include "cpu.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

using namespace yolomaster;

#define LOG_TAG "YMNcnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Handle {
    std::unique_ptr<NcnnBackend> be;
    Config cfg;
    std::string activeBackend;
    bool vulkan = false;
};

// The ncnn Vulkan instance is process-global; ref-count it across handles.
std::mutex g_gpu_mu;
int g_gpu_refs = 0;
thread_local std::string g_last_error;

// Try to make a usable Vulkan GPU available. Returns false (and stays CPU) when ncnn
// was built without Vulkan, no device exists, or instance creation fails.
bool acquire_gpu() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> lk(g_gpu_mu);
    if (g_gpu_refs == 0 && ncnn::create_gpu_instance() != 0) return false;
    if (ncnn::get_gpu_count() <= 0) {
        if (g_gpu_refs == 0) ncnn::destroy_gpu_instance();
        return false;
    }
    ++g_gpu_refs;
    return true;
#else
    return false;
#endif
}

void release_gpu() {
#if NCNN_VULKAN
    std::lock_guard<std::mutex> lk(g_gpu_mu);
    if (g_gpu_refs > 0 && --g_gpu_refs == 0) ncnn::destroy_gpu_instance();
#endif
}

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return r;
}

// Android ARGB_8888 Bitmap -> cv::Mat BGR (deep-copied out before unlock). Empty on failure.
cv::Mat bitmap_to_bgr(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return {};
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return {};
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return {};
    cv::Mat bgr;
    try {
        cv::Mat rgba((int)info.height, (int)info.width, CV_8UC4, pixels, info.stride);
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);   // copies out of the locked buffer
    } catch (...) { /* fall through to unlock */ }
    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInit(JNIEnv* env, jobject, jstring jModelDir,
                                                   jboolean useVulkan, jint threads) {
    g_last_error.clear();
    const std::string dir = jstr(env, jModelDir);
    const std::string param = dir + "/model.ncnn.param";
    const std::string bin = dir + "/model.ncnn.bin";

    bool haveVk = false;
    if (useVulkan) {
        haveVk = acquire_gpu();
        if (!haveVk) LOGI("Vulkan requested but unavailable; using CPU-fp32");
    }
    const int th = threads > 0 ? (int)threads : std::max(1, ncnn::get_big_cpu_count());

    try {
        auto h = std::make_unique<Handle>();
        h->vulkan = haveVk;
        h->be = std::make_unique<NcnnBackend>(param, bin, th, haveVk);
        // Build the inference Config from the model's own metadata (mirrors main.cpp:163-174):
        // the ncnn graph bakes attention token counts at the training imgsz, so it is fixed.
        Config& c = h->cfg;
        c.imgsz = h->be->fixed_imgsz > 0 ? h->be->fixed_imgsz
                  : (h->be->meta_imgsz > 0 ? h->be->meta_imgsz : 640);
        c.class_names = h->be->meta_names;  // may be empty -> labels fall back to the class index
        h->activeBackend = haveVk ? "ncnn-Vulkan" : "ncnn-CPU-fp32";
        LOGI("init ok: %s imgsz=%d classes=%zu threads=%d", h->activeBackend.c_str(), c.imgsz,
             c.class_names.size(), th);
        return reinterpret_cast<jlong>(h.release());
    } catch (const std::exception& e) {
        g_last_error = e.what();
        LOGE("init failed: %s", e.what());
        if (haveVk) release_gpu();
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeSetConfig(JNIEnv*, jobject, jlong handle, jfloat conf,
                                                        jfloat iou, jint maxDet) {
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h) return;
    h->cfg.conf_thresh = conf;
    h->cfg.iou_thresh = iou;
    if (maxDet > 0) h->cfg.max_det = maxDet;
}

// Returns a flat float[]: [n, then n*6 = (x1,y1,x2,y2,conf,cls) per detection]. null on error.
JNIEXPORT jfloatArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeInfer(JNIEnv* env, jobject, jlong handle,
                                                    jobject bitmap) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return nullptr; }
    cv::Mat bgr = bitmap_to_bgr(env, bitmap);
    if (bgr.empty()) { g_last_error = "bitmap must be ARGB_8888 and lockable"; return nullptr; }

    std::vector<Detection> dets;
    try {
        dets = h->be->infer(bgr, h->cfg);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }

    const int n = (int)dets.size();
    std::vector<float> buf(1 + (size_t)n * 6);
    buf[0] = (float)n;
    for (int i = 0; i < n; ++i) {
        const auto& d = dets[i];
        float* p = &buf[1 + (size_t)i * 6];
        p[0] = d.box.x;
        p[1] = d.box.y;
        p[2] = d.box.x + d.box.width;
        p[3] = d.box.y + d.box.height;
        p[4] = d.conf;
        p[5] = (float)d.class_id;
    }
    jfloatArray arr = env->NewFloatArray((jsize)buf.size());
    if (!arr) { g_last_error = "oom"; return nullptr; }
    env->SetFloatArrayRegion(arr, 0, (jsize)buf.size(), buf.data());
    return arr;
}

// Segmentation overlay built from the LAST infer's cached candidates/proto (no new forward,
// "forward once, tune cheap"). Writes [w,h] into dimsOut and returns an RGBA byte[] (w*h*4).
// null if the model is not segmentation or nothing has been inferred yet.
JNIEXPORT jbyteArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeSegOverlay(JNIEnv* env, jobject, jlong handle,
                                                         jintArray dimsOut) {
    g_last_error.clear();
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h || !h->be) { g_last_error = "null handle"; return nullptr; }
    auto& be = *h->be;
    if (!be.is_seg() || be.candidates.empty()) { g_last_error = "no segmentation output cached"; return nullptr; }

    cv::Mat rgba;
    try {
        std::vector<Detection> dets = nms_and_cap(be.candidates, h->cfg, be.cand_orig_w, be.cand_orig_h);
        rgba = seg_overlay(dets, be.proto, be.proto_c, be.proto_h, be.proto_w, be.cand_lb, h->cfg.imgsz,
                           be.cand_orig_w, be.cand_orig_h);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
    if (rgba.empty()) { g_last_error = "empty overlay"; return nullptr; }

    cv::Mat cont = rgba.isContinuous() ? rgba : rgba.clone();
    const jsize len = (jsize)(cont.total() * cont.elemSize());
    jbyteArray out = env->NewByteArray(len);
    if (!out) { g_last_error = "oom"; return nullptr; }
    env->SetByteArrayRegion(out, 0, len, reinterpret_cast<const jbyte*>(cont.data));
    if (dimsOut && env->GetArrayLength(dimsOut) >= 2) {
        jint d[2] = {be.cand_orig_w, be.cand_orig_h};
        env->SetIntArrayRegion(dimsOut, 0, 2, d);
    }
    return out;
}

JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeActiveBackend(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    return env->NewStringUTF(h ? h->activeBackend.c_str() : "");
}

JNIEXPORT jobjectArray JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeMetaNames(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    static const std::vector<std::string> kEmpty;
    const std::vector<std::string>& names = h ? h->cfg.class_names : kEmpty;
    jclass strCls = env->FindClass("java/lang/String");
    jobjectArray arr = env->NewObjectArray((jsize)names.size(), strCls, nullptr);
    for (jsize i = 0; i < (jsize)names.size(); ++i) {
        jstring s = env->NewStringUTF(names[i].c_str());
        env->SetObjectArrayElement(arr, i, s);
        env->DeleteLocalRef(s);
    }
    return arr;
}

JNIEXPORT jstring JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

JNIEXPORT void JNICALL
Java_dev_yolomaster_ncnn_YoloMasterNcnn_nativeRelease(JNIEnv*, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h) return;
    const bool vk = h->vulkan;
    delete h;
    if (vk) release_gpu();
}

}  // extern "C"
