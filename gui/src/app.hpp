// Portable GUI application state + logic (no Win32 / no D3D11).
// The platform layer (main_win.cpp) provides texture upload + a file-open dialog
// through the Platform struct, then calls App::frame() once per rendered frame.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "yolomaster.hpp"

namespace gui {

// A GPU texture handle owned by the platform layer. `id` is an ImTextureID (an SRV
// on D3D11) stored type-erased so this header stays platform-free.
struct Texture {
    void* id = nullptr;
    int   w = 0, h = 0;
};

// Services the platform layer injects into the portable app.
struct Platform {
    // Upload/replace `rgba` (CV_8UC4) into `tex`, (re)allocating if size changed. false on failure.
    std::function<bool(const cv::Mat& rgba, Texture& tex)> upload;
    // Native open-file dialog; returns "" if cancelled. `filter` is a platform-specific spec.
    std::function<std::string(const char* title, const char* filter)> open_file;
    // Native folder-picker; returns "" if cancelled.
    std::function<std::string(const char* title)> open_folder;
};

enum class BoxStyle { Hud, Solid, Neon };
enum class LabelMode { Full, Min, Off };
enum class Preprocess { Letterbox, Stretch };
enum class Overlay { Both, Masks, Boxes };   // segmentation: what to show (masks / boxes / both)
enum class Device { CPU, GPU };              // GPU = onnx:CUDA, ncnn:Vulkan, mnn:OpenCL/Vulkan

class App {
public:
    ~App();
    void frame(const Platform& plat);   // draw one ImGui frame (sidebar + preview)

private:
    // ---- backend / model ----
    char        model_path_[1024] = "";
    int         backend_sel_ = 0;       // 0 auto,1 onnx,2 ncnn,3 mnn
    int         threads_ = 4;
    std::unique_ptr<yolomaster::Backend> be_;
    std::string be_name_, be_err_, be_ep_;
    yolomaster::Config cfg_;            // display config (real conf/iou live here)

    // ---- source image ----
    cv::Mat     img_bgr_;              // original loaded image
    Texture     img_tex_;             // uploaded to GPU once per load
    std::string img_path_, load_err_;

    // ---- folder-batch ----
    std::vector<std::string> folder_imgs_;   // sorted image paths (empty = single-image mode)
    int         cur_idx_ = -1;               // index into folder_imgs_ of the shown image
    std::string folder_path_;
    bool        scroll_to_cur_ = false;      // request the file list to scroll the current item into view

    // ---- video ----
    cv::VideoCapture cap_;
    bool        is_video_ = false;
    bool        playing_ = false;
    int         frame_idx_ = 0, total_frames_ = 0;
    double      video_fps_ = 30.0, play_accum_ = 0.0;   // play_accum_ paces playback to real time
    std::string video_path_;

    // ---- webcam ----
    cv::VideoCapture cam_;
    bool        is_cam_ = false;
    bool        cam_mirror_ = true;
    double      cam_fps_ema_ = 0.0, cam_ms_ema_ = 0.0;  // EMA-smoothed feed fps / inference ms

    // ---- async inference worker (video + webcam): decode/display on main, infer off-thread ----
    std::thread worker_;
    std::mutex  job_mtx_;
    std::condition_variable job_cv_;
    cv::Mat     job_frame_;
    yolomaster::Config job_cfg_;
    bool        job_pending_ = false, worker_quit_ = false;
    std::atomic<bool> worker_busy_{false};
    std::mutex  res_mtx_;
    struct AsyncResult {
        std::vector<yolomaster::Detection> dets;
        std::vector<yolomaster::RawDet>    cands;        // for cheap re-NMS when paused
        std::vector<float> proto; int pc = 0, ph = 0, pw = 0;
        yolomaster::LetterboxInfo lb; int ow = 0, oh = 0;
        double inf_ms = 0;
        bool seg = false;
    };
    AsyncResult ares_;
    bool        ares_dirty_ = false;
    bool        async_mode_ = false;   // true while a video/webcam worker owns the backend
    bool        seg_model_ = false;    // stable is_seg() flag (safe to read from the main thread)

    // ---- results ("forward once, tune cheap") ----
    std::vector<yolomaster::Detection> dets_;
    Texture mask_tex_;                // segmentation overlay (RGBA), rebuilt when dets change
    bool  has_mask_ = false;
    bool  need_reinfer_ = false;      // model/source/preprocess/threads changed
    bool  need_renms_   = false;      // conf/iou changed (cheap: re-run nms on cached candidates)
    bool  need_overlay_ = false;      // dets changed -> recomposite the seg mask overlay
    double pre_ms_ = 0, inf_ms_ = 0, post_ms_ = 0;
    std::vector<int> class_counts_;

    // ---- appearance ----
    float      conf_ = 0.25f, iou_ = 0.50f;
    BoxStyle   style_ = BoxStyle::Hud;
    LabelMode  labels_ = LabelMode::Full;
    Preprocess prep_ = Preprocess::Letterbox;
    Overlay    overlay_ = Overlay::Both;
    Device     device_ = Device::CPU;

    static constexpr float kConfFloor = 0.05f;   // cache candidates down to here

    void load_model();
    void load_image(const std::string& path, const Platform& plat);
    void load_folder(const std::string& dir, const Platform& plat);
    void select_index(int i, const Platform& plat);   // load folder_imgs_[i]
    void open_media(const std::string& path, const Platform& plat);   // autodetect image vs video
    void open_video(const std::string& path, const Platform& plat);
    void show_frame(const cv::Mat& frame, const Platform& plat);   // upload frame texture
    void seek_video(int idx, const Platform& plat);   // random-access seek + read
    bool advance_video(const Platform& plat);         // sequential next frame; false at end
    void close_video();
    // webcam + async worker
    void open_camera(const Platform& plat);
    void close_camera();
    void start_worker();
    void stop_worker();
    void worker_loop();
    void submit_job(const cv::Mat& frame);            // non-blocking; drops if worker busy
    void consume_async(const Platform& plat);         // pull latest result -> dets_ / overlay
    void build_overlay(const std::vector<yolomaster::Detection>& dets, const std::vector<float>& proto,
                       int pc, int ph, int pw, const yolomaster::LetterboxInfo& lb,
                       int ow, int oh, const Platform& plat);
    std::string gpu_device_str() const;               // backend-specific GPU token for the Device
    void run_inference();             // full forward pass -> cache candidates
    void recompute_nms();             // cheap: nms_and_cap on cached candidates
    void rebuild_overlay(const Platform& plat);   // recomposite seg masks -> mask_tex_
    void draw_sidebar(const Platform& plat);
    void draw_filelist(const Platform& plat);   // folder-batch navigator panel
    void draw_preview(const Platform& plat);
    void draw_transport(const Platform& plat);  // video play/pause + scrubber
    void draw_camera_hud();                     // webcam fps/latency/objects overlay
};

} // namespace gui
