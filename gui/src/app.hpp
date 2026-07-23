// Portable GUI application state + logic (no Win32 / no D3D11).
// The platform layer (main_win.cpp) provides texture upload + a file-open dialog
// through the Platform struct, then calls App::frame() once per rendered frame.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
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

class App {
public:
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

    // ---- results ("forward once, tune cheap") ----
    std::vector<yolomaster::Detection> dets_;
    bool  need_reinfer_ = false;      // model/source/preprocess/threads changed
    bool  need_renms_   = false;      // conf/iou changed (cheap: re-run nms on cached candidates)
    double pre_ms_ = 0, inf_ms_ = 0, post_ms_ = 0;
    std::vector<int> class_counts_;

    // ---- appearance ----
    float      conf_ = 0.25f, iou_ = 0.50f;
    BoxStyle   style_ = BoxStyle::Hud;
    LabelMode  labels_ = LabelMode::Full;
    Preprocess prep_ = Preprocess::Letterbox;

    static constexpr float kConfFloor = 0.05f;   // cache candidates down to here

    void load_model();
    void load_image(const std::string& path, const Platform& plat);
    void load_folder(const std::string& dir, const Platform& plat);
    void select_index(int i, const Platform& plat);   // load folder_imgs_[i]
    void run_inference();             // full forward pass -> cache candidates
    void recompute_nms();             // cheap: nms_and_cap on cached candidates
    void draw_sidebar(const Platform& plat);
    void draw_filelist(const Platform& plat);   // folder-batch navigator panel
    void draw_preview();
};

} // namespace gui
