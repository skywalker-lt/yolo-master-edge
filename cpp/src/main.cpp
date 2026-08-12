// yolomaster_edge - universal, adaptive YOLO-Master edge runner.
// Runtime model loading (no baked-in weights), backend/classes/imgsz auto-detected
// from the model, versatile --source (image / dir / video / dataset.yaml).
#include "yolomaster.hpp"
#include "slicing.hpp"
#include "annotate_export.hpp"
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
#include "CLI11.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

using namespace yolomaster;
namespace fs = std::filesystem;

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// image I/O via stb (avoids OpenCV imgcodecs -> GDAL/DB/poppler dependency closure)
static cv::Mat imread_bgr(const std::string& path) {
    int w, h, n;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &n, 3);   // force 3-channel RGB
    if (!d) return cv::Mat();
    cv::Mat bgr;
    cv::cvtColor(cv::Mat(h, w, CV_8UC3, d), bgr, cv::COLOR_RGB2BGR);
    stbi_image_free(d);
    return bgr;
}
static bool imwrite_jpg(const std::string& path, const cv::Mat& bgr) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return stbi_write_jpg(path.c_str(), rgb.cols, rgb.rows, 3, rgb.data, 90) != 0;
}

int main(int argc, char** argv) {
    CLI::App app{"yolomaster_edge - universal YOLO-Master edge runner (ONNX / ncnn / MNN)"};
    std::string model, source, backend = "auto", classes_opt = "auto", outdir = "runs_edge";
    std::string device = "cpu", savetxt;
    int imgsz = 0, threads = 4, limit = 0, max_det = 300;
    float conf = 0.25f, iou = 0.50f;
    bool no_save = false, quiet = false, multilabel = false, stretch = false;
    std::string slicing = "off", label_format = "yolo", sampling = "1s", export_labels;
    int tile_size = 0;
    bool slicing_masks = false, cw_nms = false;
    float sigma = 0.1f;

    app.add_option("-m,--model", model, "model: .onnx file, or ncnn dir / .param")->required();
    app.add_option("-s,--source", source, "image / directory / video / dataset.yaml")->required();
    app.add_option("-b,--backend", backend, "auto|onnx|ncnn|mnn")->default_str("auto");
    app.add_option("-d,--device", device, "cpu|cuda|trt|coreml (onnx backend; trt=TensorRT EP, coreml=Apple CoreML EP)")->default_str("cpu");
    app.add_option("--classes", classes_opt, "auto|visdrone|sku (auto = from model metadata)")->default_str("auto");
    app.add_option("--imgsz", imgsz, "inference size (0 = from model / 640)");
    app.add_option("--conf", conf, "confidence threshold")->capture_default_str();
    app.add_option("--iou", iou, "NMS IoU threshold")->capture_default_str();
    app.add_option("--max-det", max_det, "max detections per image after NMS")->capture_default_str();
    app.add_option("--threads", threads, "CPU threads")->capture_default_str();
    app.add_option("--limit", limit, "cap #inputs (0 = all)");
    app.add_option("--out", outdir, "output dir for annotated results")->capture_default_str();
    app.add_option("--save-txt", savetxt, "dir to write per-image predictions ('class conf x1 y1 x2 y2')");
    app.add_flag("--multi-label", multilabel, "one detection per class>conf per anchor (matches ultralytics val mAP)");
    app.add_flag("--stretch", stretch, "preprocess by stretching to square instead of aspect-preserving letterbox");
    app.add_flag("--no-save", no_save, "do not write annotated outputs");
    app.add_flag("--quiet", quiet, "suppress per-image logs");
    app.add_option("--slicing", slicing, "off|dense|sparse: global pass + tile passes (Sparse SAHI); images/dirs only")->default_str("off");
    app.add_option("--tile-size", tile_size, "requested tile edge in source px (0 = model imgsz); clamped per image to [imgsz, max(imgsz, shortSide/4)]");
    app.add_flag("--slicing-masks", slicing_masks, "keep the global pass's masks (+proto) in sliced runs (seg models)");
    app.add_flag("--cw-nms", cw_nms, "Cluster-Weighted NMS: refine survivor boxes by their cluster's weighted average");
    app.add_option("--sigma", sigma, "CW-NMS weight falloff (0.01-0.5)")->capture_default_str();
    app.add_option("--export-labels", export_labels, "dir to write annotation labels (WYSIWYG at the current conf/iou/nms settings)");
    app.add_option("--label-format", label_format, "yolo|coco|voc")->default_str("yolo");
    app.add_option("--sampling", sampling, "video label export: all|1s|N (every Nth frame)")->default_str("1s");
    CLI11_PARSE(app, argc, argv);

    SliceMode slice_mode = SliceMode::Off;
    if (slicing == "dense") slice_mode = SliceMode::Dense;
    else if (slicing == "sparse") slice_mode = SliceMode::Sparse;
    else if (slicing != "off") { std::cerr << "unknown --slicing mode: " << slicing << "\n"; return 2; }
    annot::Format lfmt = annot::Format::YoloTXT;
    if (label_format == "coco") lfmt = annot::Format::CocoJSON;
    else if (label_format == "voc") lfmt = annot::Format::PascalVOC;
    else if (label_format != "yolo") { std::cerr << "unknown --label-format: " << label_format << "\n"; return 2; }

    // ---- backend auto-detect from the model path ----
    if (backend == "auto") {
        std::error_code ec;
        if (fs::is_directory(model, ec) || ends_with(model, ".param")) backend = "ncnn";
        else if (ends_with(model, ".onnx")) backend = "onnx";
        else if (ends_with(model, ".mnn")) backend = "mnn";
        else if (ends_with(model, ".engine") || ends_with(model, ".trt")) backend = "trt";
        else { std::cerr << "cannot infer backend from '" << model << "'; pass --backend\n"; return 2; }
    }

    // ---- construct backend ----
    std::unique_ptr<Backend> be;
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            be = std::make_unique<OrtBackend>(model, threads, device);
#else
            std::cerr << "built without ONNXRuntime backend\n"; return 2;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            std::string param = model, bin;
            std::error_code ec;
            if (fs::is_directory(model, ec)) {
                param = (fs::path(model) / "model.ncnn.param").string();
                bin = (fs::path(model) / "model.ncnn.bin").string();
            } else bin = param.substr(0, param.rfind('.')) + ".bin";
            be = std::make_unique<NcnnBackend>(param, bin, threads);
#else
            std::cerr << "built without ncnn backend\n"; return 2;
#endif
        } else if (backend == "mnn") {
#ifdef USE_MNN
            be = std::make_unique<MnnBackend>(model, threads, device == "cuda" ? "cuda" : "cpu");
#else
            std::cerr << "built without MNN backend (rebuild with -DUSE_MNN=ON)\n"; return 2;
#endif
        } else if (backend == "trt") {
#ifdef USE_TRT
            be = std::make_unique<TrtBackend>(model);
#else
            std::cerr << "built without TensorRT backend (rebuild with -DUSE_TRT=ON)\n"; return 2;
#endif
        } else { std::cerr << "unknown backend: " << backend << "\n"; return 2; }
    } catch (const std::exception& e) {
        std::cerr << "backend init failed: " << e.what() << "\n"; return 3;
    }

    // ---- resolve config: --flag > model metadata > default ----
    Config cfg;
    cfg.conf_thresh = conf;
    cfg.iou_thresh = iou;
    cfg.max_det = max_det;
    cfg.multi_label = multilabel;
    cfg.stretch = stretch;
    cfg.nms_mode = cw_nms ? NmsMode::ClusterWeighted : NmsMode::Standard;
    cfg.cw_sigma = std::min(0.5f, std::max(0.01f, sigma));
    int want = imgsz > 0 ? imgsz : (be->meta_imgsz > 0 ? be->meta_imgsz : 640);
    if (be->fixed_imgsz > 0 && want != be->fixed_imgsz) {
        std::cerr << "[warn] model requires fixed imgsz=" << be->fixed_imgsz
                  << "; overriding requested imgsz=" << want << "\n";
        want = be->fixed_imgsz;
    }
    cfg.imgsz = want;
    std::string classes_src;
    if (classes_opt == "visdrone") { cfg.class_names = visdrone_classes(); classes_src = "flag:visdrone"; }
    else if (classes_opt == "sku" || classes_opt == "sku110k") { cfg.class_names = sku110k_classes(); classes_src = "flag:sku"; }
    else if (!be->meta_names.empty()) { cfg.class_names = be->meta_names; classes_src = "model-metadata"; }
    else { cfg.class_names = visdrone_classes(); classes_src = "fallback:visdrone"; }

    std::cout << "[model] " << model << "  backend=" << backend << "  ep=" << be->active_ep
              << "  imgsz=" << cfg.imgsz << "  nc=" << cfg.num_classes() << " (" << classes_src << ")"
              << "  conf=" << cfg.conf_thresh << "  iou=" << cfg.iou_thresh << "  max_det=" << cfg.max_det << "\n";

    if (!no_save) { std::error_code ec; fs::create_directories(outdir, ec); }
    if (!savetxt.empty()) { std::error_code ec; fs::create_directories(savetxt, ec); }

    // ---- run over the source ----
    const SourceKind kind = classify_source(source);
    if (slice_mode != SliceMode::Off && kind == SourceKind::Video) {
        std::cerr << "[warn] slicing applies to images and folders only - video runs single-pass\n";
        slice_mode = SliceMode::Off;
    }
    SliceConfig sconf;
    sconf.mode = slice_mode;
    sconf.tile_size = tile_size;
    sconf.keep_global_masks = slicing_masks;
    TileStats tstats;

    // Label export: the sink is created lazily after the first forward, when the run's
    // dialect is known (WYSIWYG: seg polygons only when the backend actually carries mask
    // data for this run - sliced without --slicing-masks degrades to boxes).
    std::unique_ptr<AnnotationSink> sink;
    std::string labels_dir = export_labels, frames_dir;
    if (!export_labels.empty()) {
        std::error_code ec;
        if (kind == SourceKind::Video) {
            frames_dir = (fs::path(export_labels) / "frames").string();
            fs::create_directories(frames_dir, ec);
            if (lfmt != annot::Format::CocoJSON) {
                labels_dir = (fs::path(export_labels) / "labels").string();
                fs::create_directories(labels_dir, ec);
            }
        } else fs::create_directories(export_labels, ec);
    }
    auto ensure_sink = [&]() -> AnnotationSink& {
        if (!sink) {
            const bool dialect = be->is_seg();
            sink = std::make_unique<AnnotationSink>(
                lfmt, labels_dir, (fs::path(export_labels) / "annotations.coco.json").string(),
                cfg.class_names, dialect);
        }
        return *sink;
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    long frames = 0, total_dets = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0;

    // Video sources: annotated output becomes ONE mp4 (per-frame jpgs would overwrite each
    // other - "11.mp4#930" stems to "11"), and --save-txt gets frame-indexed names.
    const bool video_mode = (kind == SourceKind::Video);
    double src_fps = 30.0;
#ifdef HAVE_VIDEOIO
    cv::VideoWriter vwriter;                       // lazily opened on the first saved frame
    std::string vwriter_path;
#endif

    // coco_file/coco_id: the COCO doc's file_name (may carry "frames/") and explicit image
    // id (0 = sequence). export=false skips label emission (non-sampled video frames).
    auto run_one = [&](const cv::Mat& img, const std::string& tag,
                       bool do_export = true, const std::string& coco_file = "", int coco_id = 0) {
        if (img.empty()) { std::cerr << "  [skip] unreadable: " << tag << "\n"; return; }
        std::vector<Detection> dets;
        std::string slice_note;
        try {
            if (slice_mode != SliceMode::Off) {
                const SliceOutput so = sliced_candidates(*be, img, cfg, sconf);
                dets = nms_and_cap(be->candidates, cfg, img.cols, img.rows);
                tstats.add(so.tiles_run, so.tiles_total, so.tile_size_used,
                           so.used_fallback, so.capped);
                slice_note = "  tiles=" + std::to_string(so.tiles_run) + "/"
                           + std::to_string(so.tiles_total) + " @" + std::to_string(so.tile_size_used) + "px"
                           + (so.used_fallback ? " [fallback]" : "") + (so.capped ? " [capped]" : "");
            } else {
                dets = be->infer(img, cfg);
            }
        } catch (const std::exception& e) {
            std::cerr << "  [skip] inference error on " << tag << ": " << e.what() << "\n";
            return;
        }
        if (!export_labels.empty() && do_export) {
            AnnotationSink& s = ensure_sink();
            annot::Image aimg;
            aimg.name = fs::path(tag).stem().string();
            if (coco_id > 0) aimg.name = fs::path(coco_file).stem().string();
            aimg.width = img.cols; aimg.height = img.rows;
            aimg.instances = annotation_instances(dets, be->is_seg(), be->proto, be->proto_c,
                                                  be->proto_h, be->proto_w, be->cand_lb, cfg.imgsz);
            s.add(aimg, coco_file.empty() ? fs::path(tag).filename().string() : coco_file, coco_id);
        }
        frames++; total_dets += static_cast<long>(dets.size());
        sum_pre += be->pre_ms; sum_inf += be->infer_ms; sum_post += be->post_ms;
        if (!quiet)
            std::cout << "  " << tag << "  dets=" << dets.size()
                      << "  infer=" << be->infer_ms << "ms" << slice_note << "\n";
        if (!no_save) {
            cv::Mat vis = img.clone();
            if (be->is_seg()) {                       // alpha-composite segmentation masks under the boxes
                cv::Mat ov = seg_overlay(dets, be->proto, be->proto_c, be->proto_h, be->proto_w,
                                         be->cand_lb, cfg.imgsz, img.cols, img.rows);
                for (int y = 0; y < vis.rows; ++y) {
                    const uint8_t* o = ov.ptr<uint8_t>(y);
                    uint8_t* v = vis.ptr<uint8_t>(y);
                    for (int x = 0; x < vis.cols; ++x) {
                        const float a = o[x * 4 + 3] / 255.f;
                        if (a <= 0) continue;
                        v[x * 3 + 0] = (uint8_t)(v[x * 3 + 0] * (1 - a) + o[x * 4 + 2] * a);  // B<-B
                        v[x * 3 + 1] = (uint8_t)(v[x * 3 + 1] * (1 - a) + o[x * 4 + 1] * a);  // G<-G
                        v[x * 3 + 2] = (uint8_t)(v[x * 3 + 2] * (1 - a) + o[x * 4 + 0] * a);  // R<-R
                    }
                }
            }
            draw(vis, dets, cfg);
#ifdef HAVE_VIDEOIO
            if (video_mode) {                         // one annotated mp4, not overwriting jpgs
                if (!vwriter.isOpened()) {
                    vwriter_path = (fs::path(outdir) /
                        (fs::path(source).stem().string() + "_annotated.mp4")).string();
                    vwriter.open(vwriter_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                 src_fps, vis.size());
                    if (!vwriter.isOpened())
                        std::cerr << "  [warn] cannot open " << vwriter_path << " for writing\n";
                }
                if (vwriter.isOpened()) vwriter.write(vis);
            } else
#endif
            imwrite_jpg((fs::path(outdir) / (fs::path(tag).stem().string() + ".jpg")).string(), vis);
        }
        if (!savetxt.empty()) {                       // 'class conf x1 y1 x2 y2' (pixel xyxy)
            std::string tstem = fs::path(tag).stem().string();
            if (video_mode && coco_id > 0) {          // frame-unique name (stem collides at "11")
                char b[64];
                std::snprintf(b, sizeof(b), "%s_%06d", fs::path(source).stem().string().c_str(),
                              coco_id - 1);
                tstem = b;
            }
            std::ofstream f((fs::path(savetxt) / (tstem + ".txt")).string());
            for (const auto& d : dets)
                f << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
                  << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height) << '\n';
        }
    };

    if (kind == SourceKind::Video) {
#ifdef HAVE_VIDEOIO
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) { std::cerr << "cannot open video: " << source << "\n"; return 4; }
        const double fps_probe = cap.get(cv::CAP_PROP_FPS);
        src_fps = (fps_probe > 1.0 && fps_probe < 1000.0) ? fps_probe : 30.0;
        // label-export sampling stride: all=1, 1s=round(fps), N=every Nth
        int stride = 1;
        if (!export_labels.empty()) {
            if (sampling == "1s") {
                const double fps = cap.get(cv::CAP_PROP_FPS);
                stride = std::max(1, static_cast<int>(std::lround(fps > 0 ? fps : 30)));
            } else if (sampling != "all") {
                try { stride = std::max(1, std::stoi(sampling)); }
                catch (...) { std::cerr << "unknown --sampling: " << sampling << "\n"; return 2; }
            }
        }
        const std::string vstem = fs::path(source).stem().string();
        cv::Mat frame; long idx = 0;
        while (cap.read(frame)) {
            if (limit > 0 && idx >= limit) break;
            const bool sampled = !export_labels.empty() && idx % stride == 0;
            std::string coco_file;
            if (sampled) {
                char fn[64];
                std::snprintf(fn, sizeof(fn), "%s_%06ld.jpg", vstem.c_str(), idx);
                write_jpg((fs::path(frames_dir) / fn).string(), frame);
                coco_file = std::string("frames/") + fn;
            }
            run_one(frame, source + "#" + std::to_string(idx), sampled, coco_file,
                    static_cast<int>(idx) + 1);
            ++idx;
        }
#else
        std::cerr << "video source not supported in this portable build; use image/dir/dataset\n";
        return 4;
#endif
    } else {
        auto imgs = gather_images(source, limit);
        if (imgs.empty()) { std::cerr << "no inputs resolved from source: " << source << "\n"; return 4; }
        for (const auto& p : imgs) run_one(imread_bgr(p), p);
    }

    if (frames == 0) { std::cerr << "no frames processed\n"; return 5; }
    const double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    const double avg = (sum_pre + sum_inf + sum_post) / frames;
    std::cout << "\n[summary] frames=" << frames << "  total_dets=" << total_dets
              << "  avg/frame: pre=" << sum_pre / frames << " infer=" << sum_inf / frames
              << " post=" << sum_post / frames << " total=" << avg << "ms"
              << "  model-FPS=" << 1000.0 / avg << "  wall=" << wall << "s";
    if (cfg.nms_mode == NmsMode::ClusterWeighted)
        std::cout << "  nms=cw(sigma=" << cfg.cw_sigma << ")";
    std::cout << "\n";
    if (slice_mode != SliceMode::Off)
        std::cout << "[slicing] mode=" << slicing << "  tiles=" << tstats.tiles_run << "/"
                  << tstats.tiles_total << "  size=" << tstats.tile_size_label()
                  << "  fallbacks=" << tstats.fallbacks << "  capped=" << tstats.capped << "\n";
    if (sink) {
        const AnnotationSink::Result r = sink->finish();
        if (!r.error.empty()) std::cerr << "[labels] export failed: " << r.error << "\n";
        else std::cout << "[labels] " << annot::label(lfmt) << "  images=" << r.images
                       << "  instances=" << r.instances << " -> " << export_labels << "/\n";
    }
    if (!no_save) {
#ifdef HAVE_VIDEOIO
        if (video_mode && vwriter.isOpened()) {
            vwriter.release();
            std::cout << "[saved] annotated video -> " << vwriter_path << "\n";
        } else
#endif
        std::cout << "[saved] annotated -> " << outdir << "/\n";
    }
    return 0;
}
