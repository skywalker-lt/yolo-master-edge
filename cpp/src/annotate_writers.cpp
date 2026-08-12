// Pure annotation writers - see annotate.hpp. Parity reference:
// mac/Sources/YOLOMasterKit/AnnotationExport.swift (AnnotationWriter).
#include "annotate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace yolomaster { namespace annot {

static std::string f6(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.6f", v);
    return b;
}

// px coords/areas in the COCO doc ("%.3f": sub-pixel precision without double noise)
static std::string f3(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.3f", v);
    return b;
}

static std::string xml_esc(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '&':  o += "&amp;"; break;
        case '<':  o += "&lt;"; break;
        case '>':  o += "&gt;"; break;
        case '"':  o += "&quot;"; break;
        default:   o += c;
        }
    }
    return o;
}

static std::string json_esc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                o += b;
            } else o += static_cast<char>(c);
        }
    }
    return o;
}

std::string yolo_lines(const Image& img, bool seg_dialect) {
    const double W = std::max(1, img.width), H = std::max(1, img.height);
    auto nx = [&](double v) { return std::min(1.0, std::max(0.0, v / W)); };
    auto ny = [&](double v) { return std::min(1.0, std::max(0.0, v / H)); };
    std::string out;
    for (const Instance& inst : img.instances) {
        const std::vector<float>* poly =
            (seg_dialect && !inst.polygons.empty() && inst.polygons.front().size() >= 6)
                ? &inst.polygons.front() : nullptr;
        std::string line = std::to_string(inst.cls);
        if (poly) {
            for (size_t i = 0; i + 1 < poly->size(); i += 2) {
                line += " " + f6(nx((*poly)[i]));
                line += " " + f6(ny((*poly)[i + 1]));
            }
        } else {
            line += " " + f6(nx(inst.x + inst.w / 2)) + " " + f6(ny(inst.y + inst.h / 2))
                  + " " + f6(nx(inst.w)) + " " + f6(ny(inst.h));
        }
        out += line + "\n";
    }
    return out;
}

std::string classes_txt(const std::vector<std::string>& names) {
    std::string out;
    for (const std::string& n : names) { out += n; out += "\n"; }
    return out;
}

std::string voc_xml(const Image& img, const std::vector<std::string>& names,
                    const std::string& folder) {
    std::string x = "<annotation>\n"
        "\t<folder>" + xml_esc(folder) + "</folder>\n"
        "\t<filename>" + xml_esc(img.name) + ".jpg</filename>\n"
        "\t<size>\n"
        "\t\t<width>" + std::to_string(img.width) + "</width>\n"
        "\t\t<height>" + std::to_string(img.height) + "</height>\n"
        "\t\t<depth>3</depth>\n"
        "\t</size>\n"
        "\t<segmented>0</segmented>\n";
    for (const Instance& inst : img.instances) {
        const std::string name = inst.cls >= 0 && inst.cls < static_cast<int>(names.size())
            ? names[inst.cls] : "class" + std::to_string(inst.cls);
        const int xmin = std::max(1, std::min(img.width,  static_cast<int>(std::lround(inst.x)) + 1));
        const int ymin = std::max(1, std::min(img.height, static_cast<int>(std::lround(inst.y)) + 1));
        const int xmax = std::max(xmin, std::min(img.width,  static_cast<int>(std::lround(inst.x + inst.w))));
        const int ymax = std::max(ymin, std::min(img.height, static_cast<int>(std::lround(inst.y + inst.h))));
        x += "\t<object>\n"
            "\t\t<name>" + xml_esc(name) + "</name>\n"
            "\t\t<pose>Unspecified</pose>\n"
            "\t\t<truncated>0</truncated>\n"
            "\t\t<difficult>0</difficult>\n"
            "\t\t<bndbox>\n"
            "\t\t\t<xmin>" + std::to_string(xmin) + "</xmin>\n"
            "\t\t\t<ymin>" + std::to_string(ymin) + "</ymin>\n"
            "\t\t\t<xmax>" + std::to_string(xmax) + "</xmax>\n"
            "\t\t\t<ymax>" + std::to_string(ymax) + "</ymax>\n"
            "\t\t</bndbox>\n"
            "\t</object>\n";
    }
    x += "</annotation>\n";
    return x;
}

static double shoelace(const std::vector<float>& flat) {
    const size_t n = flat.size() / 2;
    if (n < 3) return 0;
    double a = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = (i + 1) % n;
        a += static_cast<double>(flat[2 * i]) * flat[2 * j + 1]
           - static_cast<double>(flat[2 * j]) * flat[2 * i + 1];
    }
    return std::abs(a) / 2;
}

std::string coco_json(const std::vector<Image>& images,
                      const std::vector<std::string>& file_names,
                      const std::vector<std::string>& names, bool seg_dialect,
                      const std::vector<int>* image_ids) {
    // Keys inside each object are emitted alphabetically (Mac JSONEncoder .sortedKeys parity).
    std::string imgs, anns;
    int ann_id = 1;
    for (size_t i = 0; i < images.size(); ++i) {
        const Image& img = images[i];
        const int iid = image_ids ? (*image_ids)[i] : static_cast<int>(i) + 1;
        if (!imgs.empty()) imgs += ",";
        imgs += "{\"file_name\":\"" + json_esc(file_names[i]) + "\","
                "\"height\":" + std::to_string(img.height) + ","
                "\"id\":" + std::to_string(iid) + ","
                "\"width\":" + std::to_string(img.width) + "}";
        for (const Instance& inst : img.instances) {
            const double bbox_area = static_cast<double>(inst.w) * inst.h;
            double area = bbox_area;
            std::string segs;
            if (seg_dialect) {
                area = 0;
                for (const auto& p : inst.polygons) {
                    area += shoelace(p);
                    if (p.size() < 6) continue;
                    if (!segs.empty()) segs += ",";
                    segs += "[";
                    for (size_t k = 0; k < p.size(); ++k) {
                        if (k) segs += ",";
                        segs += f3(p[k]);
                    }
                    segs += "]";
                }
                if (area <= 0) area = bbox_area;
            }
            if (!anns.empty()) anns += ",";
            anns += "{\"area\":" + f3(area) + ","
                    "\"bbox\":[" + f3(inst.x) + "," + f3(inst.y) + ","
                                 + f3(inst.w) + "," + f3(inst.h) + "],"
                    "\"category_id\":" + std::to_string(inst.cls + 1) + ","
                    "\"id\":" + std::to_string(ann_id) + ","
                    "\"image_id\":" + std::to_string(iid) + ","
                    "\"iscrowd\":0,"
                    "\"score\":" + f6(inst.score);
            if (seg_dialect) anns += ",\"segmentation\":[" + segs + "]";
            anns += "}";
            ++ann_id;
        }
    }
    std::string cats;
    for (size_t c = 0; c < names.size(); ++c) {
        if (!cats.empty()) cats += ",";
        cats += "{\"id\":" + std::to_string(c + 1) + ",\"name\":\"" + json_esc(names[c]) + "\"}";
    }
    return "{\"annotations\":[" + anns + "],"
           "\"categories\":[" + cats + "],"
           "\"images\":[" + imgs + "],"
           "\"info\":{\"description\":\"YOLO-Master edge runner export\"}}";
}

}} // namespace yolomaster::annot
