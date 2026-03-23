#include "detector/postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace agx_zed {

namespace {

float compute_iou(const BoundingBox& a, const BoundingBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float union_area = a.width() * a.height() + b.width() * b.height() - inter;
    if (union_area <= 0.0f) return 0.0f;
    return inter / union_area;
}

BoundingBox remap_xyxy_box(float net_x1, float net_y1, float net_x2, float net_y2, const LetterboxMeta& meta) {
    BoundingBox box;
    box.x1 = std::clamp((net_x1 - meta.pad_x) / meta.scale, 0.0f, static_cast<float>(meta.original_width));
    box.y1 = std::clamp((net_y1 - meta.pad_y) / meta.scale, 0.0f, static_cast<float>(meta.original_height));
    box.x2 = std::clamp((net_x2 - meta.pad_x) / meta.scale, 0.0f, static_cast<float>(meta.original_width));
    box.y2 = std::clamp((net_y2 - meta.pad_y) / meta.scale, 0.0f, static_cast<float>(meta.original_height));
    return box;
}

BoundingBox remap_box(float cx, float cy, float w, float h, const LetterboxMeta& meta) {
    return remap_xyxy_box(cx - w * 0.5f,
                          cy - h * 0.5f,
                          cx + w * 0.5f,
                          cy + h * 0.5f,
                          meta);
}

std::vector<Detection2D> nms_keep_topk(std::vector<Detection2D> detections, float iou_threshold, int keep_topk) {
    std::sort(detections.begin(), detections.end(), [](const Detection2D& a, const Detection2D& b) {
        return a.score > b.score;
    });

    std::vector<Detection2D> kept;
    for (const auto& det : detections) {
        bool suppressed = false;
        for (const auto& prev : kept) {
            if (compute_iou(det.bbox, prev.bbox) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(det);
            if (keep_topk > 0 && static_cast<int>(kept.size()) >= keep_topk) {
                break;
            }
        }
    }
    return kept;
}

}  // namespace

std::vector<Detection2D> decode_yolo_detections(const std::vector<float>& output,
                                                const TensorShape& shape,
                                                const LetterboxMeta& meta,
                                                float conf_threshold,
                                                float iou_threshold,
                                                int class_id,
                                                int keep_topk) {
    const int dim1 = shape.dim1;
    const int dim2 = shape.dim2;
    if (dim1 <= 0 || dim2 <= 0) {
        return {};
    }

    const std::size_t expected = static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(dim1) * static_cast<std::size_t>(dim2);
    if (output.size() < expected) {
        throw std::runtime_error("output tensor size is smaller than TensorShape");
    }

    bool channels_first = false;
    int element_count = 0;
    int candidate_count = 0;
    if (dim1 == 5 || dim1 == 6) {
        channels_first = true;
        element_count = dim1;
        candidate_count = dim2;
    } else if (dim2 == 5 || dim2 == 6) {
        channels_first = false;
        candidate_count = dim1;
        element_count = dim2;
    } else if (dim1 < dim2) {
        channels_first = true;
        element_count = dim1;
        candidate_count = dim2;
    } else if (dim2 < dim1) {
        channels_first = false;
        candidate_count = dim1;
        element_count = dim2;
    } else {
        throw std::runtime_error("unable to infer YOLO output layout");
    }

    if (element_count < 5) {
        throw std::runtime_error("YOLO output must contain at least 5 elements per candidate");
    }

    std::vector<Detection2D> detections;
    detections.reserve(candidate_count);

    for (int i = 0; i < candidate_count; ++i) {
        const float v0 = channels_first ? output[0 * candidate_count + i] : output[i * element_count + 0];
        const float v1 = channels_first ? output[1 * candidate_count + i] : output[i * element_count + 1];
        const float v2 = channels_first ? output[2 * candidate_count + i] : output[i * element_count + 2];
        const float v3 = channels_first ? output[3 * candidate_count + i] : output[i * element_count + 3];
        const float score = channels_first ? output[4 * candidate_count + i] : output[i * element_count + 4];
        const float raw_class = element_count >= 6
            ? (channels_first ? output[5 * candidate_count + i] : output[i * element_count + 5])
            : static_cast<float>(class_id);

        if (!std::isfinite(score) || score < conf_threshold) {
            continue;
        }

        Detection2D det;
        det.class_id = element_count >= 6 ? static_cast<int>(raw_class) : class_id;
        det.score = score;
        det.bbox = element_count == 6
            ? remap_xyxy_box(v0, v1, v2, v3, meta)
            : remap_box(v0, v1, v2, v3, meta);
        if (det.bbox.valid()) {
            detections.push_back(det);
        }
    }

    return nms_keep_topk(std::move(detections), iou_threshold, keep_topk);
}

}  // namespace agx_zed
