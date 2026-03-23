#pragma once

#include <vector>

#include "common/types.h"

namespace agx_zed {

struct LetterboxMeta {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int network_width = 640;
    int network_height = 640;
    int original_width = 640;
    int original_height = 640;
};

struct TensorShape {
    int batch = 1;
    int dim1 = 0;
    int dim2 = 0;
};

std::vector<Detection2D> decode_yolo_detections(const std::vector<float>& output,
                                                const TensorShape& shape,
                                                const LetterboxMeta& meta,
                                                float conf_threshold,
                                                float iou_threshold,
                                                int class_id,
                                                int keep_topk);

}  // namespace agx_zed
