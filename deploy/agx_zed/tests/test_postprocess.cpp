#include <iostream>
#include <vector>

#include "common/types.h"
#include "detector/postprocess.h"

namespace {

int require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    using namespace agx_zed;

    LetterboxMeta meta;
    meta.scale = 1.0f;
    meta.pad_x = 0.0f;
    meta.pad_y = 0.0f;
    meta.network_width = 640;
    meta.network_height = 640;
    meta.original_width = 640;
    meta.original_height = 640;

    TensorShape shape_c_first{1, 5, 2};
    std::vector<float> output_c_first = {
        320.0f, 100.0f,
        240.0f, 120.0f,
        40.0f,  30.0f,
        20.0f,  20.0f,
        0.90f,  0.10f,
    };
    auto dets_c_first = decode_yolo_detections(output_c_first, shape_c_first, meta, 0.25f, 0.45f, 0, 1);
    if (int rc = require(dets_c_first.size() == 1, "expected one detection from channel-first tensor")) return rc;
    if (int rc = require(dets_c_first.front().class_id == 0, "expected class 0")) return rc;
    if (int rc = require(dets_c_first.front().score > 0.89f, "expected high score")) return rc;
    if (int rc = require(dets_c_first.front().bbox.x1 == 300.0f, "expected x1 == 300")) return rc;
    if (int rc = require(dets_c_first.front().bbox.y1 == 230.0f, "expected y1 == 230")) return rc;

    TensorShape shape_n_first{1, 2, 5};
    std::vector<float> output_n_first = {
        320.0f, 240.0f, 40.0f, 20.0f, 0.95f,
        400.0f, 240.0f, 40.0f, 20.0f, 0.10f,
    };
    auto dets_n_first = decode_yolo_detections(output_n_first, shape_n_first, meta, 0.25f, 0.45f, 0, 1);
    if (int rc = require(dets_n_first.size() == 1, "expected one detection from candidate-first tensor")) return rc;
    if (int rc = require(dets_n_first.front().bbox.x2 == 340.0f, "expected x2 == 340")) return rc;
    if (int rc = require(dets_n_first.front().bbox.y2 == 250.0f, "expected y2 == 250")) return rc;

    TensorShape shape_end2end{1, 2, 6};
    std::vector<float> output_end2end = {
        300.0f, 230.0f, 340.0f, 250.0f, 0.95f, 7.0f,
        100.0f, 120.0f, 140.0f, 160.0f, 0.10f, 3.0f,
    };
    auto dets_end2end = decode_yolo_detections(output_end2end, shape_end2end, meta, 0.25f, 0.45f, 0, 1);
    if (int rc = require(dets_end2end.size() == 1, "expected one detection from end-to-end tensor")) return rc;
    if (int rc = require(dets_end2end.front().class_id == 7, "expected class id from tensor")) return rc;
    if (int rc = require(dets_end2end.front().bbox.x1 == 300.0f, "expected end-to-end x1 == 300")) return rc;
    if (int rc = require(dets_end2end.front().bbox.y2 == 250.0f, "expected end-to-end y2 == 250")) return rc;

    return 0;
}
