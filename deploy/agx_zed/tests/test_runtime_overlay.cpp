#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "common/visualization.h"

namespace {

int require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return 1;
    }
    return 0;
}

agx_zed::PipelineResult make_result(bool has_detection, bool depth_valid, bool relative_valid) {
    agx_zed::PipelineResult result;
    result.frame.left_bgr = cv::Mat::zeros(240, 320, CV_8UC3);

    if (has_detection) {
        result.has_detection = true;
        result.detection.bbox = {100.0f, 120.0f, 160.0f, 180.0f};
        result.detection.score = 0.91f;
        result.detection.class_id = 0;
    }

    result.depth_sample.valid = depth_valid;
    result.depth_sample.depth_m = 2.4f;
    result.depth_sample.valid_count = depth_valid ? 9 : 0;

    result.relative.valid = relative_valid;
    result.relative.x_m = 0.25f;
    result.relative.y_m = -0.10f;
    result.relative.distance_m = 2.42f;
    result.relative.depth_z_m = 2.4f;
    return result;
}

bool has_drawn_pixels(const cv::Mat& image) {
    return cv::countNonZero(image.reshape(1)) > 0;
}

}  // namespace

int main() {
    using namespace agx_zed;

    auto camera_unavailable = make_result(false, false, false);
    camera_unavailable.camera_frame_transient_error = true;
    if (int rc = require(make_status_text(camera_unavailable) == "Camera frame unavailable",
                         "expected camera-unavailable status text")) return rc;

    auto runtime_warning = make_result(false, false, false);
    runtime_warning.has_runtime_error = true;
    runtime_warning.runtime_error_message = "TensorRT enqueueV2 failed";
    if (int rc = require(make_status_text(runtime_warning) == "Runtime warning",
                         "expected runtime-warning status text")) return rc;

    const auto no_detection = make_result(false, false, false);
    if (int rc = require(make_status_text(no_detection) == "No detection",
                         "expected no-detection status text")) return rc;

    const auto invalid_depth = make_result(true, false, false);
    if (int rc = require(make_status_text(invalid_depth) == "Detection but invalid depth",
                         "expected invalid-depth status text")) return rc;

    const auto tracking = make_result(true, true, true);
    if (int rc = require(make_status_text(tracking) == "Tracking ball",
                         "expected tracking status text")) return rc;

    cv::Mat overlay_frame = tracking.frame.left_bgr.clone();
    draw_overlay(overlay_frame, tracking);
    if (int rc = require(has_drawn_pixels(overlay_frame),
                         "expected overlay to draw on frame")) return rc;

    cv::Mat empty_frame;
    draw_overlay(empty_frame, tracking);
    if (int rc = require(empty_frame.empty(),
                         "expected empty frame to remain empty")) return rc;

    return 0;
}
