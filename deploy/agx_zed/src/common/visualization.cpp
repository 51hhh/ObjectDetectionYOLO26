#include "common/visualization.h"

#include <cmath>
#include <cstdio>

namespace agx_zed {

namespace {

void put_line(cv::Mat& frame, const std::string& text, int line_index, const cv::Scalar& color) {
    const cv::Point origin(20, 30 + line_index * 28);
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
}

std::string format_detection_text(const PipelineResult& result) {
    char buffer[128] = {};
    std::snprintf(buffer,
                  sizeof(buffer),
                  "score=%.2f cls=%d",
                  result.detection.score,
                  result.detection.class_id);
    return std::string(buffer);
}

cv::Point clamp_point(const cv::Mat& frame, int x, int y) {
    const int clamped_x = std::max(0, std::min(x, frame.cols - 1));
    const int clamped_y = std::max(0, std::min(y, frame.rows - 1));
    return cv::Point(clamped_x, clamped_y);
}

std::string format_relative_text(const PipelineResult& result) {
    char buffer[160] = {};
    std::snprintf(buffer,
                  sizeof(buffer),
                  "x=%.2fm y=%.2fm d=%.2fm z=%.2fm",
                  result.relative.x_m,
                  result.relative.y_m,
                  result.relative.distance_m,
                  result.relative.depth_z_m);
    return std::string(buffer);
}

std::string format_depth_text(const PipelineResult& result) {
    char buffer[160] = {};
    std::snprintf(buffer,
                  sizeof(buffer),
                  "depth=%.2fm valid_count=%d",
                  result.depth_sample.depth_m,
                  result.depth_sample.valid_count);
    return std::string(buffer);
}

}  // namespace

std::string make_status_text(const PipelineResult& result) {
    if (result.has_runtime_error) {
        return "Runtime warning";
    }
    if (result.camera_frame_transient_error) {
        return "Camera frame unavailable";
    }
    if (!result.has_detection) {
        return "No detection";
    }
    if (!result.depth_sample.valid || !result.relative.valid) {
        return "Detection but invalid depth";
    }
    return "Tracking ball";
}

void draw_overlay(cv::Mat& frame, const PipelineResult& result) {
    if (frame.empty()) {
        return;
    }

    const std::string status = make_status_text(result);
    cv::Scalar status_color;
    if (result.camera_frame_transient_error || result.has_runtime_error) {
        status_color = cv::Scalar(0, 0, 255);   // red: error
    } else if (result.relative.valid) {
        status_color = cv::Scalar(0, 255, 0);   // green: tracking
    } else if (result.has_detection) {
        status_color = cv::Scalar(0, 255, 255); // yellow: detection but no depth
    } else {
        status_color = cv::Scalar(0, 165, 255); // orange: no detection
    }

    put_line(frame, status, 0, status_color);

    int next_line_index = 1;
    if (result.has_runtime_error && !result.runtime_error_message.empty()) {
        put_line(frame, result.runtime_error_message.substr(0, 80), next_line_index++, cv::Scalar(0, 0, 255));
    }
    put_line(frame, "Press q or ESC to exit", next_line_index++, cv::Scalar(255, 255, 255));

    if (!result.has_detection) {
        return;
    }

    const auto& bbox = result.detection.bbox;
    const cv::Point top_left = clamp_point(frame,
                                           static_cast<int>(std::round(bbox.x1)),
                                           static_cast<int>(std::round(bbox.y1)));
    const cv::Point bottom_right = clamp_point(frame,
                                               static_cast<int>(std::round(bbox.x2)),
                                               static_cast<int>(std::round(bbox.y2)));
    const cv::Point center = clamp_point(frame,
                                         static_cast<int>(std::round(bbox.center_x())),
                                         static_cast<int>(std::round(bbox.center_y())));
    const int detection_line_index = next_line_index;

    cv::rectangle(frame, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
    cv::circle(frame, center, 4, cv::Scalar(0, 0, 255), -1);
    put_line(frame, format_detection_text(result), detection_line_index, cv::Scalar(255, 255, 255));

    if (result.depth_sample.valid && result.relative.valid) {
        put_line(frame, format_relative_text(result), detection_line_index + 1, cv::Scalar(0, 255, 0));
        put_line(frame, format_depth_text(result), detection_line_index + 2, cv::Scalar(0, 255, 0));
        return;
    }

    put_line(frame, "depth invalid", detection_line_index + 1, cv::Scalar(0, 255, 255));
    if (result.depth_sample.valid_count > 0) {
        put_line(frame, format_depth_text(result), detection_line_index + 2, cv::Scalar(0, 255, 255));
    }
}

}  // namespace agx_zed
