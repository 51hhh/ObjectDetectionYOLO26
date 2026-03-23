#pragma once

#include <string>

#include "common/types.h"

namespace agx_zed {

struct AppConfig {
    std::string weights_path;
    std::string onnx_path;
    std::string engine_path;
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.45f;
    int class_id = 0;
    int keep_topk = 1;
    bool show = true;
    int depth_window_radius = 2;
    float min_depth_m = 0.1f;
    float max_depth_m = 20.0f;
    std::string resolution = "HD1200";
    int fps = 60;
};

AppConfig load_app_config(const std::string& file_path);
RigidTransform load_camera_pose(const std::string& file_path);

}  // namespace agx_zed
