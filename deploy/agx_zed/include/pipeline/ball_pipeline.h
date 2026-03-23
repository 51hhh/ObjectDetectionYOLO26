#pragma once

#include <string>

#include "camera/zed_camera.h"
#include "common/config_loader.h"
#include "common/types.h"
#include "detector/detector.h"

namespace agx_zed {

struct PipelineResult {
    CameraFrame frame;
    Detection2D detection;
    DepthSample depth_sample;
    RelativeObservation relative;
    CameraPoint world_point;
    bool has_detection = false;
    bool has_world_point = false;
    bool camera_frame_transient_error = false;
    bool has_runtime_error = false;
    std::string runtime_error_message;
};

class BallPipeline {
public:
    BallPipeline(ICamera* camera = nullptr, IDetector* detector = nullptr);

    bool initialize(const AppConfig& config,
                    const RigidTransform& world_from_camera,
                    std::string* error_message = nullptr);
    bool process_once(PipelineResult& result, std::string* error_message = nullptr);

private:
    AppConfig config_{};
    RigidTransform world_from_camera_{};
    ZedCamera owned_camera_{};
    Detector owned_detector_{};
    ICamera* camera_ = nullptr;
    IDetector* detector_ = nullptr;
};

}  // namespace agx_zed
