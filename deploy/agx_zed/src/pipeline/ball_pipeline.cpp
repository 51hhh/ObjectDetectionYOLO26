#include "pipeline/ball_pipeline.h"

#include "geometry/depth_sampler.h"
#include "geometry/projection.h"
#include "geometry/world_transform.h"

namespace agx_zed {

BallPipeline::BallPipeline(ICamera* camera, IDetector* detector)
    : camera_(camera ? camera : &owned_camera_),
      detector_(detector ? detector : &owned_detector_) {}

bool BallPipeline::initialize(const AppConfig& config,
                              const RigidTransform& world_from_camera,
                              std::string* error_message) {
    config_ = config;
    world_from_camera_ = world_from_camera;

    if (!camera_->open(config, error_message)) {
        return false;
    }
    if (!detector_->initialize(config, error_message)) {
        return false;
    }
    return true;
}

bool BallPipeline::process_once(PipelineResult& result, std::string* error_message) {
    CameraFrame frame;
    result = PipelineResult{};

    const CameraGrabStatus grab_status = camera_->grab(frame, error_message);
    if (grab_status == CameraGrabStatus::FatalFailure) {
        result.camera_frame_transient_error = true;
        result.has_runtime_error = true;
        if (error_message) {
            result.runtime_error_message = *error_message;
        }
        return true;
    }
    if (grab_status == CameraGrabStatus::TransientFailure) {
        result.camera_frame_transient_error = true;
        if (error_message) {
            result.runtime_error_message = *error_message;
        }
        return true;
    }

    result.frame = frame;
    if (error_message) {
        error_message->clear();
    }

    std::vector<Detection2D> detections;
    if (!detector_->detect(frame.left_bgr, detections, error_message)) {
        result.has_runtime_error = true;
        if (error_message) {
            result.runtime_error_message = *error_message;
        }
        return true;
    }
    if (detections.empty()) {
        return true;
    }

    result.detection = detections.front();
    result.has_detection = true;

    DepthMap depth_map;
    depth_map.width = frame.depth_m.cols;
    depth_map.height = frame.depth_m.rows;
    depth_map.values.assign(reinterpret_cast<const float*>(frame.depth_m.datastart),
                            reinterpret_cast<const float*>(frame.depth_m.dataend));

    const float u = result.detection.bbox.center_x();
    const float v = result.detection.bbox.center_y();
    result.depth_sample = sample_depth_window(depth_map,
                                              u,
                                              v,
                                              config_.depth_window_radius,
                                              config_.min_depth_m,
                                              config_.max_depth_m);
    if (!result.depth_sample.valid) {
        return true;
    }

    const auto point_cam = project_pixel_to_camera(u, v, result.depth_sample.depth_m, frame.intrinsics);
    result.relative = make_relative_observation(point_cam, u, v);
    result.world_point = transform_camera_to_world(point_cam, world_from_camera_);
    result.has_world_point = result.world_point.valid;
    return true;
}

}  // namespace agx_zed
