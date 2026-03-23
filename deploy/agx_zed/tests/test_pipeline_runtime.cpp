#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "camera/camera_model.h"
#include "camera/zed_camera.h"
#include "common/config_loader.h"
#include "detector/detector.h"
#include "pipeline/ball_pipeline.h"

namespace {

int require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return 1;
    }
    return 0;
}

struct FakeCamera : agx_zed::ICamera {
    bool open(const agx_zed::AppConfig&, std::string*) override {
        opened = true;
        return true;
    }

    agx_zed::CameraGrabStatus grab(agx_zed::CameraFrame& frame, std::string* error_message) override {
        if (error_message) {
            *error_message = grab_error;
        }
        if (grab_status != agx_zed::CameraGrabStatus::Ok) {
            return grab_status;
        }

        frame.left_bgr = cv::Mat::zeros(480, 640, CV_8UC3);
        frame.depth_m = cv::Mat(480, 640, CV_32F, cv::Scalar(2.0f)).clone();
        frame.intrinsics = {400.0f, 400.0f, 320.0f, 240.0f};
        return agx_zed::CameraGrabStatus::Ok;
    }

    void close() override {
        opened = false;
    }

    bool is_open() const override {
        return opened;
    }

    bool opened = false;
    agx_zed::CameraGrabStatus grab_status = agx_zed::CameraGrabStatus::Ok;
    std::string grab_error;
};

struct FakeDetector : agx_zed::IDetector {
    bool initialize(const agx_zed::AppConfig&, std::string*) override {
        return true;
    }

    bool detect(const cv::Mat&, std::vector<agx_zed::Detection2D>& detections_out, std::string* error_message) override {
        ++detect_calls;
        detections_out = detections;
        if (error_message) {
            *error_message = detect_error;
        }
        return detect_ok;
    }

    bool detect_ok = true;
    int detect_calls = 0;
    std::string detect_error;
    std::vector<agx_zed::Detection2D> detections;
};

}  // namespace

int main() {
    using namespace agx_zed;

    {
        FakeCamera camera;
        FakeDetector detector;
        detector.detect_ok = false;
        detector.detect_error = "TensorRT enqueueV2 failed";

        BallPipeline pipeline(&camera, &detector);
        std::string error;
        if (int rc = require(pipeline.initialize(AppConfig{}, RigidTransform{}, &error),
                             "expected pipeline init with fake components to succeed")) return rc;

        PipelineResult result;
        const bool ok = pipeline.process_once(result, &error);
        if (int rc = require(ok, "expected process_once to continue when detector reports runtime error")) return rc;
        if (int rc = require(result.has_runtime_error,
                             "expected detector runtime error flag to be set")) return rc;
        if (int rc = require(result.runtime_error_message == "TensorRT enqueueV2 failed",
                             "expected detector runtime error message to be preserved")) return rc;
        if (int rc = require(error == "TensorRT enqueueV2 failed",
                             "expected detector runtime error to propagate")) return rc;
        if (int rc = require(!result.frame.left_bgr.empty(),
                             "expected frame to remain available on detector runtime error")) return rc;
    }

    {
        FakeCamera camera;
        FakeDetector detector;

        BallPipeline pipeline(&camera, &detector);
        std::string error = "stale";
        if (int rc = require(pipeline.initialize(AppConfig{}, RigidTransform{}, &error),
                             "expected pipeline init with fake components to succeed")) return rc;

        PipelineResult result;
        const bool ok = pipeline.process_once(result, &error);
        if (int rc = require(ok, "expected process_once to succeed when detector returns no detections")) return rc;
        if (int rc = require(!result.has_detection, "expected no detections in result")) return rc;
        if (int rc = require(error.empty(), "expected no stale detector error after empty detection result")) return rc;
        if (int rc = require(!result.frame.left_bgr.empty(), "expected frame to be preserved for visualization")) return rc;
    }

    {
        FakeCamera camera;
        FakeDetector detector;
        camera.grab_status = CameraGrabStatus::TransientFailure;
        camera.grab_error = "ZED grab failed";

        BallPipeline pipeline(&camera, &detector);
        std::string error = "stale";
        if (int rc = require(pipeline.initialize(AppConfig{}, RigidTransform{}, &error),
                             "expected pipeline init with fake components to succeed")) return rc;

        PipelineResult result;
        const bool ok = pipeline.process_once(result, &error);
        if (int rc = require(ok, "expected transient camera grab failure to be tolerated")) return rc;
        if (int rc = require(error == "ZED grab failed",
                             "expected transient camera error to be reported")) return rc;
        if (int rc = require(result.camera_frame_transient_error,
                             "expected transient camera error flag to be set")) return rc;
        if (int rc = require(result.frame.left_bgr.empty(),
                             "expected no frame on transient camera error")) return rc;
        if (int rc = require(detector.detect_calls == 0,
                             "expected detector not to run when camera frame is unavailable")) return rc;
    }

    {
        FakeCamera camera;
        FakeDetector detector;
        camera.grab_status = CameraGrabStatus::FatalFailure;
        camera.grab_error = "ZED camera is not open";

        BallPipeline pipeline(&camera, &detector);
        std::string error;
        if (int rc = require(pipeline.initialize(AppConfig{}, RigidTransform{}, &error),
                             "expected pipeline init with fake components to succeed")) return rc;

        PipelineResult result;
        const bool ok = pipeline.process_once(result, &error);
        if (int rc = require(ok, "expected fatal camera grab failure to be reported without stopping pipeline")) return rc;
        if (int rc = require(result.has_runtime_error,
                             "expected fatal camera error flag to be set")) return rc;
        if (int rc = require(result.runtime_error_message == "ZED camera is not open",
                             "expected fatal camera error message to be preserved")) return rc;
        if (int rc = require(!result.camera_frame_transient_error,
                             "expected fatal camera error not to be mislabeled as transient")) return rc;
        if (int rc = require(error == "ZED camera is not open",
                             "expected fatal camera error to propagate")) return rc;
    }

    return 0;
}
