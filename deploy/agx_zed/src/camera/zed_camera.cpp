#include "camera/zed_camera.h"

#include <memory>
#include <string>

#include <opencv2/imgproc.hpp>
#include <sl/Camera.hpp>

namespace {

bool is_fatal_grab_error(sl::ERROR_CODE err) {
    switch (err) {
        case sl::ERROR_CODE::CAMERA_NOT_DETECTED:
        case sl::ERROR_CODE::CAMERA_NOT_INITIALIZED:
        case sl::ERROR_CODE::CAMERA_FAILED_TO_SETUP:
        case sl::ERROR_CODE::CAMERA_DETECTION_ISSUE:
        case sl::ERROR_CODE::CANNOT_START_CAMERA_STREAM:
        case sl::ERROR_CODE::DRIVER_FAILURE:
        case sl::ERROR_CODE::CORRUPTED_SDK_INSTALLATION:
        case sl::ERROR_CODE::NO_GPU_DETECTED:
        case sl::ERROR_CODE::CUDA_ERROR:
        case sl::ERROR_CODE::NVIDIA_DRIVER_OUT_OF_DATE:
        case sl::ERROR_CODE::INVALID_FUNCTION_CALL:
            return true;
        default:
            return false;
    }
}

}  // namespace

namespace agx_zed {

struct ZedCamera::Impl {
    sl::Camera camera;
};

namespace {

sl::RESOLUTION resolve_resolution(const std::string& value) {
    if (value == "HD1080") return sl::RESOLUTION::HD1080;
    if (value == "HD1200") return sl::RESOLUTION::HD1200;
    if (value == "SVGA") return sl::RESOLUTION::SVGA;
    if (value == "VGA") return sl::RESOLUTION::VGA;
    return sl::RESOLUTION::HD720;
}

}  // namespace

ZedCamera::ZedCamera() = default;

ZedCamera::~ZedCamera() {
    close();
}

bool ZedCamera::open(const AppConfig& config, std::string* error_message) {
    close();
    impl_ = new Impl();

    sl::InitParameters params;
    params.camera_resolution = resolve_resolution(config.resolution);
    params.camera_fps = config.fps;
    params.coordinate_units = sl::UNIT::METER;
    params.depth_mode = sl::DEPTH_MODE::NEURAL;
    params.depth_minimum_distance = config.min_depth_m;
    params.depth_maximum_distance = config.max_depth_m;

    const auto err = impl_->camera.open(params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        if (error_message) {
            *error_message = sl::toString(err).c_str();
        }
        close();
        return false;
    }
    return true;
}

CameraGrabStatus ZedCamera::grab(CameraFrame& frame, std::string* error_message) {
    if (!impl_) {
        if (error_message) {
            *error_message = "ZED camera is not open";
        }
        return CameraGrabStatus::FatalFailure;
    }

    const sl::ERROR_CODE grab_error = impl_->camera.grab();
    if (grab_error != sl::ERROR_CODE::SUCCESS) {
        if (error_message) {
            *error_message = std::string("ZED grab failed: ") + sl::toString(grab_error).c_str();
        }
        return is_fatal_grab_error(grab_error)
            ? CameraGrabStatus::FatalFailure
            : CameraGrabStatus::TransientFailure;
    }

    sl::Mat left;
    sl::Mat depth;
    const sl::ERROR_CODE retrieve_err = impl_->camera.retrieveImage(left, sl::VIEW::LEFT, sl::MEM::CPU);
    if (retrieve_err != sl::ERROR_CODE::SUCCESS) {
        if (error_message) {
            *error_message = std::string("ZED retrieveImage failed: ") + sl::toString(retrieve_err).c_str();
        }
        return CameraGrabStatus::TransientFailure;
    }

    const sl::ERROR_CODE depth_err = impl_->camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
    if (depth_err != sl::ERROR_CODE::SUCCESS) {
        if (error_message) {
            *error_message = std::string("ZED retrieveMeasure failed: ") + sl::toString(depth_err).c_str();
        }
        return CameraGrabStatus::TransientFailure;
    }

    const int width = left.getWidth();
    const int height = left.getHeight();
    if (width <= 0 || height <= 0) {
        if (error_message) {
            *error_message = "ZED returned empty LEFT frame";
        }
        return CameraGrabStatus::TransientFailure;
    }

    if (left.getChannels() == 4) {
        cv::Mat rgba(height, width, CV_8UC4, left.getPtr<sl::uchar1>(sl::MEM::CPU), left.getStepBytes());
        cv::cvtColor(rgba, frame.left_bgr, cv::COLOR_BGRA2BGR);
    } else {
        cv::Mat bgr(height, width, CV_8UC3, left.getPtr<sl::uchar1>(sl::MEM::CPU), left.getStepBytes());
        frame.left_bgr = bgr.clone();
    }

    cv::Mat depth_mat(depth.getHeight(), depth.getWidth(), CV_32F, depth.getPtr<float>(sl::MEM::CPU), depth.getStepBytes());
    frame.depth_m = depth_mat.clone();

    const auto calib = impl_->camera.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
    frame.intrinsics.fx = calib.fx;
    frame.intrinsics.fy = calib.fy;
    frame.intrinsics.cx = calib.cx;
    frame.intrinsics.cy = calib.cy;
    frame.timestamp_ns = static_cast<std::uint64_t>(impl_->camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds());

    if (!frame.valid()) {
        if (error_message) {
            *error_message = "ZED returned invalid frame payload";
        }
        return CameraGrabStatus::TransientFailure;
    }

    return CameraGrabStatus::Ok;
}

void ZedCamera::close() {
    if (impl_) {
        if (impl_->camera.isOpened()) {
            impl_->camera.close();
        }
        delete impl_;
        impl_ = nullptr;
    }
}

bool ZedCamera::is_open() const {
    return impl_ != nullptr;
}

}  // namespace agx_zed
