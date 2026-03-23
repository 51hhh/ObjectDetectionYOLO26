#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp>

#include "common/config_loader.h"
#include "common/visualization.h"
#include "pipeline/ball_pipeline.h"

namespace {

std::string format_console_line(const agx_zed::PipelineResult& result) {
    const std::string status = agx_zed::make_status_text(result);
    if (!result.relative.valid) {
        return status;
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2)
           << status
           << " x=" << result.relative.x_m
           << " y=" << result.relative.y_m
           << " d=" << result.relative.distance_m
           << " z=" << result.relative.depth_z_m;
    if (result.has_world_point) {
        stream << " world=("
               << result.world_point.x_m << ", "
               << result.world_point.y_m << ", "
               << result.world_point.z_m << ")";
    }
    return stream.str();
}

bool should_log_result(const agx_zed::PipelineResult& result,
                       const std::string& status,
                       const std::string& last_status,
                       std::uint64_t frame_index) {
    return status != last_status || (result.relative.valid && frame_index % 30 == 0);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config_path = argc > 1 ? argv[1] : "../../configs/deploy/agx_zed_yolo26.yaml";
    const std::string pose_path = argc > 2 ? argv[2] : "../../configs/deploy/camera_pose.yaml";
    constexpr const char* kWindowName = "agx_zed_runtime";

    try {
        const auto config = agx_zed::load_app_config(config_path);
        const auto world_from_camera = agx_zed::load_camera_pose(pose_path);

        agx_zed::BallPipeline pipeline;
        std::string error;
        if (!pipeline.initialize(config, world_from_camera, &error)) {
            std::cerr << "Failed to initialize pipeline: " << error << std::endl;
            return 1;
        }

        if (config.show) {
            try {
                cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
            } catch (const cv::Exception& ex) {
                std::cerr << "Failed to create OpenCV window: " << ex.what() << std::endl;
                std::cerr << "Use show=false or start from an AGX desktop session." << std::endl;
                return 1;
            }
        }

        std::string last_status;
        std::string last_camera_warning;
        std::string last_runtime_error;
        std::uint64_t frame_index = 0;
        int consecutive_camera_errors = 0;
        constexpr int kMaxTransientBackoffMs = 1000;
        constexpr int kTransientBackoffStepMs = 50;
        while (true) {
            agx_zed::PipelineResult result;
            const bool ok = pipeline.process_once(result, &error);
            if (!ok) {
                std::cerr << "Pipeline processing failed: " << error << std::endl;
                if (config.show) {
                    cv::destroyWindow(kWindowName);
                }
                return 1;
            }

            if (result.has_runtime_error) {
                const std::string runtime_error = result.runtime_error_message.empty()
                    ? error
                    : result.runtime_error_message;
                if (runtime_error != last_runtime_error) {
                    std::cerr << "Runtime warning: " << runtime_error << std::endl;
                    last_runtime_error = runtime_error;
                }
            } else {
                last_runtime_error.clear();
            }

            if (result.camera_frame_transient_error) {
                ++consecutive_camera_errors;
                const std::string camera_warning = result.runtime_error_message.empty()
                    ? error
                    : result.runtime_error_message;
                if (camera_warning != last_camera_warning) {
                    std::cerr << "Camera frame unavailable (" << consecutive_camera_errors
                              << "): " << camera_warning << std::endl;
                    last_camera_warning = camera_warning;
                }
                const int backoff_ms = std::min(
                    consecutive_camera_errors * kTransientBackoffStepMs,
                    kMaxTransientBackoffMs);
                if (config.show) {
                    const int key = cv::waitKey(backoff_ms > 0 ? backoff_ms : 1);
                    if (key == 27 || key == 'q' || key == 'Q') {
                        break;
                    }
                } else if (backoff_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
                }
                continue;
            }
            consecutive_camera_errors = 0;
            last_camera_warning.clear();

            const std::string status = agx_zed::make_status_text(result);
            if (should_log_result(result, status, last_status, frame_index)) {
                std::cout << format_console_line(result) << std::endl;
                last_status = status;
            }

            if (config.show) {
                if (result.frame.left_bgr.empty()) {
                    std::cerr << "Runtime warning: Pipeline returned empty LEFT frame" << std::endl;
                    const int key = cv::waitKey(1);
                    if (key == 27 || key == 'q' || key == 'Q') {
                        break;
                    }
                    continue;
                }

                try {
                    cv::Mat overlay_frame = result.frame.left_bgr.clone();
                    agx_zed::draw_overlay(overlay_frame, result);
                    cv::imshow(kWindowName, overlay_frame);
                } catch (const cv::Exception& ex) {
                    std::cerr << "Runtime warning: OpenCV display failed: " << ex.what() << std::endl;
                }

                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }

            ++frame_index;
        }

        if (config.show) {
            cv::destroyWindow(kWindowName);
        }
    } catch (const cv::Exception& ex) {
        std::cerr << "OpenCV exception: " << ex.what() << std::endl;
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
