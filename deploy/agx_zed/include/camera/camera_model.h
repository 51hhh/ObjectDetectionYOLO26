#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

#include "common/types.h"

namespace agx_zed {

struct CameraFrame {
    cv::Mat left_bgr;
    cv::Mat depth_m;
    CameraIntrinsics intrinsics;
    std::uint64_t timestamp_ns = 0;

    bool valid() const {
        return !left_bgr.empty() && !depth_m.empty() && intrinsics.valid();
    }
};

}  // namespace agx_zed
