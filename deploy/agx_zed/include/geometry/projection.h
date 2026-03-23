#pragma once

#include "common/types.h"

namespace agx_zed {

CameraPoint project_pixel_to_camera(float u_px,
                                    float v_px,
                                    float depth_z_m,
                                    const CameraIntrinsics& intrinsics);

RelativeObservation make_relative_observation(const CameraPoint& point_cam,
                                              float u_px = 0.0f,
                                              float v_px = 0.0f);

}  // namespace agx_zed
