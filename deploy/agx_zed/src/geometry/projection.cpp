#include "geometry/projection.h"

#include <cmath>

namespace agx_zed {

CameraPoint project_pixel_to_camera(float u_px,
                                    float v_px,
                                    float depth_z_m,
                                    const CameraIntrinsics& intrinsics) {
    CameraPoint point;
    if (!intrinsics.valid() || !std::isfinite(depth_z_m) || depth_z_m <= 0.0f) {
        return point;
    }

    point.x_m = ((u_px - intrinsics.cx) / intrinsics.fx) * depth_z_m;
    point.y_m = -((v_px - intrinsics.cy) / intrinsics.fy) * depth_z_m;
    point.z_m = depth_z_m;
    point.valid = true;
    return point;
}

RelativeObservation make_relative_observation(const CameraPoint& point_cam,
                                              float u_px,
                                              float v_px) {
    RelativeObservation observation;
    observation.u_px = u_px;
    observation.v_px = v_px;
    observation.point_cam = point_cam;
    observation.valid = point_cam.valid;
    if (!point_cam.valid) {
        return observation;
    }

    observation.x_m = point_cam.x_m;
    observation.y_m = point_cam.y_m;
    observation.depth_z_m = point_cam.z_m;
    observation.distance_m = euclidean_norm(point_cam);
    return observation;
}

}  // namespace agx_zed
