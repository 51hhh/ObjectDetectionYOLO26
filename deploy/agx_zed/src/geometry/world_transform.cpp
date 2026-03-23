#include "geometry/world_transform.h"

namespace agx_zed {

RigidTransform identity_transform() {
    return RigidTransform{};
}

CameraPoint transform_camera_to_world(const CameraPoint& point_cam, const RigidTransform& world_from_camera) {
    CameraPoint point_world;
    if (!point_cam.valid || !world_from_camera.valid) {
        return point_world;
    }

    const auto& r = world_from_camera.rotation;
    point_world.x_m = r[0] * point_cam.x_m + r[1] * point_cam.y_m + r[2] * point_cam.z_m + world_from_camera.translation.x_m;
    point_world.y_m = r[3] * point_cam.x_m + r[4] * point_cam.y_m + r[5] * point_cam.z_m + world_from_camera.translation.y_m;
    point_world.z_m = r[6] * point_cam.x_m + r[7] * point_cam.y_m + r[8] * point_cam.z_m + world_from_camera.translation.z_m;
    point_world.valid = true;
    return point_world;
}

}  // namespace agx_zed
