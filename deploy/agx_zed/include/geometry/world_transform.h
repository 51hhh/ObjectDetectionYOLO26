#pragma once

#include "common/types.h"

namespace agx_zed {

RigidTransform identity_transform();
CameraPoint transform_camera_to_world(const CameraPoint& point_cam, const RigidTransform& world_from_camera);

}  // namespace agx_zed
