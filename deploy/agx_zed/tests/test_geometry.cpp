#include <cmath>
#include <iostream>

#include "common/types.h"
#include "geometry/depth_sampler.h"
#include "geometry/projection.h"
#include "geometry/world_transform.h"

namespace {

bool almost_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    using namespace agx_zed;

    DepthMap depth_map;
    depth_map.width = 5;
    depth_map.height = 5;
    depth_map.values = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 2.1f, 2.2f, 0.0f,
        0.0f, 2.0f, 2.0f, 2.2f, 0.0f,
        0.0f, 1.9f, 2.0f, 2.3f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto depth_sample = sample_depth_window(depth_map, 2.0f, 2.0f, 1, 0.1f, 10.0f);
    if (int rc = require(depth_sample.valid, "expected median depth sample to be valid")) return rc;
    if (int rc = require(almost_equal(depth_sample.depth_m, 2.0f), "expected median depth to equal 2.0m")) return rc;

    CameraIntrinsics intrinsics{400.0f, 400.0f, 320.0f, 240.0f};
    auto camera_point = project_pixel_to_camera(420.0f, 140.0f, 2.0f, intrinsics);
    if (int rc = require(almost_equal(camera_point.x_m, 0.5f), "expected projected x to be +0.5m")) return rc;
    if (int rc = require(almost_equal(camera_point.y_m, 0.5f), "expected projected y to be +0.5m")) return rc;
    if (int rc = require(almost_equal(camera_point.z_m, 2.0f), "expected projected z to be 2.0m")) return rc;

    auto relative = make_relative_observation(camera_point);
    if (int rc = require(almost_equal(relative.x_m, 0.5f), "expected relative x to track camera x")) return rc;
    if (int rc = require(almost_equal(relative.y_m, 0.5f), "expected relative y to track camera y")) return rc;
    if (int rc = require(almost_equal(relative.distance_m, std::sqrt(4.5f)), "expected Euclidean distance")) return rc;

    RigidTransform pose = identity_transform();
    pose.translation = {1.0f, 2.0f, 3.0f};
    auto world_point = transform_camera_to_world(camera_point, pose);
    if (int rc = require(almost_equal(world_point.x_m, 1.5f), "expected translated world x")) return rc;
    if (int rc = require(almost_equal(world_point.y_m, 2.5f), "expected translated world y")) return rc;
    if (int rc = require(almost_equal(world_point.z_m, 5.0f), "expected translated world z")) return rc;

    return 0;
}
