#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace agx_zed {

struct BoundingBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    bool valid() const { return width() > 0.0f && height() > 0.0f; }
    float center_x() const { return (x1 + x2) * 0.5f; }
    float center_y() const { return (y1 + y2) * 0.5f; }
};

struct Detection2D {
    BoundingBox bbox;
    float score = 0.0f;
    int class_id = -1;
};

struct CameraIntrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    bool valid() const { return fx > 0.0f && fy > 0.0f; }
};

struct DepthMap {
    int width = 0;
    int height = 0;
    std::vector<float> values;

    bool empty() const { return width <= 0 || height <= 0 || values.empty(); }
    bool contains(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < height; }

    float at(int x, int y) const {
        if (!contains(x, y)) {
            throw std::out_of_range("DepthMap index out of range");
        }
        return values[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
    }
};

struct CameraPoint {
    float x_m = 0.0f;
    float y_m = 0.0f;
    float z_m = 0.0f;
    bool valid = false;
};

struct RelativeObservation {
    float x_m = 0.0f;
    float y_m = 0.0f;
    float distance_m = 0.0f;
    float depth_z_m = 0.0f;
    float u_px = 0.0f;
    float v_px = 0.0f;
    bool valid = false;
    CameraPoint point_cam;
};

struct DepthSample {
    float depth_m = 0.0f;
    int valid_count = 0;
    int radius_px = 0;
    bool valid = false;
};

struct Vec3f {
    float x_m = 0.0f;
    float y_m = 0.0f;
    float z_m = 0.0f;
};

struct RigidTransform {
    std::array<float, 9> rotation = {1.0f, 0.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f};
    Vec3f translation{};
    bool valid = true;
};

inline float euclidean_norm(const CameraPoint& point) {
    return std::sqrt(point.x_m * point.x_m + point.y_m * point.y_m + point.z_m * point.z_m);
}

}  // namespace agx_zed
