#include "geometry/depth_sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace agx_zed {

DepthSample sample_depth_window(const DepthMap& depth_map,
                                float u_px,
                                float v_px,
                                int radius_px,
                                float min_depth_m,
                                float max_depth_m) {
    DepthSample sample;
    sample.radius_px = radius_px;

    if (depth_map.empty() || radius_px < 0) {
        return sample;
    }

    const int u = static_cast<int>(std::lround(u_px));
    const int v = static_cast<int>(std::lround(v_px));

    std::vector<float> valid_depths;
    for (int dy = -radius_px; dy <= radius_px; ++dy) {
        for (int dx = -radius_px; dx <= radius_px; ++dx) {
            const int x = u + dx;
            const int y = v + dy;
            if (!depth_map.contains(x, y)) {
                continue;
            }

            const float depth = depth_map.at(x, y);
            if (!std::isfinite(depth) || depth < min_depth_m || depth > max_depth_m) {
                continue;
            }
            valid_depths.push_back(depth);
        }
    }

    sample.valid_count = static_cast<int>(valid_depths.size());
    if (valid_depths.empty()) {
        return sample;
    }

    const auto middle = valid_depths.begin() + (valid_depths.size() / 2);
    std::nth_element(valid_depths.begin(), middle, valid_depths.end());
    sample.depth_m = *middle;
    sample.valid = true;
    return sample;
}

}  // namespace agx_zed
