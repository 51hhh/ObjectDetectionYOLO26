#pragma once

#include "common/types.h"

namespace agx_zed {

DepthSample sample_depth_window(const DepthMap& depth_map,
                                float u_px,
                                float v_px,
                                int radius_px,
                                float min_depth_m,
                                float max_depth_m);

}  // namespace agx_zed
