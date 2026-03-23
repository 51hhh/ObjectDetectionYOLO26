#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "pipeline/ball_pipeline.h"

namespace agx_zed {

std::string make_status_text(const PipelineResult& result);
void draw_overlay(cv::Mat& frame, const PipelineResult& result);

}  // namespace agx_zed
