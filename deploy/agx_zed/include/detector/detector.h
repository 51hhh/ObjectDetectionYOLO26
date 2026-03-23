#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "common/config_loader.h"
#include "common/types.h"
#include "detector/postprocess.h"
#include "detector/trt_engine.h"

namespace agx_zed {

class IDetector {
public:
    virtual ~IDetector() = default;

    virtual bool initialize(const AppConfig& config, std::string* error_message = nullptr) = 0;
    virtual bool detect(const cv::Mat& image,
                        std::vector<Detection2D>& detections_out,
                        std::string* error_message = nullptr) = 0;
};

class Detector : public IDetector {
public:
    Detector();

    bool initialize(const AppConfig& config, std::string* error_message = nullptr) override;
    bool detect(const cv::Mat& image,
                std::vector<Detection2D>& detections_out,
                std::string* error_message = nullptr) override;

private:
    AppConfig config_{};
    TrtEngine engine_;
};

}  // namespace agx_zed
