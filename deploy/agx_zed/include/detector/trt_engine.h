#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detector/postprocess.h"

namespace agx_zed {

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    bool load(const std::string& engine_path, std::string* error_message = nullptr);
    bool is_loaded() const;
    const TensorShape& output_shape() const;
    const std::vector<float>& infer(const cv::Mat& image_bgr, std::string* error_message = nullptr);

private:
    struct Impl;

    bool prepare_input(const cv::Mat& image_bgr, std::vector<float>& host_input, std::string* error_message) const;
    void release();

    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
    TensorShape output_shape_{1, 0, 0};
    std::vector<float> host_output_{};
};

}  // namespace agx_zed
