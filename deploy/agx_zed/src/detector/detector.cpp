#include "detector/detector.h"

#include <algorithm>

namespace agx_zed {

namespace {

LetterboxMeta make_letterbox_meta(const cv::Mat& image, int imgsz) {
    const float scale = std::min(static_cast<float>(imgsz) / static_cast<float>(image.cols),
                                 static_cast<float>(imgsz) / static_cast<float>(image.rows));
    const int resized_w = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image.rows * scale)));

    LetterboxMeta meta;
    meta.scale = scale;
    meta.pad_x = static_cast<float>((imgsz - resized_w) / 2);
    meta.pad_y = static_cast<float>((imgsz - resized_h) / 2);
    meta.network_width = imgsz;
    meta.network_height = imgsz;
    meta.original_width = image.cols;
    meta.original_height = image.rows;
    return meta;
}

}  // namespace

Detector::Detector() = default;

bool Detector::initialize(const AppConfig& config, std::string* error_message) {
    config_ = config;
    return engine_.load(config.engine_path, error_message);
}

bool Detector::detect(const cv::Mat& image,
                      std::vector<Detection2D>& detections_out,
                      std::string* error_message) {
    detections_out.clear();
    if (error_message) {
        error_message->clear();
    }

    if (image.empty()) {
        if (error_message) {
            *error_message = "input image is empty";
        }
        return false;
    }
    if (!engine_.is_loaded()) {
        if (error_message) {
            *error_message = "TensorRT engine is not loaded";
        }
        return false;
    }

    const LetterboxMeta meta = make_letterbox_meta(image, config_.imgsz);
    const auto& output = engine_.infer(image, error_message);
    if (output.empty()) {
        if (error_message && !error_message->empty()) {
            return false;
        }
        return true;  // empty output but no error: legitimate "no detections"
    }

    detections_out = decode_yolo_detections(output,
                                            engine_.output_shape(),
                                            meta,
                                            config_.conf,
                                            config_.iou,
                                            config_.class_id,
                                            config_.keep_topk);
    return true;
}

}  // namespace agx_zed
