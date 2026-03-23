#pragma once

#include <string>

#include "camera/camera_model.h"
#include "common/config_loader.h"

namespace agx_zed {

enum class CameraGrabStatus {
    Ok,
    TransientFailure,
    FatalFailure,
};

class ICamera {
public:
    virtual ~ICamera() = default;

    virtual bool open(const AppConfig& config, std::string* error_message = nullptr) = 0;
    virtual CameraGrabStatus grab(CameraFrame& frame, std::string* error_message = nullptr) = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
};

class ZedCamera : public ICamera {
public:
    ZedCamera();
    ~ZedCamera() override;

    bool open(const AppConfig& config, std::string* error_message = nullptr) override;
    CameraGrabStatus grab(CameraFrame& frame, std::string* error_message = nullptr) override;
    void close() override;
    bool is_open() const override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace agx_zed
