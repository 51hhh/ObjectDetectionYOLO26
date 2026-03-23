#include "detector/trt_engine.h"

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace agx_zed {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

TrtLogger g_logger;

bool is_float_tensor(nvinfer1::DataType dtype) {
    return dtype == nvinfer1::DataType::kFLOAT;
}

std::size_t volume_of(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        return 0;
    }
    std::size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        volume *= static_cast<std::size_t>(dims.d[i]);
    }
    return volume;
}

}  // namespace

struct TrtEngine::Impl {
    struct RuntimeDeleter {
        void operator()(nvinfer1::IRuntime* ptr) const {
            if (ptr) ptr->destroy();
        }
    };

    struct EngineDeleter {
        void operator()(nvinfer1::ICudaEngine* ptr) const {
            if (ptr) ptr->destroy();
        }
    };

    struct ContextDeleter {
        void operator()(nvinfer1::IExecutionContext* ptr) const {
            if (ptr) ptr->destroy();
        }
    };

    struct DeviceBuffer {
        void* ptr = nullptr;
        std::size_t bytes = 0;

        ~DeviceBuffer() {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context;
    cudaStream_t stream = nullptr;
    int input_binding_index = -1;
    int output_binding_index = -1;
    std::size_t input_volume = 0;
    std::size_t output_volume = 0;
    DeviceBuffer input_buffer;
    DeviceBuffer output_buffer;

    ~Impl() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

TrtEngine::TrtEngine() = default;

TrtEngine::~TrtEngine() {
    release();
}

bool TrtEngine::prepare_input(const cv::Mat& image_bgr, std::vector<float>& host_input, std::string* error_message) const {
    if (image_bgr.empty()) {
        if (error_message) {
            *error_message = "input image is empty";
        }
        return false;
    }
    if (!impl_ || !impl_->engine || impl_->input_binding_index < 0) {
        if (error_message) {
            *error_message = "TensorRT engine is not initialized";
        }
        return false;
    }

    const auto input_dims = impl_->engine->getBindingDimensions(impl_->input_binding_index);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3) {
        if (error_message) {
            *error_message = "expected input binding dims [1,3,H,W]";
        }
        return false;
    }

    const int input_h = input_dims.d[2];
    const int input_w = input_dims.d[3];
    if (input_h <= 0 || input_w <= 0) {
        if (error_message) {
            *error_message = "invalid TensorRT input size";
        }
        return false;
    }

    const float scale = std::min(static_cast<float>(input_w) / static_cast<float>(image_bgr.cols),
                                 static_cast<float>(input_h) / static_cast<float>(image_bgr.rows));
    const int resized_w = std::max(1, static_cast<int>(std::round(image_bgr.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(image_bgr.rows * scale)));
    const int pad_x = (input_w - resized_w) / 2;
    const int pad_y = (input_h - resized_h) / 2;

    cv::Mat resized;
    cv::resize(image_bgr, resized, cv::Size(resized_w, resized_h));

    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_rgb;
    rgb.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);

    host_input.assign(impl_->input_volume, 0.0f);
    const int plane_size = input_h * input_w;
    for (int y = 0; y < input_h; ++y) {
        const auto* row = float_rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < input_w; ++x) {
            const auto& pixel = row[x];
            const int index = y * input_w + x;
            host_input[0 * plane_size + index] = pixel[0];
            host_input[1 * plane_size + index] = pixel[1];
            host_input[2 * plane_size + index] = pixel[2];
        }
    }

    return true;
}

void TrtEngine::release() {
    host_output_.clear();
    impl_.reset();
    output_shape_ = TensorShape{1, 0, 0};
    loaded_ = false;
}

bool TrtEngine::load(const std::string& engine_path, std::string* error_message) {
    release();

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        if (error_message) {
            *error_message = "engine file not found: " + engine_path;
        }
        return false;
    }

    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), static_cast<std::streamsize>(size));
    file.close();

    impl_ = std::make_unique<Impl>();
    impl_->runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!impl_->runtime) {
        if (error_message) {
            *error_message = "createInferRuntime failed";
        }
        release();
        return false;
    }

    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(engine_data.data(), size, nullptr));
    if (!impl_->engine) {
        if (error_message) {
            *error_message = "deserializeCudaEngine failed";
        }
        release();
        return false;
    }

    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        if (error_message) {
            *error_message = "createExecutionContext failed";
        }
        release();
        return false;
    }

    const int num_bindings = impl_->engine->getNbBindings();
    if (num_bindings != 2) {
        if (error_message) {
            *error_message = "expected single input and single output bindings";
        }
        release();
        return false;
    }

    for (int i = 0; i < num_bindings; ++i) {
        const auto dims = impl_->engine->getBindingDimensions(i);
        const auto dtype = impl_->engine->getBindingDataType(i);
        if (!is_float_tensor(dtype)) {
            if (error_message) {
                *error_message = "only float TensorRT bindings are supported";
            }
            release();
            return false;
        }

        if (impl_->engine->bindingIsInput(i)) {
            impl_->input_binding_index = i;
            impl_->input_volume = volume_of(dims);
            if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3) {
                if (error_message) {
                    *error_message = "expected input binding dims [1,3,H,W]";
                }
                release();
                return false;
            }
        } else {
            impl_->output_binding_index = i;
            impl_->output_volume = volume_of(dims);
            if (dims.nbDims != 3 || dims.d[0] != 1) {
                if (error_message) {
                    *error_message = "expected output binding dims [1,N,6] or [1,6,N]";
                }
                release();
                return false;
            }
            output_shape_.batch = dims.d[0];
            output_shape_.dim1 = dims.d[1];
            output_shape_.dim2 = dims.d[2];
        }
    }

    if (impl_->input_binding_index < 0 || impl_->output_binding_index < 0 || impl_->input_volume == 0 || impl_->output_volume == 0) {
        if (error_message) {
            *error_message = "failed to locate valid TensorRT bindings";
        }
        release();
        return false;
    }

    const auto input_bytes = impl_->input_volume * sizeof(float);
    const auto output_bytes = impl_->output_volume * sizeof(float);

    cudaError_t status = cudaStreamCreate(&impl_->stream);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaStreamCreate failed: ") + cudaGetErrorString(status);
        }
        release();
        return false;
    }

    status = cudaMalloc(&impl_->input_buffer.ptr, input_bytes);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaMalloc input failed: ") + cudaGetErrorString(status);
        }
        release();
        return false;
    }
    impl_->input_buffer.bytes = input_bytes;

    status = cudaMalloc(&impl_->output_buffer.ptr, output_bytes);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaMalloc output failed: ") + cudaGetErrorString(status);
        }
        release();
        return false;
    }
    impl_->output_buffer.bytes = output_bytes;

    host_output_.assign(impl_->output_volume, 0.0f);
    loaded_ = true;
    return true;
}

bool TrtEngine::is_loaded() const {
    return loaded_;
}

const TensorShape& TrtEngine::output_shape() const {
    return output_shape_;
}

const std::vector<float>& TrtEngine::infer(const cv::Mat& image_bgr, std::string* error_message) {
    host_output_.clear();
    if (!loaded_ || !impl_) {
        if (error_message) {
            *error_message = "TensorRT engine is not loaded";
        }
        return host_output_;
    }

    std::vector<float> host_input;
    if (!prepare_input(image_bgr, host_input, error_message)) {
        return host_output_;
    }

    cudaError_t status = cudaMemcpyAsync(impl_->input_buffer.ptr,
                                         host_input.data(),
                                         impl_->input_buffer.bytes,
                                         cudaMemcpyHostToDevice,
                                         impl_->stream);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaMemcpyAsync input failed: ") + cudaGetErrorString(status);
        }
        return host_output_;
    }

    void* bindings[2] = {nullptr, nullptr};
    bindings[impl_->input_binding_index] = impl_->input_buffer.ptr;
    bindings[impl_->output_binding_index] = impl_->output_buffer.ptr;

    if (!impl_->context->enqueueV2(bindings, impl_->stream, nullptr)) {
        if (error_message) {
            *error_message = "TensorRT enqueueV2 failed";
        }
        return host_output_;
    }

    host_output_.assign(impl_->output_volume, 0.0f);
    status = cudaMemcpyAsync(host_output_.data(),
                             impl_->output_buffer.ptr,
                             impl_->output_buffer.bytes,
                             cudaMemcpyDeviceToHost,
                             impl_->stream);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaMemcpyAsync output failed: ") + cudaGetErrorString(status);
        }
        host_output_.clear();
        return host_output_;
    }

    status = cudaStreamSynchronize(impl_->stream);
    if (status != cudaSuccess) {
        if (error_message) {
            *error_message = std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(status);
        }
        host_output_.clear();
        return host_output_;
    }

    return host_output_;
}

}  // namespace agx_zed
