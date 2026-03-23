#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "common/config_loader.h"

namespace {

namespace fs = std::filesystem;

int require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        return 1;
    }
    return 0;
}

fs::path find_project_root(fs::path start) {
    start = fs::absolute(start).lexically_normal();
    while (!start.empty()) {
        if (fs::exists(start / "configs") && fs::exists(start / "deploy")) {
            return start;
        }
        if (start == start.root_path()) {
            break;
        }
        start = start.parent_path();
    }
    return {};
}

}  // namespace

int main() {
    using namespace agx_zed;

    const fs::path temp_root = fs::current_path() / "config_loader_test_project";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root / "configs" / "deploy");
    fs::create_directories(temp_root / "deploy" / "agx_zed" / "models");

    const fs::path config_path = temp_root / "configs" / "deploy" / "agx_zed_yolo26.yaml";
    {
        std::ofstream config_file(config_path);
        config_file << "weights: weight/best.pt\n";
        config_file << "onnx: deploy/agx_zed/models/best.onnx\n";
        config_file << "engine: deploy/agx_zed/models/best.engine\n";
        config_file << "show: false\n";
        config_file << "camera:\n";
        config_file << "  resolution: HD1200\n";
        config_file << "  fps: 60\n";
    }

    const auto config = load_app_config(config_path.string());
    const fs::path expected_weights = fs::absolute(temp_root / "weight" / "best.pt").lexically_normal();
    const fs::path expected_onnx = fs::absolute(temp_root / "deploy" / "agx_zed" / "models" / "best.onnx").lexically_normal();
    const fs::path expected_engine = fs::absolute(temp_root / "deploy" / "agx_zed" / "models" / "best.engine").lexically_normal();

    if (int rc = require(fs::path(config.weights_path) == expected_weights,
                         "expected weights path to resolve from project root")) return rc;
    if (int rc = require(fs::path(config.onnx_path) == expected_onnx,
                         "expected onnx path to resolve from project root")) return rc;
    if (int rc = require(fs::path(config.engine_path) == expected_engine,
                         "expected engine path to resolve from project root")) return rc;
    if (int rc = require(config.show == false, "expected show=false to be parsed")) return rc;

    const fs::path repo_root = find_project_root(fs::current_path());
    if (int rc = require(!repo_root.empty(), "expected to find repo root from current working directory")) return rc;

    const fs::path external_dir = fs::current_path() / "external_config_test";
    fs::remove_all(external_dir);
    fs::create_directories(external_dir);

    const fs::path external_config_path = external_dir / "agx_zed_yolo26.yaml";
    {
        std::ofstream external_config_file(external_config_path);
        external_config_file << "weights: weight/best.pt\n";
        external_config_file << "onnx: deploy/agx_zed/models/best.onnx\n";
        external_config_file << "engine: deploy/agx_zed/models/best.engine\n";
    }

    const auto external_config = load_app_config(external_config_path.string());
    const fs::path expected_external_engine = fs::absolute(repo_root / "deploy" / "agx_zed" / "models" / "best.engine").lexically_normal();
    if (int rc = require(fs::path(external_config.engine_path) == expected_external_engine,
                         "expected external config to fall back to repo root from cwd")) return rc;

    fs::remove_all(external_dir);
    fs::remove_all(temp_root);
    return 0;
}
