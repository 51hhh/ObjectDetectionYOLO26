#include "common/config_loader.h"

#include <filesystem>

#include <yaml-cpp/yaml.h>

namespace agx_zed {

namespace {

namespace fs = std::filesystem;

std::string get_string(const YAML::Node& node, const char* key, const std::string& default_value) {
    return node[key] ? node[key].as<std::string>() : default_value;
}

int get_int(const YAML::Node& node, const char* key, int default_value) {
    return node[key] ? node[key].as<int>() : default_value;
}

float get_float(const YAML::Node& node, const char* key, float default_value) {
    return node[key] ? node[key].as<float>() : default_value;
}

bool get_bool(const YAML::Node& node, const char* key, bool default_value) {
    return node[key] ? node[key].as<bool>() : default_value;
}

bool looks_like_project_root(const fs::path& path) {
    return fs::exists(path / "configs") && fs::exists(path / "deploy");
}

fs::path find_project_root_from(fs::path start) {
    start = fs::absolute(start).lexically_normal();
    if (!fs::is_directory(start)) {
        start = start.parent_path();
    }

    fs::path current = start;
    while (!current.empty()) {
        if (looks_like_project_root(current)) {
            return current;
        }
        if (current == current.root_path()) {
            break;
        }
        current = current.parent_path();
    }
    return {};
}

fs::path infer_project_root(const fs::path& config_path) {
    const fs::path from_config = find_project_root_from(config_path);
    if (!from_config.empty()) {
        return from_config;
    }

    const fs::path from_cwd = find_project_root_from(fs::current_path());
    if (!from_cwd.empty()) {
        return from_cwd;
    }

    return fs::absolute(config_path).lexically_normal().parent_path();
}

std::string resolve_project_path(const fs::path& project_root, const std::string& raw_path) {
    const fs::path path(raw_path);
    if (path.is_absolute()) {
        return path.lexically_normal().string();
    }
    return fs::absolute(project_root / path).lexically_normal().string();
}

}  // namespace

AppConfig load_app_config(const std::string& file_path) {
    const YAML::Node root = YAML::LoadFile(file_path);
    const fs::path project_root = infer_project_root(file_path);

    AppConfig config;
    config.weights_path = resolve_project_path(project_root, get_string(root, "weights", "weight/best.pt"));
    config.onnx_path = resolve_project_path(project_root, get_string(root, "onnx", "deploy/agx_zed/models/best.onnx"));
    config.engine_path = resolve_project_path(project_root, get_string(root, "engine", "deploy/agx_zed/models/best.engine"));
    config.imgsz = get_int(root, "imgsz", 640);
    config.conf = get_float(root, "conf", 0.25f);
    config.iou = get_float(root, "iou", 0.45f);
    config.class_id = get_int(root, "class_id", 0);
    config.keep_topk = get_int(root, "keep_topk", 1);
    config.show = get_bool(root, "show", true);

    const YAML::Node camera = root["camera"];
    if (camera) {
        config.depth_window_radius = get_int(camera, "depth_window_radius", 2);
        config.min_depth_m = get_float(camera, "min_depth_m", 0.1f);
        config.max_depth_m = get_float(camera, "max_depth_m", 20.0f);
        config.resolution = get_string(camera, "resolution", "HD1200");
        config.fps = get_int(camera, "fps", 60);
    }

    return config;
}

RigidTransform load_camera_pose(const std::string& file_path) {
    const YAML::Node root = YAML::LoadFile(file_path);
    RigidTransform tf;

    const YAML::Node node = root["world_from_camera"];
    if (!node) {
        return tf;
    }

    const YAML::Node rotation = node["rotation"];
    if (rotation && rotation.IsSequence() && rotation.size() == 3) {
        int idx = 0;
        for (std::size_t row = 0; row < 3; ++row) {
            const YAML::Node row_node = rotation[row];
            for (std::size_t col = 0; col < 3; ++col) {
                tf.rotation[idx++] = row_node[col].as<float>();
            }
        }
    }

    const YAML::Node translation = node["translation_m"];
    if (translation && translation.IsSequence() && translation.size() == 3) {
        tf.translation.x_m = translation[0].as<float>();
        tf.translation.y_m = translation[1].as<float>();
        tf.translation.z_m = translation[2].as<float>();
    }

    return tf;
}

}  // namespace agx_zed
