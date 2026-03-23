# AGX + ZED Deploy Runtime

该目录是当前仓库的 AGX / ZED 部署工程，目标是：
- 使用 ZED 双目深度，但只取 `LEFT` 图像做球检测
- 用 `DEPTH + 相机内参` 解算球相对相机坐标 `(x, y, d)`
- 预留 `P_cam -> P_world` 的世界坐标变换接口
- 在 AGX 上持续运行，并支持 OpenCV 可视化窗口

## 当前状态

当前已经打通：
- `best.pt -> best.onnx -> best.engine` 导出与构建链路
- TensorRT engine 反序列化与单输入单输出推理
- YOLO26 end2end 输出 `(1,300,6)` 的后处理
- ZED X 实机打开，当前验证通过模式为 `HD1200@60`
- `agx_zed_runtime` 持续循环、状态打印与窗口叠加显示
- C++ 测试：几何、后处理、运行时 overlay

## 依赖前提

在 AGX 上需要已有：
- JetPack / CUDA / TensorRT
- ZED SDK
- OpenCV 4
- `yaml-cpp`
- 可用的 `best.engine`

模型默认路径：
- ONNX：`deploy/agx_zed/models/best.onnx`
- Engine：`deploy/agx_zed/models/best.engine`

## 最短启动步骤

以下命令默认在仓库根目录执行。

### 1. 导出 ONNX

```bash
python ./scripts/export_onnx.py --weights ./weight/best.pt --imgsz 640
```

### 2. 在 AGX 上构建 TensorRT engine

```bash
trtexec --onnx=./deploy/agx_zed/models/best.onnx --saveEngine=./deploy/agx_zed/models/best.engine --fp16
```

### 3. 构建 C++ 运行时

```bash
cmake -S ./deploy/agx_zed -B ./deploy/agx_zed/build
cmake --build ./deploy/agx_zed/build
```

### 4. 运行测试

```bash
cd ./deploy/agx_zed/build && ctest --output-on-failure
```

### 5. 启动运行时

GUI 模式由配置文件中的 `show` 控制。

#### GUI 模式

把 `configs/deploy/agx_zed_yolo26.yaml` 中的 `show` 设为 `true`，然后在 AGX 本地桌面终端启动：

```bash
./deploy/agx_zed/build/agx_zed_runtime ./configs/deploy/agx_zed_yolo26.yaml ./configs/deploy/camera_pose.yaml
```

运行后会持续显示 `LEFT` 画面、检测框、状态文本和 `(x, y, d, z)` 叠加信息，按 `q` 或 `ESC` 退出。

#### Headless 模式

把 `show` 设为 `false`，然后启动：

```bash
./deploy/agx_zed/build/agx_zed_runtime ./configs/deploy/agx_zed_yolo26.yaml ./configs/deploy/camera_pose.yaml
```

此时不会打开窗口，只会持续运行并在控制台打印状态。

## 配置说明

主配置文件：`configs/deploy/agx_zed_yolo26.yaml`

当前关键字段：
- `engine`：TensorRT engine 路径
- `imgsz`：网络输入尺寸，当前默认 `640`
- `conf`：置信度阈值
- `iou`：NMS IoU 阈值
- `show`：是否开启 OpenCV GUI
- `camera.resolution`：当前默认 `HD1200`
- `camera.fps`：当前默认 `60`
- `camera.depth_window_radius`：深度中值采样窗口半径

相机位姿文件：`configs/deploy/camera_pose.yaml`

## 常见问题

### `engine file not found`
- 检查 `configs/deploy/agx_zed_yolo26.yaml` 中的 `engine` 路径
- 确认 `best.engine` 已经放到 AGX 上对应位置

### `deserializeCudaEngine failed`
- 说明 engine 与当前 AGX / TensorRT 环境不匹配
- 在目标 AGX 上重新执行 `trtexec` 构建 engine

### `ZED grab failed`
- 检查 ZED 相机连接、电源与 SDK
- 确认当前分辨率/帧率组合有效，当前已验证 `HD1200@60`

### `show=true` 但没有窗口
- 这是显示环境问题，不是检测逻辑问题
- 直接 SSH 进入 AGX 通常不会弹出 OpenCV 窗口
- 请在 AGX 本地桌面终端运行，或确认当前会话具备可用的 `DISPLAY`

### 只有 `No detection` 或 `Detection but invalid depth`
- 先确认球是否进入画面且检测框稳定
- 再检查深度窗口附近是否有有效深度值
- 必要时降低遮挡、增加目标尺寸或检查光照

## 目录说明

- `include/camera/`：ZED 图像、深度、内参接口
- `include/detector/`：TensorRT 与 YOLO 后处理接口
- `include/geometry/`：深度采样、投影、世界坐标变换
- `include/pipeline/`：主流程串联
- `src/common/visualization.cpp`：运行时状态文本与窗口 overlay
- `tests/`：C++ 小型测试

更多背景与完整步骤见：`docs/02_部署推理文档.md`
