# ObjectDetectionYOLO26

基于 Ultralytics YOLO26 的排球检测项目。

当前项目已完成：
- 数据集整理与训练集生成
- YOLO26 本地训练
- Python 推理脚本
- AGX + ZED 部署运行时骨架与实时可视化入口
- 仓库结构规范化

## 仓库目录结构

```text
ObjectDetectionYOLO26/
├─ configs/                     # 训练与部署配置文件
│  ├─ train_volleyball.yaml
│  └─ deploy/
│     ├─ agx_zed_yolo26.yaml
│     └─ camera_pose.yaml
├─ docs/                        # 项目文档
│  ├─ 01_训练环境搭建与训练步骤.md
│  ├─ 02_部署推理文档.md
│  └─ 03_仓库文件夹架构规范.md
├─ scripts/                     # 可执行脚本
│  ├─ prepare_dataset.py        # 数据集转换脚本
│  ├─ train_volleyball.py       # 训练脚本
│  ├─ infer_volleyball.py       # Python 推理脚本
│  └─ export_onnx.py            # ONNX 导出脚本
├─ deploy/                      # AGX/ZED 部署工程
│  └─ agx_zed/
├─ coco/                        # 训练数据集（YOLO/COCO 格式）
├─ runs/                        # 训练与推理输出目录
├─ images.zip                   # 正样本压缩包
├─ negative_samples.zip         # 负样本压缩包
├─ yolo26n.pt                   # YOLO26 预训练权重
├─ .gitignore
└─ README.md
```

## 快速开始

### 1. 创建并使用虚拟环境

```bash
python -m venv .venv --system-site-packages
"./.venv/Scripts/python.exe" -m pip install --upgrade pip
"./.venv/Scripts/python.exe" -m pip install ultralytics==8.4.24 pyyaml pillow
```

### 2. 准备数据集

```bash
"./.venv/Scripts/python.exe" "./scripts/prepare_dataset.py"
```

### 3. 开始训练

```bash
"./.venv/Scripts/python.exe" "./scripts/train_volleyball.py"
```

### 4. Python 推理测试

```bash
"./.venv/Scripts/python.exe" "./scripts/infer_volleyball.py" "你的图片或视频路径"
```

## AGX / ZED 部署入口

### 1. 导出 ONNX

```bash
python ./scripts/export_onnx.py --weights ./weight/best.pt --imgsz 640
```

默认输出到：
- `deploy/agx_zed/models/best.onnx`

### 2. 在 AGX 上构建 TensorRT engine

```bash
trtexec --onnx=./deploy/agx_zed/models/best.onnx --saveEngine=./deploy/agx_zed/models/best.engine --fp16
```

### 3. 在 AGX 上构建运行时

```bash
cmake -S ./deploy/agx_zed -B ./deploy/agx_zed/build
cmake --build ./deploy/agx_zed/build
```

### 4. 运行测试

Python：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

AGX C++：

```bash
cd ./deploy/agx_zed/build && ctest --output-on-failure
```

### 5. 启动 AGX 运行时

显式传入配置文件路径：

```bash
./deploy/agx_zed/build/agx_zed_runtime ./configs/deploy/agx_zed_yolo26.yaml ./configs/deploy/camera_pose.yaml
```

- `show: true`：尝试打开 OpenCV 窗口，显示实时画面与 overlay
- `show: false`：无窗口，仅控制台持续输出状态
- GUI 模式建议在 AGX 本地桌面终端启动

## 详细文档

- 训练环境与训练步骤：`docs/01_训练环境搭建与训练步骤.md`
- 部署与推理文档：`docs/02_部署推理文档.md`
- AGX/ZED 子工程说明：`deploy/agx_zed/README.md`
- 仓库结构规范：`docs/03_仓库文件夹架构规范.md`
