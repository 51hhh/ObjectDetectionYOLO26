# ObjectDetectionYOLO26

基于 Ultralytics YOLO26 的排球检测项目。

当前项目已完成：
- 数据集整理与训练集生成
- YOLO26 本地训练
- 推理脚本编写
- 仓库结构规范化

## 仓库目录结构

```text
ObjectDetectionYOLO26/
├─ configs/                     # 训练配置文件
│  └─ train_volleyball.yaml
├─ docs/                        # 项目文档
│  ├─ 01_训练环境搭建与训练步骤.md
│  ├─ 02_部署推理文档.md
│  └─ 03_仓库文件夹架构规范.md
├─ scripts/                     # 可执行脚本
│  ├─ prepare_dataset.py        # 数据集转换脚本
│  ├─ train_volleyball.py       # 训练脚本
│  └─ infer_volleyball.py       # 推理脚本
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

### 4. 推理测试

```bash
"./.venv/Scripts/python.exe" "./scripts/infer_volleyball.py" "你的图片或视频路径"
```

## 详细文档

- 训练环境与训练步骤：`docs/01_训练环境搭建与训练步骤.md`
- 部署与推理文档：`docs/02_部署推理文档.md`
- 仓库结构规范：`docs/03_仓库文件夹架构规范.md`
