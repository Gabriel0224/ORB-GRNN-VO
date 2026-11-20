# ORB-GRNN-VO

[English](README.md) | **简体中文**

结合 **ORB 前端**与 **GRNN-VO 网络**的单目视觉里程计。本仓库提供图像预处理、监督训练、验证和断点恢复代码。

```text
图像序列 → ORB 特征匹配 → 图建模与时序建模 → 相机相对位姿
```

## 安装

使用 Python 3.10–3.12。GPU 训练需要与本机 CUDA 环境兼容的 PyTorch。

```bash
sudo apt-get install build-essential cmake libopencv-dev libeigen3-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 训练

准备自己的图像序列、相机内参和相机位姿标签。训练集与验证集相互独立，每条序列至少包含 11 幅图像。

将[数据配置示例](configs/images.example.yaml)复制为 `dataset.yaml`，填写路径和标定参数。位姿格式见[输入说明](docs/IMAGES.zh-CN.md)。

```bash
grnn-vo-train --dataset ./dataset.yaml --config configs/grnn_vo.yaml \
  --work-dir ./runs/grnn_vo --device cuda:0
```

程序自动编译 ORB 前端、生成特征并开始训练。添加 `--prepare-only` 可先完成输入准备，添加 `--resume` 可继续已有训练。训练设置见[配置文件](configs/grnn_vo.yaml)。

权重、日志和验证轨迹保存在运行目录中，生成的缓存会保留以便复用。仓库只提供源码，不包含数据集、模型权重或预生成缓存。

## 许可证

[GPL-3.0-or-later](LICENSE)。ORB-SLAM3 和 MLLA 的来源及声明见[第三方许可说明](THIRD_PARTY_NOTICES.md)。
