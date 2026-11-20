# M：从自备图像到训练

本代码包包含 **P1 前端、图像到点集的预处理、M 模型、监督训练、完整验证及断点恢复**。用户输入自己的有序图像，不需要提供预先生成的 P1 缓存。没有附带数据、标签、权重、缓存、预编译程序、测试集评测脚本或测速脚本。

**训练需要图像、相机内参和对应位姿标签。** 只有图像与内参时可以提取匹配点，但不能训练当前这个有监督模型。该包没有将 M 改成自监督模型，也不包含从无标签图像估计训练真值的程序。

本项目主许可证为 **GPL-3.0-or-later**（GNU GPL 第3版或更新版本），见 [LICENSE](LICENSE) 和 [许可范围](LICENSING.md)。P1 的 GPL 及 MLLA 的 MIT 来源声明继续保留，见 [第三方声明](THIRD_PARTY_NOTICES.md)。GPL 允许遵守条款的商业使用；本项目没有另加非商业限制。

## 一条命令训练自己的图像

需要 Python 3.10–3.12、C++ 编译器、CMake、OpenCV 4 开发库和 Eigen3。Linux Debian/Ubuntu 的原生构建依赖可安装为：

```bash
sudo apt-get install build-essential cmake libopencv-dev libeigen3-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch 版本为 2.4.1；GPU 训练应安装与本机驱动兼容的 CUDA 构建。Python OpenCV 与系统 C++ OpenCV 是不同依赖，安装 Python wheel 不能代替 `libopencv-dev`。安装本包不会下载数据或模型。P1 不需要 ORB 词袋文件或任何预训练权重。

复制 `configs/images.example.yaml` 为自己的 `dataset.yaml`，填写：

- 一个或多个训练序列、一个独立验证序列，各至少 11 幅按时间排列的图像；不要混用训练和验证片段。
- 每个序列真实的相机内参，以及已经去畸变/校正的图像。
- 同一相机的逐图像 `camera_to_world` 位姿，平移单位米；时间戳或帧率。

详细约定见 [原始图像输入说明](docs/IMAGES.md)。然后运行：

```bash
m-train-images --dataset ./dataset.yaml --config configs/m_images.yaml \
  --work-dir ./runs/m_own_images --device cuda:0
```

程序自动完成：编译 P1 → 按图像顺序提取和匹配 → H/F 几何内点筛选 → 六维点特征和跨帧关联 → 标签与窗口核验 → CPU 批次预打包 → seed42 随机初始化训练 → 每轮完整验证 → 选权与完成核验。

`--work-dir` 必须是新目录，不能指向源图像目录。所有输入只读。**程序正常生成 P1 点缓存和预打包批次，训练完成后继续保留，不自动删除。** 中断和恢复时也保留它们。源码发行包及 Git 仓库不含我们的既有缓存、数据或权重；用户本机运行后产生的缓存属于用户的运行产物。

先检查自己的数据、不启动训练：

```bash
m-train-images --dataset ./dataset.yaml --config configs/m_images.yaml \
  --work-dir ./runs/m_own_images --device cuda:0 --prepare-only
```

之后同样命令增加 `--resume`，并去掉 `--prepare-only`。训练期间安全暂停和恢复也使用这个入口：

```bash
m-train-images --dataset ./dataset.yaml --config configs/m_images.yaml \
  --work-dir ./runs/m_own_images --device cuda:0 --resume
```

`SIGINT` / `SIGTERM` 请求训练器在安全提交点保存状态。恢复核验配置、源码、图像内容及顺序、位姿、时间戳、标签有效性/来源边界、输入和预打包文件哈希。失败时保留日志和中间产物，不覆盖已有结果，不自动重试。

## 模型和默认协议

```text
当前及上一张图像
→ P1：ORB提取、相邻帧匹配、H/F RANSAC内点
→ 每个匹配的 [x1,y1,x2,y2,dx,dy]
→ MLP 6→64→64
→ 两层因果轨迹图层，宽64
→ Point-MILA，宽64、4头、self+8近邻
→ concat(mean,max)，128维
→ 两层单向LSTM128，dropout=0.2
→ Linear(128,6)
```

**345,542 个可训练参数**。几何向量 G 不输入网络。保留浮点端点及全部被选中模型的内点，不做整数像素冲突删点；同一张图像的特征下标用于跨帧连接。Point-MILA 的 RoPE 使用第二端点浮点 UV/8，等距近邻按第二端点特征下标排序。

| 项目 | 默认值 |
|---|---|
| 初始化 | seed42，全部模型随机初始化，新 Adam |
| 窗口 | 10 个连续帧对，即 11 幅图像；步长1 |
| batch | 64；按连续有效窗口区间分组，保留尾组 |
| 损失 | 六轴加权 MSE，`[300,300,300,1,1,1]`；监督窗口内全部位置 |
| 优化器 | Adam，默认 betas/epsilon，weight_decay=0，clip=10 |
| 精度 | FP32，关闭 CUDA matmul/cuDNN TF32；LSTM 使用 ATen 路径 |
| 学习率 | `1e-4,7e-5,4e-5,1e-5,7e-6,4e-6,1e-6,7e-7,4e-7,1e-7` |
| 调度与停止 | 完整验证 loss 连续20轮不严格改善降档；最低档再20轮不改善停止；最多300轮 |
| 选权 | 实际完成的最后50轮中，分别选择最低验证 loss 和最低未滤波初始位姿对齐 ATE |

自备数据的窗口数按实际图像和标签计算。旧 KITTI/fused10 协议的 13,736/246 训练窗口/组、2,751/43 验证窗口/组只适用于旧冻结数据，**不是自备图像必须满足的数量**。不同数据及相机标定不保证复现已有实验数值。

每个窗口独立切断外部图连接并重置 LSTM；验证按截至当前的最近10帧对输出新位姿，开头使用递增长度短窗口，不读取未来帧。完整验证保留全部帧对，不能用少量批次代替。验证 ATE 不参与学习率调度；相同 loss/ATE 保留较早轮次。没有测试集入口参与选权。

## 缓存是什么，是否影响公平性

此处缓存只是**同一组原图经固定 P1 计算得到的点特征**，避免每个 epoch 重复提取 ORB 和 RANSAC。P1 只读取相邻图像与前端设置；点集构造只读取匹配记录和相机内参。位姿标签单独读取，仅用于监督和验证，不进入特征计算。追加未来图像不应改变此前帧对的输入。

因此使用这种缓存并不增加模型可获得的信息，也不会因缓存本身改变任务。若预处理使用真值、未来帧或测试结果才会造成泄漏，不能只凭“有缓存”判断是否泄漏。报告系统速度时必须包括读图、P1、点集构建和网络等实际阶段，不能用跳过前端的缓存推理速度冒充完整流程速度。

这里的“从图像到训练”是完整的软件使用流程。P1 是固定、非可微前端；M 仍学习点集特征，**不是直接对 RGB 像素反向传播的端到端视觉模型**。

## 独立预处理入口

只从图片提取 P1（无需位姿标签）：

```bash
m-extract --dataset ./dataset.yaml --output ./generated/p1 \
  --build-dir ./build/p1 --seed 42
```

`m-extract` 是显式保留预处理结果的工具，不自动清理它的输出。需要同时生成自备监督标签时加 `--with-labels`。一般用户使用上面的一键训练入口即可。

兼容已有冻结 P1/fused10 数据的低层入口 `m-prepare` / `m-train` 也保留，见 [中间文件格式](docs/INPUTS.md)。`configs/m.yaml` 与 `m_compact.yaml` 用于旧数据协议；自备图片使用 `m_images.yaml`。包内不提供任何一套缓存或 fused10 标签。

## 执行与输出

默认使用参考 canonical32 分块和激活检查点。`implementation: compact` 可以复用相同图上下文的计算，保留相同 attention padding；不得跨不同上下文随意复用。资源选项不改变 batch、点数、loss 或 FP32；在新设备更改执行策略应先核验等价。包内不带机器专用 CUDA Graph 捕获、调优探针或测试脚本，不承诺某台机器的历史吞吐。

训练输出在 `work-dir/training/`：

- `initial.pt`、`last.pt`、每轮 `epoch_XXX.pt`、`final.pt`。
- 全历史过程最优 `best_loss_all.pt` / `best_ate_origin_all.pt`。
- 最后50轮选权 `best_loss_last50.pt` / `best_ate_origin_last50.pt` 与 `selection.json`。
- `steps.jsonl`、`training.jsonl`、每轮完整验证轨迹 `trajectories/epoch_XXX.npz`。
- 配置、运行环境、数据顺序及完成记录。

每50步、每轮训练结束和完整验证结束原子保存恢复点。checkpoint 保存标准 `model_state_dict`，以及 Adam、调度、全部随机状态、epoch/batch、损失累积和最佳记录。旧 M 模型参数可严格加载；旧训练器的完整状态不能直接作为本入口的恢复格式。仅加载可信来源 checkpoint；完整恢复需 `torch.load(..., weights_only=False)`。

`input_provenance.json` 和 `intermediate_inventory.json` 保存输入与构建哈希。生成的点缓存位于 `generated-inputs/`，预打包批次位于 `prepared/`；它们与训练产物一起保留，便于复用和复核。程序没有训练结束后自动删除缓存的步骤。运行产物不要提交到源码仓库。

该源代码包排除了私人路径、主机地址、凭据、原 Git 历史、个人记录、数据、权重、缓存、结果、测试集评测/测速/消融脚本和面板。**训练所需的验证代码仍保留。**
