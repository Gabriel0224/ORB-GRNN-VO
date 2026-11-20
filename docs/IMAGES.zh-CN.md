# 输入说明

[English](IMAGES.md) | **简体中文**

从 [images.example.yaml](../configs/images.example.yaml) 开始配置。相对路径以数据 YAML 所在目录为基准，示例内参需替换为自己的相机标定。

## 图像与标定

- 提供一条或多条 `train` 序列，以及一条独立的 `validation` 序列，每条至少 11 幅图像。
- 使用 `images` 和 `pattern` 按文件名自然顺序读取目录，或使用 `image_list` 明确指定图像顺序。列表内的相对路径以列表文件所在目录为基准。
- 图像应已去畸变或校正，设置 `rectified: true` 并提供对应的零 skew、3×3 `intrinsics`。同一序列图像尺寸须一致，程序不自动缩放。
- 训练与验证数据保持独立。ORB 前端只处理相邻图像，不读取位姿标签或未来帧。

## 位姿标签

监督训练需要每幅图像对应的相机到世界位姿，使用输入图像对应的相机坐标系。设置 `pose_convention: camera_to_world` 和 `translation_unit: meter`。

`poses` 文本每行是按行展开的 3×4 矩阵：

```text
r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

旋转部分须为有效旋转矩阵，相对标签由 `inverse(T[t]) @ T[t+1]` 生成。时间信息使用严格递增的秒制 `timestamps`，或正数 `fps`。

可选的 `valid_pairs`、`source_ids` 文件每个相邻帧对一项。训练窗口内的帧对须全部有效且属于同一来源；验证序列须连续、完整标注。

## 生成文件

`generated-inputs/` 保存 ORB 观测、点特征及整理后的标签；`prepared/` 保存训练批次。训练完成后两者均保留。开发接口见[中间格式](INPUTS.zh-CN.md)。

仅提取特征、不使用位姿标签：

```bash
grnn-vo-extract --dataset ./dataset.yaml --output ./generated/orb \
  --build-dir ./build/orb
```
