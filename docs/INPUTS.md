# 中间输入格式（低层接口）

一般用户直接使用 `m-train-images` 输入自己的图像，不必手工创建这些文件。以下是自动预处理产物和旧冻结输入的容器格式；没有任何一份实际缓存随代码分发。

```text
p1_inputs/
  manifest.json
  seq00_candidates.npz
  seq02_candidates.npz
  seq07_candidates.npz
  seq08_candidates.npz
  seq05_candidates.npz
  reference/
    manifest.json
    labels/seq00.npz  ...  seq05.npz
    poses/00.txt     ...  05.txt
    calib/00.txt     ...  05.txt
```

输入目录可以另有测试序列，但训练代码不读取它们。它不会自动下载数据，也不会从官方位姿重新生成或替换 fused10 标签。

## 每序列 candidates NPZ

设 `P` 为帧对数，`N` 为全部帧对匹配数之和。

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| offsets | `[P+1]`，整数 | 帧对在扁平节点中的边界，从 0 至 N，单调不减 |
| x | `[N,6]`，float32 | `[x1,y1,x2,y2,x2-x1,y2-y1]` |
| uv | `[N,2,2]`，float64 | 两端点原始浮点 `[u,v]`，不取整、不删像素冲突 |
| features | `[N,2]`，整数 | 两端点各自图像内的特征下标 |
| previous | `[N]`，整数 | 上一个帧对内的匹配行号；没有前驱为 -1 |

匹配取 P1 所选几何模型的全部内点。一个帧对内两端特征下标分别唯一。`previous` 指向上一帧对的后端点特征，与本帧对的前端点特征相同；共享端点浮点坐标也必须一致。第一帧对全部为 -1。每个训练窗口独立打包，窗口首帧对重新切断父边。

逐序列使用标定中的 `P0[:3,:3]` 归一化：`x=(u-cx)/fx, y=(v-cy)/fy`。核验 P0 与 P2 的内参相同；不得固定使用另一个序列的内参。输入中的 geometry、identity 等附加字段不参与 M 网络。

## 标签 NPZ

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| targets | `[P,6]`，float32 | `[yaw,pitch,roll,tx,ty,tz]`；弧度和米 |
| valid_pair | `[P]`，布尔 | 允许训练监督的帧对 |
| source_id | `[P]`，整数 | 冻结标签的来源编号 |
| timestamps_s | `[P+1]` | 原图像时间戳，秒 |
| valid_start_10 | 一维整数 | 冻结的有效十帧对训练窗口起点 |

旋转约定 `R=Rz(yaw) @ Ry(pitch) @ Rx(roll)`；相对位姿为 `inverse(T[t]) @ T[t+1]`。`poses/XX.txt` 为 cam0 到世界的 KITTI 3×4 行格式，标签采用 image_2 相机系。读取器从 P0/P2 的外参核验相机转换后相对标签，误差门槛 `atol=1e-7, rtol=0`。不会把标签 yaw 当作 XZ 平面航向。

训练窗口必须全部 valid 且 source_id 一致；起点从逐帧字段重算，并与 `valid_start_10` 核对。分组先按连续起点区间划分，再每 64 个窗口分组，尾组保留。05 沿原协议使用全部连续窗口和开头递增短窗口，保证每个真实帧对最终恰有一次轨迹预测。

## 清单字段

`manifest.json` 需要 `sequences` 映射。每个序列至少有：

```json
{
  "sequences": {
    "00": {
      "pairs": 4540,
      "windows": 4387,
      "batches": 78,
      "all_geometric_matches": 2628367,
      "P0_intrinsics": [[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]],
      "point_cache_sha256": "<sha256>",
      "labels_sha256": "<sha256>",
      "ground_truth_sha256": "<sha256>",
      "calibration_sha256": "<sha256>"
    }
  }
}
```

示例中的摘要是占位符；实际须填写对应文件的 SHA-256，所有训练/验证序列均须具备记录。

`reference/manifest.json` 每序列至少有 `labels_file`（相对 reference 的路径）、`labels_sha256`、`cam0_pose_text_sha256`。路径不得越出输入目录。

准备时核验每个文件哈希，输出自己的 `input_audit.json`，只记录相对文件名。训练和恢复会再次核对冻结输入、预打包批次和数据顺序。哈希锁只防止输入被悄悄替换，不替代数据授权或标签生成方法的说明。

## 自备图像适配

`m-extract --with-labels` / `m-train-images` 自动产生相同布局，序列名可为 `train_a` 等安全名称。相机内参来自数据YAML；容器中的 `P0_intrinsics` 与 `calib/P0,P2` 是兼容字段，两者填写同一实际输入相机K、零外参，不表示使用另一个真实相机。`reference/manifest.json` 中 `pose_frame: camera` 表明位姿已是该输入相机系，读取器不做cam0外参转换。旧fused10容器缺省 `pose_frame: cam0`，继续执行原image_2相机转换。两种来源不会混为一谈。
