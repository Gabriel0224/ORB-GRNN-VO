# 中间格式

[English](INPUTS.md) | **简体中文**

图像处理流程会自动生成这些文件，普通使用只需参考[输入说明](IMAGES.zh-CN.md)。

设相邻帧对数为 `P`、总匹配数为 `N`，每个 `seq<name>_candidates.npz` 包含：

| 字段 | 形状 | 内容 |
|---|---|---|
| `offsets` | P+1 | 各帧对点集的边界 |
| `x` | N×6，float32 | 归一化的 `[x1,y1,x2,y2,dx,dy]` |
| `uv` | N×2×2，float64 | 两端浮点像素坐标 |
| `features` | N×2 | 两张图像内的特征下标 |
| `previous` | N | 前一帧对内的匹配行号，无前驱为 −1 |

匹配取 ORB 前端的几何内点。跨帧边使用共享图像的特征身份，每个窗口首帧对切断外部连接；坐标使用本序列内参归一化。

`reference/labels/seq<name>.npz` 包含 `targets`、`valid_pair`、`source_id`、`timestamps_s`、`valid_start_10`。标签为 `[yaw,pitch,roll,tx,ty,tz]`，单位弧度和米，旋转约定为 `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

清单记录文件哈希、标定、数量和标签约定。自备图像位姿使用 `pose_frame: camera`，旧 KITTI 读取接口支持 `pose_frame: cam0`，两种坐标系分别处理。自备输入生成的 `P0_intrinsics`、P0/P2 字段填写用户提供的相机标定。

预打包保留独立的十帧对窗口和原分组规则。准备及恢复时核验哈希；生成文件保留在用户本地，不随源码仓库分发。
