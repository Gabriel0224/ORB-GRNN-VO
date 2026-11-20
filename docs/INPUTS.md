# Intermediate format

**English** | [简体中文](INPUTS.zh-CN.md)

The image pipeline generates these files automatically. Users normally only need the [input guide](IMAGES.md).

For `P` adjacent frame pairs and `N` total matches, each `seq<name>_candidates.npz` contains:

| Field | Shape | Content |
|---|---|---|
| `offsets` | P+1 | Boundaries of each pair's points |
| `x` | N×6, float32 | Normalized `[x1,y1,x2,y2,dx,dy]` |
| `uv` | N×2×2, float64 | Floating-point pixel endpoints |
| `features` | N×2 | Feature indices in the two images |
| `previous` | N | Match index in the preceding pair, or −1 |

Matches are the ORB frontend's geometric inliers. Temporal edges use shared-image feature identities and are cut at each window boundary. Coordinates are normalized with the sequence's own intrinsics.

`reference/labels/seq<name>.npz` stores `targets`, `valid_pair`, `source_id`, `timestamps_s`, and `valid_start_10`. Targets are `[yaw,pitch,roll,tx,ty,tz]` in radians and meters, with `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

Manifests record file hashes, calibration, counts, and label conventions. Custom-image poses use `pose_frame: camera`; the legacy KITTI reader supports `pose_frame: cam0`. These are distinct coordinate conventions. The generated `P0_intrinsics` and P0/P2 calibration fields hold the supplied camera calibration for custom inputs.

Prepared batches preserve independent ten-pair windows and the original grouping rules. Hashes are checked during preparation and recovery. Generated files are retained locally and excluded from this source repository.
