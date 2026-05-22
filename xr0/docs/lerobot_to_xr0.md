# LeRobot 转 XR0 数据集

本文档只说明一个脚本：

- `tools/convert_lerobot_to_xr0.py`

它的作用是把 LeRobot 数据集转换成 XR0 训练直接可用的格式，并完成下面三件事：

1. 按 `episode` 拆分视频。
2. 把 `joint / gripper` 映射到 XR0 的 `proprios` 和 `actions` 字段。
3. 用 PiPER FK 计算 `ee_pos / ee_rotm`。

## 1. 输出格式

转换完成后，输出目录会长这样：

```text
data/your_dataset_xr0/
├── json/
│   ├── episode_000000.json
│   ├── episode_000001.json
│   └── ...
└── videos/
    ├── episode_000000_ego.mp4
    ├── episode_000000_wrist_left.mp4
    ├── episode_000000_wrist_right.mp4
    └── ...
```

这些 `json/` 和 `videos/` 可以直接接到 XR0 的数据准备和训练流程里。

## 2. 运行前准备

在 Linux 训练机上进入 XR0 项目：

```bash
cd ~/xiaomi/xr0
conda activate mibot
```

如果环境里还没有依赖，先安装：

```bash
pip install pyarrow imageio imageio-ffmpeg opencv-python-headless
```

如果 `piper_sdk` 不在当前 Python 环境里，但在某个源码目录下，可以后面运行命令时加：

```bash
--piper-sdk-root /path/to/piper_sdk_parent
```

## 3. 最常用命令

这是最直接的转换命令，适合你现在这份不带触觉的 LeRobot 数据：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/convert_lerobot_to_xr0.py \
  --lerobot-root /home/enine/SACM/lerobot_dataset/5_18_hil_fold_towel \
  --output-root ./data/5_18_hil_fold_towel_xr0 \
  --ego-camera-key left_top \
  --left-wrist-camera-key left_wrist \
  --right-wrist-camera-key right_wrist \
  --joint-unit deg \
  --position-unit mm
```

这条命令的含义是：

- 读取 LeRobot 数据集根目录 `/home/enine/SACM/lerobot_dataset/5_18_hil_fold_towel`
- 输出到 `./data/5_18_hil_fold_towel_xr0`
- 把 `left_top` 视角映射成 XR0 的 `ego`
- 把 `left_wrist` / `right_wrist` 映射成 XR0 的左右腕部视角
- 按角度制 `deg` 解释 joint
- FK 输出位置单位使用 `mm`

## 4. 如果想手动指定任务文本

如果你不想用 `meta/tasks.parquet` 里的任务描述，或者数据里没有任务文本，可以手动指定：

```bash
python tools/convert_lerobot_to_xr0.py \
  --lerobot-root /home/enine/SACM/lerobot_dataset/5_18_hil_fold_towel \
  --output-root ./data/5_18_hil_fold_towel_xr0 \
  --ego-camera-key left_top \
  --left-wrist-camera-key left_wrist \
  --right-wrist-camera-key right_wrist \
  --joint-unit deg \
  --position-unit mm \
  --task "Fold the towel with both arms."
```

## 5. 只转一部分 episode

只转换一段 episode，适合先做小规模检查：

```bash
python tools/convert_lerobot_to_xr0.py \
  --lerobot-root /home/enine/SACM/lerobot_dataset/5_18_hil_fold_towel \
  --output-root ./data/5_18_hil_fold_towel_xr0_debug \
  --ego-camera-key left_top \
  --left-wrist-camera-key left_wrist \
  --right-wrist-camera-key right_wrist \
  --joint-unit deg \
  --position-unit mm \
  --episode-start 0 \
  --episode-end 1
```

或者只保留前 2 个 episode：

```bash
python tools/convert_lerobot_to_xr0.py \
  --lerobot-root /home/enine/SACM/lerobot_dataset/5_18_hil_fold_towel \
  --output-root ./data/5_18_hil_fold_towel_xr0_debug \
  --ego-camera-key left_top \
  --left-wrist-camera-key left_wrist \
  --right-wrist-camera-key right_wrist \
  --joint-unit deg \
  --position-unit mm \
  --max-episodes 2
```

## 6. 常用参数

- `--lerobot-root`
  - LeRobot 数据集根目录，里面应包含 `data/`、`meta/`、`videos/`
- `--output-root`
  - XR0 输出目录，脚本会在里面生成 `json/` 和 `videos/`
- `--ego-camera-key`
  - LeRobot 里哪一路相机映射到 XR0 的 `observations.ego`
- `--left-wrist-camera-key`
  - LeRobot 里哪一路相机映射到 XR0 的 `observations.wrist_left`
- `--right-wrist-camera-key`
  - LeRobot 里哪一路相机映射到 XR0 的 `observations.wrist_right`
- `--joint-unit`
  - LeRobot 的 `observation.state` 和 `action` 里 joint 的单位，支持 `deg` 或 `rad`
- `--position-unit`
  - FK 输出的末端位置单位，支持 `mm` 或 `m`
- `--task`
  - 强制覆盖任务文本
- `--episode-start` / `--episode-end`
  - 只转换指定范围的 episode
- `--max-episodes`
  - 最多只转前 N 个 episode
- `--absolute-video-paths`
  - JSON 里写绝对路径；默认写相对 `xr0` 项目根目录的相对路径
- `--piper-sdk-root`
  - `piper_sdk` 无法 import 时，用这个参数补充 Python 路径

## 7. 默认假设

这份脚本目前默认以下前提成立：

1. `observation.state` 是双臂 14 维：
   - 左臂 6 维 joint + 1 维 gripper
   - 右臂 6 维 joint + 1 维 gripper
2. `action` 也是双臂 14 维，并且表示目标 `joint / gripper`，不是 delta。
3. 每个 `episode` 在源数据里对应一段连续帧序列。
4. 三路视频和 parquet 行数逐帧对齐。

如果你的 LeRobot 数据不满足这几个前提，需要再改一下脚本。

## 8. 转完之后怎么检查

先看输出目录：

```bash
find ./data/5_18_hil_fold_towel_xr0 -maxdepth 2 | sort
```

再看生成了多少 JSON：

```bash
find ./data/5_18_hil_fold_towel_xr0/json -name "*.json" | wc -l
```

随便打开一个 JSON 看字段：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("./data/5_18_hil_fold_towel_xr0/json/episode_000000.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(data.keys())
print(data["observations"].keys())
print(data["proprios"].keys())
print(data["actions"].keys())
print(data["num_frames"])
PY
```

## 9. 帮助命令

忘了参数时直接看：

```bash
python tools/convert_lerobot_to_xr0.py --help
```
