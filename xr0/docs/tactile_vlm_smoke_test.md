# XR0 触觉五路图像训练冒烟测试

## 结论先说

可以实现。

XR0 现在这套训练链路并没有把输入图像数量写死成 3 张，真正起作用的是：

- `instruction.general[].images`
- 人类提示词里的多个 `<image>`
- `JsonDataset` 按 `images` 里的键去读视频
- `CustomCollate` 再把这些图像一起送进 `Qwen3-VL`

所以从机制上说，把：

- `基座相机`
- `左腕相机`
- `右腕相机`

扩成：

- `基座相机`
- `左腕相机`
- `右腕相机`
- `左夹爪触觉热力图`
- `右夹爪触觉热力图`

是可行的。

## 这次补丁里提供了什么

新增了两个最小工具：

- `tools/add_tactile_views_to_json_dataset.py`
  把 XR0 的 JSON 标注扩成 5 路图像提示。
- `tools/tactile_multiview_batch_smoke.py`
  不跑整套大模型训练，只验证 “数据集读取 + prompt 组装 + processor 编码” 能不能吃下 5 张图。
- `tools/generate_mock_xr0_dataset.py`
  当你手上还没有任何 XR0 真数据时，先生成一套完全假的基础数据集，用来做训练冒烟测试。

## 如果你现在一条 XR0 数据都没有

先生成一套假的基础 XR0 数据：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/generate_mock_xr0_dataset.py \
  --output-root data/mock_xr0_base \
  --episodes 2 \
  --num-frames 48 \
  --fps 10
```

这条命令会生成：

- `data/mock_xr0_base/json/*.json`
- `data/mock_xr0_base/videos/*_ego.mp4`
- `data/mock_xr0_base/videos/*_wrist_left.mp4`
- `data/mock_xr0_base/videos/*_wrist_right.mp4`
- `data/mock_xr0_base/videos/*_tactile_left.mp4`
- `data/mock_xr0_base/videos/*_tactile_right.mp4`

注意：

- 这是一套“格式正确但内容是假的”数据
- 目的不是训练有效策略，而是验证你后面的 5 路图像 + 触觉热力图训练链路能不能跑通
- 如果你的环境里没有视频写入依赖，先装：

```bash
pip install opencv-python-headless imageio imageio-ffmpeg
```

生成完以后，再把基础 JSON 扩成触觉五路版本：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/add_tactile_views_to_json_dataset.py \
  --json-dir data/mock_xr0_base/json \
  --output-dir data/mock_xr0_tactile/json \
  --left-source replace_wrist_left_suffix \
  --right-source replace_wrist_right_suffix \
  --check-files
```

这样更接近你未来真实采集时的流程：

- 基础 XR0 数据先有 3 路 RGB JSON
- 触觉热力图视频单独存在
- 再把 3 路 JSON 扩成 5 路 JSON

## 最小思路

先不要一上来就接真实触觉。

先拿现有的 3 路相机数据，临时把：

- `左触觉热力图` 复用成 `左腕相机`
- `右触觉热力图` 复用成 `右腕相机`

这样虽然没有真实触觉信息，但可以最快验证：

- JSON 格式能不能过
- prompt 能不能过
- `Qwen3-VL` 的 processor 能不能过
- 一步训练能不能跑起来

等这个 smoke test 通了，再换成真实的触觉热力图 mp4。

## 第一步：把 JSON 扩成五路图像

假设你原始 XR0 数据集已经有了，并且在：

```bash
data/bi_piper_xr0/json
```

运行：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/add_tactile_views_to_json_dataset.py \
  --json-dir data/bi_piper_xr0/json \
  --output-dir data/bi_piper_xr0_tactile/json \
  --left-source reuse_wrist_left \
  --right-source reuse_wrist_right
```

这一步会做两件事：

- 给每个 episode JSON 新增
  - `observations.tactile_left`
  - `observations.tactile_right`
- 把提示词改成 5 张图版本：
  - `# Ego View`
  - `# Left-Wrist View`
  - `# Right-Wrist View`
  - `# Left-Gripper Tactile Heatmap`
  - `# Right-Gripper Tactile Heatmap`

## 第二步：先做轻量级 batch 冒烟

这一步不真正训练，只测数据和 processor。

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/tactile_multiview_batch_smoke.py \
  --json-dir data/bi_piper_xr0_tactile/json \
  --batch-size 1
```

如果通过，你会看到类似输出：

```text
images_per_sample=5
...
Smoke test passed: XR0 preprocessing accepted the 5-view prompt.
```

这说明：

- JSON 读取正常
- 5 张图 prompt 正常
- `AutoProcessor.apply_chat_template(...)` 正常

## 第三步：生成一个最小训练配置

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/prepare_xr0_dataset.py \
  --dataset-root data/bi_piper_xr0 \
  --json-dir data/bi_piper_xr0_tactile/json \
  --dataset-name tactile_smoke \
  --batch-size 1 \
  --trainer-max-steps 1 \
  --trainer-val-check-interval 1 \
  --trainer-save-interval 1 \
  --train-config-out configs/data/tactile_smoke.yaml \
  --stats-out data/bi_piper_xr0_tactile/stats_action30.json
```

## 第四步：跑 1 step 训练冒烟

这是最简单的“真正训练”冒烟测试。

```bash
cd ~/xiaomi/xr0
conda activate mibot

export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  data=tactile_smoke \
  model=XR0 \
  trainer.project="xr0_smoke" \
  trainer.exp_name="tactile_5view_smoke" \
  trainer.default_root_dir="./runs/tactile_5view_smoke" \
  trainer.max_steps=1 \
  trainer.val_check_interval=1 \
  trainer.save_interval=1 \
  model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
```

如果这一步能过，说明最关键的问题已经验证了：

- 5 路图像输入可以进入 XR0 训练链路
- 触觉热力图作为额外“图像模态”是可训练的

## 真正接入真实触觉时，采数据是不是每个 episode 多两个 mp4 就够了

大体上是，对训练侧来说基本就是这样。

每个 episode 额外准备：

- `episode_xxx_tactile_left.mp4`
- `episode_xxx_tactile_right.mp4`

同时满足下面几点：

- 和三路 RGB 视频严格同步
- `fps` 一致
- `num_frames` 一致
- 每一帧都对应同一个控制时刻

然后在 JSON 里加：

- `observations.tactile_left`
- `observations.tactile_right`

并把 prompt 改成 5 张图版本，就可以进入训练。

## 真实触觉视频怎么接进 JSON

如果你的触觉视频命名规则是：

- `episode_xxx_wrist_left.mp4`
- `episode_xxx_wrist_right.mp4`
- `episode_xxx_tactile_left.mp4`
- `episode_xxx_tactile_right.mp4`

那可以直接用下面这条命令批量改 JSON：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/add_tactile_views_to_json_dataset.py \
  --json-dir data/bi_piper_xr0/json \
  --output-dir data/bi_piper_xr0_tactile_real/json \
  --left-source replace_wrist_left_suffix \
  --right-source replace_wrist_right_suffix \
  --check-files
```

默认替换规则是：

- `_wrist_left.mp4 -> _tactile_left.mp4`
- `_wrist_right.mp4 -> _tactile_right.mp4`

## 一个很重要的提醒

上面这些改动解决的是“训练输入”。

如果你后面还想让部署时也真的使用触觉，那还要继续改在线推理链路，也就是：

- 运行时实时生成左/右触觉热力图
- 推理客户端把这两张图也送进 VLM
- 在线 prompt 同样改成 5 张图版本

也就是说：

- `训练能不能做`：能
- `只要多两个 mp4 能不能训练`：基本能
- `部署时是不是也自动有触觉能力`：不是，还要补在线推理输入

## 推荐顺序

推荐你按这个顺序做：

1. 先用复用腕部视频的方式跑通 5 图像 smoke test
2. 再把真实触觉热力图 mp4 接进数据集
3. 再决定要不要改 `runtime client` 做在线 5 图像推理
