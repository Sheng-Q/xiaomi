# XR0 触觉五路图像冒烟测试

## 结论

这条路是可行的。

XR0 当前训练链路并没有把输入图像数量写死成 3 张，真正决定多视角输入的是：

- `instruction.general[].images`
- 人类 prompt 里的多个 `<image>`
- `JsonDataset` 按 `images` 里的键读取视频
- `CustomCollate` 再把这些图像一起送进 `Qwen3-VL`

所以，把原来的 3 路图像：

- `observations.ego`
- `observations.wrist_left`
- `observations.wrist_right`

扩成 5 路图像：

- `observations.ego`
- `observations.wrist_left`
- `observations.wrist_right`
- `observations.tactile_left`
- `observations.tactile_right`

从机制上是成立的。

这次已经完成的验证是：

- mock XR0 数据生成成功
- 5 路图像 JSON 扩展成功
- `Qwen3-VL` 的 batch preprocess smoke test 成功
- 真正训练已跑到 `Epoch 0: 100% | 1/1`

最后一次失败发生在 checkpoint 保存阶段，原因是磁盘空间不足，不是数据链路或模型链路错误。  
也就是说，从“触觉 5 视角能不能接进 XR0 训练”这个问题本身看，冒烟测试已经通过。

## 本文档只保留一条已验证路径

下面这条流程的设计原则是：

1. 下一步只使用上一步刚产出的文件
2. 先用 mock 数据把链路跑通
3. 训练命令避开已经踩过的两个坑
   - DeepSpeed `FusedAdam` 编译失败
   - `AdamW` 在单卡上额外吃显存

本流程用到的脚本分两类：

- 这次补丁新增的脚本
  - `tools/generate_mock_xr0_dataset.py`
  - `tools/add_tactile_views_to_json_dataset.py`
  - `tools/tactile_multiview_batch_smoke.py`
- XR0 原仓已有脚本
  - `tools/prepare_xr0_dataset.py`

## 文件流转总览

按下面这条链走，不要跳步骤：

1. `tools/generate_mock_xr0_dataset.py`
   产出 `data/mock_xr0_base/json/*.json` 和 `data/mock_xr0_base/videos/*.mp4`
2. `tools/add_tactile_views_to_json_dataset.py`
   读取 `data/mock_xr0_base/json`
   产出 `data/mock_xr0_tactile/json`
3. `tools/tactile_multiview_batch_smoke.py`
   读取 `data/mock_xr0_tactile/json`
4. `tools/prepare_xr0_dataset.py`
   读取 `data/mock_xr0_tactile/json`
   产出 `configs/tactile_smoke.yaml` 和 `data/mock_xr0_tactile/stats_action30.json`
5. `bash scripts/train.sh --config-name tactile_smoke ...`
   读取上一步产出的 `configs/tactile_smoke.yaml`

## 第 0 步：环境和可选清理

```bash
cd ~/xiaomi/xr0
conda activate mibot
```

如果你之前跑过同名实验，建议先清掉旧输出，避免误判：

```bash
rm -rf data/mock_xr0_base
rm -rf data/mock_xr0_tactile
rm -rf runs/tactile_5view_smoke
```

如果环境里没有视频写入依赖，先装：

```bash
pip install opencv-python-headless imageio imageio-ffmpeg
```

## 第 1 步：生成 mock XR0 基础数据

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/generate_mock_xr0_dataset.py \
  --output-root data/mock_xr0_base \
  --episodes 2 \
  --num-frames 48 \
  --fps 10
```

这一步会生成：

- `data/mock_xr0_base/json/mock_episode_001.json`
- `data/mock_xr0_base/json/mock_episode_002.json`
- `data/mock_xr0_base/videos/*_ego.mp4`
- `data/mock_xr0_base/videos/*_wrist_left.mp4`
- `data/mock_xr0_base/videos/*_wrist_right.mp4`
- `data/mock_xr0_base/videos/*_tactile_left.mp4`
- `data/mock_xr0_base/videos/*_tactile_right.mp4`

下一步只使用这里刚生成的：

- `data/mock_xr0_base/json`

## 第 2 步：把 3 路 JSON 扩成 5 路 JSON

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

这一步会做两件事：

- 在每个 episode JSON 里新增
  - `observations.tactile_left`
  - `observations.tactile_right`
- 把 prompt 从 3 张图改成 5 张图

这一步的输出是：

- `data/mock_xr0_tactile/json/*.json`

下一步只使用这里刚生成的：

- `data/mock_xr0_tactile/json`

## 第 3 步：先做轻量级 preprocess 冒烟

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/tactile_multiview_batch_smoke.py \
  --json-dir data/mock_xr0_tactile/json \
  --batch-size 1
```

期望看到类似输出：

```text
dataset_files=2
dataset_samples=...
images_per_sample=5
...
Smoke test passed: XR0 preprocessing accepted the 5-view prompt.
```

这一步如果通过，说明下面三件事已经成立：

- JSON 读取正常
- 5 张图 prompt 正常
- `Qwen3-VL` 的 processor 能处理这 5 路输入

下一步仍然只使用：

- `data/mock_xr0_tactile/json`

## 第 4 步：生成训练配置

这里用的是 XR0 原仓里的 `tools/prepare_xr0_dataset.py`。  
关键点是：训练配置要直接输出成顶层 config，不能再写成 `configs/data/tactile_smoke.yaml` 后面却用 `--config-name tactile_smoke` 去找。

推荐命令：

```bash
cd ~/xiaomi/xr0
conda activate mibot

python tools/prepare_xr0_dataset.py \
  --dataset-root data/mock_xr0_tactile \
  --json-dir data/mock_xr0_tactile/json \
  --dataset-name tactile_smoke \
  --batch-size 1 \
  --trainer-max-steps 1 \
  --trainer-val-check-interval 1 \
  --trainer-save-interval 999999 \
  --train-config-out configs/tactile_smoke.yaml \
  --stats-out data/mock_xr0_tactile/stats_action30.json
```

这一步的关键输出是：

- `configs/tactile_smoke.yaml`
- `data/mock_xr0_tactile/stats_action30.json`

下一步只使用这里刚生成的：

- `configs/tactile_smoke.yaml`

## 第 5 步：跑 1 step 真正训练冒烟

### 推荐命令

这条是当前最适合单卡 smoke test 的版本：

```bash
cd ~/xiaomi/xr0
conda activate mibot

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  --config-name tactile_smoke \
  trainer.project="xr0_smoke" \
  trainer.exp_name="tactile_5view_smoke" \
  trainer.default_root_dir="./runs/tactile_5view_smoke" \
  trainer.max_steps=1 \
  trainer.val_check_interval=1 \
  trainer.save_interval=999999 \
  model.params.model.training_repeat=1 \
  trainer.optimizer.type="torch.optim.SGD" \
  'trainer.optimizer.params.lr=1e-4' \
  'trainer.optimizer.params.weight_decay=0.1' \
  '~trainer.optimizer.params.betas' \
  '~trainer.optimizer.params.eps'
```

这条命令这样写的原因是：

- `--config-name tactile_smoke`
  直接使用上一步产出的 `configs/tactile_smoke.yaml`
- `model.params.model.training_repeat=1`
  降低训练侧重复展开带来的额外压力
- `trainer.optimizer.type="torch.optim.SGD"`
  避开 DeepSpeed `FusedAdam` 的 CUDA 编译问题
- 去掉 `betas` 和 `eps`
  避免把 Adam 专用参数继续塞给 SGD
- `trainer.save_interval=999999`
  避免在第 1 个 step 结束时就保存 checkpoint

### 已验证到什么程度

服务器上已经实际验证过的是：

- 同一组数据链路
- 同一组 `SGD + training_repeat=1` 思路
- 训练能跑到 `Epoch 0: 100% | 1/1`

最后一次报错出现在 checkpoint 保存阶段，原因是磁盘空间不足。  
上面文档里的 `trainer.save_interval=999999` 是基于那个报错位置做的规避，目的是避免第 1 step 保存 checkpoint，从而更接近 clean exit。

### 通过标准

这一轮 smoke test 的通过标准建议定义为：

1. 第 3 步打印 `images_per_sample=5`
2. 第 5 步成功进入训练
3. 第 5 步至少跑到 `Epoch 0: 100% | 1/1`

如果失败只发生在 checkpoint 保存，而前面已经跑到 `1/1`，那从“触觉 5 视角训练链路是否打通”的角度，依然算功能通过。

## 最小可复现实验总结

如果你只想记住最短的一条链，记这个：

1. 生成 mock 数据

```bash
python tools/generate_mock_xr0_dataset.py \
  --output-root data/mock_xr0_base \
  --episodes 2 \
  --num-frames 48 \
  --fps 10
```

2. 扩成 5 路 JSON

```bash
python tools/add_tactile_views_to_json_dataset.py \
  --json-dir data/mock_xr0_base/json \
  --output-dir data/mock_xr0_tactile/json \
  --left-source replace_wrist_left_suffix \
  --right-source replace_wrist_right_suffix \
  --check-files
```

3. 跑 preprocess smoke test

```bash
python tools/tactile_multiview_batch_smoke.py \
  --json-dir data/mock_xr0_tactile/json \
  --batch-size 1
```

4. 生成训练配置

```bash
python tools/prepare_xr0_dataset.py \
  --dataset-root data/mock_xr0_tactile \
  --json-dir data/mock_xr0_tactile/json \
  --dataset-name tactile_smoke \
  --batch-size 1 \
  --trainer-max-steps 1 \
  --trainer-val-check-interval 1 \
  --trainer-save-interval 999999 \
  --train-config-out configs/tactile_smoke.yaml \
  --stats-out data/mock_xr0_tactile/stats_action30.json
```

5. 跑 1 step 训练

```bash
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
  --config-name tactile_smoke \
  trainer.project="xr0_smoke" \
  trainer.exp_name="tactile_5view_smoke" \
  trainer.default_root_dir="./runs/tactile_5view_smoke" \
  trainer.max_steps=1 \
  trainer.val_check_interval=1 \
  trainer.save_interval=999999 \
  model.params.model.training_repeat=1 \
  trainer.optimizer.type="torch.optim.SGD" \
  'trainer.optimizer.params.lr=1e-4' \
  'trainer.optimizer.params.weight_decay=0.1' \
  '~trainer.optimizer.params.betas' \
  '~trainer.optimizer.params.eps'
```

## 如果后面要接真实触觉视频

当你有真实数据时，训练侧最小要求仍然是：

- 每个 episode 多两路视频
  - `*_tactile_left.mp4`
  - `*_tactile_right.mp4`
- 这两路视频与 3 路 RGB 严格同步
- JSON 里新增
  - `observations.tactile_left`
  - `observations.tactile_right`
- prompt 改成 5 张图版本

如果你的命名规则就是：

- `*_wrist_left.mp4`
- `*_wrist_right.mp4`
- `*_tactile_left.mp4`
- `*_tactile_right.mp4`

那么可以直接复用：

```bash
python tools/add_tactile_views_to_json_dataset.py \
  --json-dir data/bi_piper_xr0/json \
  --output-dir data/bi_piper_xr0_tactile/json \
  --left-source replace_wrist_left_suffix \
  --right-source replace_wrist_right_suffix \
  --check-files
```

## 重要提醒

上面这些改动解决的是“训练输入”。

如果你后面还想让部署时也真的使用触觉，那还需要继续改在线推理链路，也就是：

- 运行时实时生成左/右触觉热力图
- 推理客户端把这两张图也送进 VLM
- 在线 prompt 同样改成 5 张图版本

也就是说：

- `训练能不能做`：能
- `只要多两个 mp4 能不能训练`：基本能
- `部署时是不是自动具备触觉能力`：不是，还要补在线推理输入

## 空间占用与清理

这次冒烟测试里，真正可能很大的通常不是 mock 数据，而是模型缓存和训练输出。

常见的大目录：

- `~/.cache/huggingface/hub`
  这里会缓存 `Qwen/Qwen3-VL-4B-Instruct`
- `~/xiaomi/xr0/runs/tactile_5view_smoke`
  训练输出目录，之前磁盘满时这里可能留下半截 checkpoint
- `~/xiaomi/xr0/wandb/offline-run-*`
- `~/.triton`
- `~/.cache/torch_extensions`

先查谁大：

```bash
cd ~/xiaomi/xr0

du -sh \
  data/mock_xr0_base \
  data/mock_xr0_tactile \
  runs \
  wandb \
  ~/.cache/huggingface \
  ~/.cache/torch_extensions \
  ~/.triton 2>/dev/null
```

通常可以安全清掉：

```bash
rm -rf ~/xiaomi/xr0/runs/tactile_5view_smoke
rm -rf ~/xiaomi/xr0/wandb/offline-run-*
rm -rf ~/.cache/torch_extensions
rm -rf ~/.triton
```

如果短期内不再跑这次 XR0 触觉 smoke test，也可以删：

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct*
```

如果你确定后面不再需要在 `mibot` 环境里编译 DeepSpeed / CUDA 扩展，再考虑卸掉 conda 编译器：

```bash
conda remove -y gcc_linux-64 gxx_linux-64 gcc_impl_linux-64 gxx_impl_linux-64
conda clean -a -y
```
