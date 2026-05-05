# Piper Smoke Test

This note is for the case where:

- the inference server runs on another machine,
- this Piper industrial PC runs the robot-side client,
- you do **not** have Piper training data yet,
- and you do **not** have a post-trained XR0 checkpoint yet.

In that situation, the smallest useful smoke test is **not** to move the robot. The right goal is:

1. read Piper joint and gripper state from CAN,
2. read top and wrist cameras,
3. package them into the XR0 real-robot client format,
4. send them to a remote server over the official XR0 TCP protocol,
5. receive a `(30, 32)` action chunk back,
6. verify latency and field mapping.

## Which server/client pair matters

This repository contains two different deploy paths:

- `deploy/server.py` + `deploy/client.py`: simulation and benchmark checkpoints.
- `mibot/server/deploy.py` + `mibot/server/runtime/client.py`: post-training and real-robot runtime under `xr0/`.

For a Piper robot-side client, the relevant path is the second one under `xr0/`.

## Why a mock server is the right first step

`mibot/server/deploy.py` needs a real post-trained model directory with:

- `config.py`
- `last.ckpt/checkpoint/mp_rank_00_model_states.pt`
- normalization stats in the config

If you do not have that checkpoint yet, start with `tools/mock_runtime_server.py`.
It speaks the same length-prefixed pickle TCP protocol as the XR0 runtime server, but returns a dummy `(30, 32)` action chunk.

## Existing local client code that this adapter reuses

On this machine, the most relevant existing Piper client is in:

- `/home/whz/teleop_evo/src/lerobot/robots/piper_follower/piper_follower.py`

That code already does the parts you need on the robot side:

- CAN connection to Piper through `piper_sdk`
- joint and gripper reads
- RealSense camera reads

The adapter in `tools/piper_smoke_client.py` reuses that code and only adds the XR0-specific observation packing.

## Minimal remote-server smoke test

On the remote server machine:

```bash
cd /path/to/Xiaomi-Robotics-0/xr0
PYTHONPATH=$PWD python tools/mock_runtime_server.py   --host 0.0.0.0   --port 10086   --sleep-ms 50
```

This verifies:

- TCP connectivity
- request serialization
- response deserialization
- client loop timing

## Minimal Piper-side smoke test

On the Piper industrial PC:

```bash
PYTHONPATH=/home/whz/teleop_evo/src:/home/whz/桌面/Xiaomi-Robotics-0/xr0 python /home/whz/桌面/Xiaomi-Robotics-0/xr0/tools/piper_smoke_client.py   --host <REMOTE_SERVER_IP>   --port 10086   --processor Qwen/Qwen3-VL-4B-Instruct   --can can1   --top-camera 420222071960   --wrist-camera 419622072329   --task "smoke test"   --active-side left   --iterations 5
```

Notes:

- The camera serials above come from the existing local launcher in `桌面/lerobot远程推理测试集合/测试0408`.
- `--processor` can also be a **local** directory if the industrial PC is offline and already has the Qwen3-VL processor files copied over.
- This smoke client is intentionally **observe-only**. It does not execute returned actions on the robot.

## What this adapter changes compared with the existing local client

The local `teleop_evo` Piper client is single-arm and exposes:

- `joint_1.pos ... joint_6.pos`
- `gripper.pos`
- `ego` image
- `wrist` image

The XR0 runtime client expects a bimanual schema:

- `left_arm_joint`, `left_gripper_pos`
- `right_arm_joint`, `right_gripper_pos`
- `ego_obs`, `left_wrist_obs`, `right_wrist_obs`

So the adapter does three things:

1. maps the real arm to either the XR0 `left` side or `right` side with `--active-side`,
2. fills the missing arm state with zeros or a copy of the active arm,
3. fills the missing wrist image with either black, `ego`, or the active wrist image.

This is acceptable for a smoke test, but it is **not** a valid substitute for a true dual-arm XR0 deployment.

## Small runtime changes included for easier adaptation

Two small changes make this workflow easier:

1. `mibot/server/runtime/client.py` now accepts a configurable `processor_name_or_path`, so the client can use a local Qwen processor snapshot.
2. `mibot/utils/io.py::recover_action` now works even if the robot-side state only provides joints and grippers, without end-effector pose.

That second point matters because the existing `teleop_evo` Piper follower already exposes joint and gripper state, but not the full XR0 end-effector fields by default.

## What to do once you have a real checkpoint

Replace the mock server with the real XR0 runtime server:

```bash
cd /path/to/Xiaomi-Robotics-0/xr0
PYTHONPATH=$PWD python mibot/server/deploy.py   --model /path/to/posttrained_model_dir   --host 0.0.0.0   --port 10086
```

At that stage, the next step is to add a **guarded execution layer** on the Piper side:

- clamp large deltas,
- decide whether to trust joint-space targets or end-effector targets,
- optionally enable SDK FK with `piper_sdk.EnableFkCal()` if you want to reconstruct end-effector state too,
- and only then allow sending commands to the physical robot.

Until that checkpoint exists, transport-only smoke testing is the correct minimum.
