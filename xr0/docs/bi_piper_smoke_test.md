# Bimanual Piper Smoke Test

This note extends the single-arm smoke test to a **dual-arm Piper follower** setup.

The goal is still observe-only transport verification:

1. read both Piper arms over CAN,
2. read one ego camera plus left/right wrist cameras,
3. pack state and images into the XR0 runtime client format,
4. send the request to the remote XR0 runtime server,
5. receive a `(30, 32)` action chunk back,
6. verify that the dual-arm state mapping is correct.

It does **not** execute returned actions on the real robot.

## XR0 32-D Action Layout

XR0 uses a fixed 32-dimensional action/state layout. The slots are defined in `mibot/utils/io.py`.

| Dimensions | Component | Meaning |
|------------|-----------|---------|
| 0--2 | `left_ee_pos` | Left end-effector position delta |
| 3--5 | `left_ee_aa` | Left end-effector rotation delta (axis-angle) |
| 6 | `left_gripper` | Left gripper delta |
| 7--12 | `left_joint` | Left 6 joint deltas |
| 13 | reserved | Always 0 |
| 14--16 | `right_ee_pos` | Right end-effector position delta |
| 17--19 | `right_ee_aa` | Right end-effector rotation delta (axis-angle) |
| 20 | `right_gripper` | Right gripper delta |
| 21--26 | `right_joint` | Right 6 joint deltas |
| 27--31 | reserved | Always 0 |

## How Bimanual Piper Maps Into XR0

A dual-arm Piper follower gives you 14 directly usable proprio dimensions:

- left arm: 6 joints + 1 gripper
- right arm: 6 joints + 1 gripper

Those 14 values map into XR0 like this:

| Piper signal | XR0 state slot |
|--------------|----------------|
| `left_gripper.pos` | dim `6` |
| `left_joint_1.pos ... left_joint_6.pos` | dims `7..12` |
| `right_gripper.pos` | dim `20` |
| `right_joint_1.pos ... right_joint_6.pos` | dims `21..26` |

Everything else in the `(1, 32)` runtime state is left as zero:

- dims `0..5`: no Piper end-effector state is provided here,
- dim `13`: reserved,
- dims `14..19`: no right-arm end-effector state is provided here,
- dims `27..31`: reserved.

That means the XR0 server still returns a full `(30, 32)` action chunk, but for Piper smoke testing you normally care about these slices:

- left gripper delta: dim `6`
- left joint deltas: dims `7..12`
- right gripper delta: dim `20`
- right joint deltas: dims `21..26`

If you later want to **execute** XR0 outputs on Piper, the safest first integration is:

1. ignore the end-effector slices (`0..5`, `14..19`),
2. use only joint/gripper deltas,
3. reconstruct absolute joint/gripper targets by adding deltas to the current Piper state,
4. clamp them before sending commands.

## Camera Wiring Assumption

The XR0 runtime client needs exactly three images:

- `ego_obs`
- `left_wrist_obs`
- `right_wrist_obs`

The bimanual smoke client therefore assumes:

- one shared top/ego camera,
- one optional left wrist camera,
- one right wrist camera.

To avoid opening the same top camera twice, the script attaches the ego camera to only one Piper arm config via `--ego-camera-side left|right`.

If the left wrist camera is temporarily removed, you can simply omit `--left-wrist-camera`. The client will synthesize
`left_wrist_obs` using `--missing-left-wrist-fill`, which defaults to a black masked image.

## Bimanual Smoke Client

Use:

- `tools/bi_piper_smoke_client.py`

It reuses `teleop_evo`'s existing `BiPiperFollower` implementation and converts its observation keys:

- `left_joint_1.pos ... left_joint_6.pos`
- `left_gripper.pos`
- `right_joint_1.pos ... right_joint_6.pos`
- `right_gripper.pos`
- `left_ego` or `right_ego` for the shared top camera
- `left_wrist`
- `right_wrist`

into the XR0 runtime client request format.

If `left_wrist` is unavailable because the hardware camera is removed, the client can instead send one of:

- a black image (`black`, default)
- a copy of the ego image (`copy-ego`)
- a copy of the right wrist image (`copy-right-wrist`)

## Example Command

With all three cameras present:

```bash
PYTHONPATH=/home/whz/teleop_evo/src:/home/whz/桌面/Xiaomi-Robotics-0/xr0 \
python /home/whz/桌面/Xiaomi-Robotics-0/xr0/tools/bi_piper_smoke_client.py \
  --host <REMOTE_SERVER_IP> \
  --port 10096 \
  --processor Qwen/Qwen3-VL-4B-Instruct \
  --left-can <LEFT_CAN> \
  --right-can <RIGHT_CAN> \
  --top-camera <TOP_CAMERA_SERIAL> \
  --left-wrist-camera <LEFT_WRIST_SERIAL> \
  --right-wrist-camera <RIGHT_WRIST_SERIAL> \
  --ego-camera-side left \
  --task "bimanual smoke test" \
  --iterations 5
```

With the left wrist camera temporarily removed:

```bash
PYTHONPATH=/home/whz/teleop_evo/src:/home/whz/桌面/Xiaomi-Robotics-0/xr0 \
python /home/whz/桌面/Xiaomi-Robotics-0/xr0/tools/bi_piper_smoke_client.py \
  --host <REMOTE_SERVER_IP> \
  --port 10096 \
  --processor Qwen/Qwen3-VL-4B-Instruct \
  --left-can <LEFT_CAN> \
  --right-can <RIGHT_CAN> \
  --top-camera <TOP_CAMERA_SERIAL> \
  --right-wrist-camera <RIGHT_WRIST_SERIAL> \
  --missing-left-wrist-fill black \
  --ego-camera-side left \
  --task "bimanual smoke test" \
  --iterations 5
```

## Expected Output

Each iteration prints:

- round-trip latency,
- raw action chunk shape,
- left/right joint deltas,
- left/right gripper deltas,
- left/right reconstructed joint targets,
- left/right reconstructed gripper targets.

If the smoke test is healthy, you should see:

- successful connection to both CAN interfaces,
- successful reads from all three cameras,
- a valid `(30, 32)` action chunk from the server,
- no import / processor / proxy / socket errors.
