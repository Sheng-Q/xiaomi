# XR0 Tactile Toolkit

This package ports the useful parts of the legacy `tactile_ws` ROS driver into a ROS-free toolkit that can run inside the existing `mibot` conda environment.

## What it covers

- serial communication with the PX-6AX tactile controller
- force + distributed tactile parsing in auto-push mode
- distributed-only polling mode for 68-point reads
- zero-point calibration by averaging recent force samples
- heatmap and RGB tactile visualization
- PNG / NPZ snapshot saving for debugging and data collection

## Package layout

```text
xr0/
|-- mibot/
|   `-- tactile/
|       |-- __init__.py
|       |-- types.py
|       |-- protocol.py
|       |-- driver.py
|       |-- runtime.py
|       `-- visualizer.py
`-- tools/
    |-- tactile_monitor.py
    `-- tactile_preview_server.py
```

`mibot.tactile` is intentionally independent from XR0 model code. It can be used as a sidecar utility in robot control scripts without changing XR0's current 32-D state / action layout.

## Required Python packages

Install these into the `mibot` environment if they are not already present:

```bash
pip install pyserial scipy opencv-python-headless
```

## Example usage

From the `xr0/` directory:

```bash
python tools/tactile_monitor.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate \
  --output-dir tactile_logs \
  --save-every 30
```

This will:

- open the tactile serial port
- enable auto-push mode on the controller
- wait for frames and run zero-point calibration
- print calibrated force values for `index_middle` and `middle_middle`
- save a PNG heatmap and an NPZ snapshot every 30 frames

## Live browser preview

If you only want to observe the heatmap without saving files, run:

```bash
python tools/tactile_preview_server.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate
```

Then open:

```text
http://127.0.0.1:8765
```

If the script runs on a remote server, you can forward the port to your local machine:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

Then open `http://127.0.0.1:8765` in your local browser.

## Direct Python API

```python
from mibot.tactile import TactileRuntime, TactileSensorDriver, TactileVisualizer

driver = TactileSensorDriver(port="/dev/ttyACM0")
runtime = TactileRuntime(driver)
visualizer = TactileVisualizer()

runtime.start()
runtime.calibrate()
snapshot = runtime.get_snapshot()
image = visualizer.render_snapshot(snapshot)
runtime.stop()
```

## Integration guidance

This first step is deliberately kept outside XR0's model inputs:

- use it for calibration, logging, contact heuristics, or operator feedback first
- do not change `mibot.utils.io.compose_state()` yet
- if tactile data should become a model input later, treat that as a separate training / data-format change

That separation keeps the current XR0 runtime stable while giving the robot stack access to tactile sensing immediately.
