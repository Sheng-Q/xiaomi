# XR0 触觉工具包

这个工具包把旧版 `tactile_ws` 里和触觉传感器相关的核心能力整理成了一个不依赖 ROS 的工具模块，可以直接运行在现有的 `mibot` conda 环境中。

## 功能概览

- 通过串口和 PX-6AX 触觉控制器通信
- 在 `auto_push` 模式下解析合力和分布式触觉数据
- 在 `distributed_poll` 模式下轮询 68 点分布式触觉图
- 对三轴合力和分布式触觉图同时做零点标定
- 生成热力图和 RGB 触觉可视化图
- 按需保存 `png` 热力图和 `npz` 数据快照

## 目录结构

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

`mibot.tactile` 是一个独立的侧边工具模块，不会直接改动 XR0 当前的模型输入输出格式。你可以先把它用于标定、监控、可视化和数据采集，后续如果要把触觉真正接入模型输入，再单独处理。

## Python 依赖

如果 `mibot` 环境里还没有这些包，可以先安装：

```bash
pip install pyserial scipy opencv-python-headless
```

## 工具脚本说明

当前主要有两个可直接运行的脚本：

- `tools/tactile_monitor.py`
  终端监控脚本。主要用于看触觉数值、做快速联通测试、按需保存热力图和 `npz` 数据。

- `tools/tactile_preview_server.py`
  浏览器预览脚本。主要用于实时观察热力图，默认不保存文件。

注意：这两个脚本不要同时运行，因为它们都会占用同一个串口设备。

## 常见使用流程

下面所有命令都默认你已经进入 `xr0/` 目录，并激活了 `mibot` 环境。

### 1. 先做联通测试

这是最推荐的第一步，用来确认串口、控制器和触觉数据流都正常：

```bash
python tools/tactile_monitor.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate \
  --max-frames 20
```

这条命令会：

- 打开串口 `/dev/ttyACM0`
- 启用 `auto_push` 模式
- 在前几帧到来后自动做一次零点标定
- 在终端打印 `index_middle` 和 `middle_middle` 的标定后力值
- 读取 20 帧后自动退出

如果这一步能正常跑通，说明基本链路已经没问题了。

### 2. 只看热力图，不保存文件

如果你只是想观察触觉变化，推荐用浏览器预览脚本：

```bash
python tools/tactile_preview_server.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate
```

脚本启动后，会在本机开启一个网页服务：

```text
http://127.0.0.1:8765
```

如果脚本是跑在远程服务器上的，可以在你本地电脑做端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

然后在你本机浏览器打开 `http://127.0.0.1:8765` 即可。

### 3. 保存热力图和数据快照

如果你想把热力图图片和原始数据一起落盘，使用：

```bash
python tools/tactile_monitor.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate \
  --output-dir tactile_logs \
  --save-every 30
```

这条命令会每 30 帧保存一次：

- 一张 `png` 热力图
- 一个压缩后的 `npz` 触觉快照

## 命令行参数说明

### 两个脚本都支持的公共参数

- `--port`
  含义：触觉控制器对应的串口设备路径。
  常见取值：`/dev/ttyACM0`。
  什么时候改：如果设备实际出现在 `/dev/ttyACM1`、`/dev/ttyUSB0` 等其他路径，就改成对应设备名。

- `--baudrate`
  含义：串口波特率。
  默认值：`921600`。
  什么时候改：通常不用改，除非你的控制器固件使用了不同的波特率。

- `--timeout`
  含义：串口读取超时时间，单位是秒。
  默认值：`1.0`。
  怎么调：如果串口比较慢或者不稳定，可以适当增大；如果你希望异常更快暴露出来，可以减小。

- `--mode`
  含义：触觉数据读取模式。
  可选值：`auto_push` 或 `distributed_poll`。
  推荐用法：优先使用 `auto_push`。
  什么时候切换：如果 `auto_push` 模式下拿不到稳定数据，但你又想轮询分布式触觉图，可以试 `distributed_poll`。

- `--distributed-scale`
  含义：把原始分布式触觉字节值换算成浮点数时用的比例系数。
  默认值：`0.1`。
  怎么用：通常把它当成物理量换算比例，不建议随便改；除非你明确知道底层输出缩放关系，或者你就是想重新解释触觉强度。

- `--calibrate`
  含义：在收到初始几帧之后自动做零点标定。
  怎么用：启动脚本后，保持手指没有接触任何物体。
  推荐程度：几乎所有预览、监控、采集场景都建议打开。

- `--calibration-samples`
  含义：参与零点估计的帧数。
  默认值：`50`。
  怎么调：值越大，基线越稳；值越小，启动越快。
  推荐范围：`30` 到 `100`。

- `--calibration-interval`
  含义：每次采样之间的等待时间，单位是秒。
  默认值：`0.05`。
  怎么调：值小一点，标定结束更快；值大一点，对时间噪声更稳。

- `--calibration-warmup-frames`
  含义：正式开始统计零点之前，先丢弃多少帧预热数据。
  默认值：`20`。
  怎么调：如果你发现刚启动时前几帧波动大、标定容易漂，建议增大这个值。
  推荐范围：`10` 到 `40`。

- `--calibration-reducer`
  含义：把多帧标定样本汇总成最终零点时使用的统计方式。
  可选值：`median` 或 `mean`。
  默认值：`median`。
  怎么选：
  `median` 更抗异常值和瞬时抖动，适合你现在这种“标定不太稳定”的情况。
  `mean` 会更贴近传统平均值，但更容易被偶发尖峰影响。

- `--poll-interval`
  含义：`distributed_poll` 模式下，两次轮询之间的等待时间。
  默认值：`0.1`。
  怎么调：想提高轮询频率就减小；想减少串口压力就增大。

- `--log-level`
  含义：日志输出级别。
  常见取值：`INFO`、`WARNING`、`DEBUG`。
  怎么用：平时用 `INFO` 就够了；排查协议、时序和串口问题时可以改成 `DEBUG`。

### 两个脚本共用的可视化超参数

- `--output-size`
  含义：每个触觉子图的边长像素，最后四张图会拼成一个总图。
  默认值：`256`。
  怎么调：想截图更清晰就调大；想减轻 CPU 负担就调小。

- `--heatmap-vmin`
  含义：Fz 热力图量程下界。
  默认值：`0.0`。
  怎么理解：小于等于这个值的区域，会被映射成最暗的颜色。
  推荐用法：普通接触可视化一般保持 `0.0` 即可。

- `--heatmap-vmax`
  含义：Fz 热力图量程上界。
  默认值：`25.5`。
  怎么理解：大于等于这个值的区域，会被映射成最亮或最热的颜色。
  怎么调：
  如果轻轻一碰整张图就很亮，说明太容易饱和，应该增大这个值。
  如果明显接触了但图还是很暗，说明量程太大，应该减小这个值。

- `--heatmap-colormap`
  含义：Fz 热力图使用的颜色映射。
  可选值：`turbo`、`inferno`、`plasma`、`viridis`、`cividis`、`jet`、`hot`、`bone`。
  怎么选：
  想让颜色变化更丰富、更有冲击力，优先试 `turbo` 或 `inferno`。
  想看起来更平滑、更像科研图，试 `viridis` 或 `cividis`。
  想偏黑白/热成像风格，可以试 `bone` 或 `hot`。

- `--heatmap-gamma`
  含义：热力图归一化之后再做一次非线性增强的 gamma 参数。
  默认值：`0.75`。
  怎么理解：
  小于 `1.0` 时，弱接触区域会更容易显现出来。
  大于 `1.0` 时，低幅值会被压暗，更强调强接触区域。
  推荐起始范围：`0.4` 到 `1.0`。

- `--rgb-vmax-fz`
  含义：RGB 触觉图里 Fz 通道的量程上界。
  默认值：`25.5`。
  怎么调：如果没有特殊需求，建议和 `--heatmap-vmax` 保持一致，这样热力图和 RGB 图对同样受力的响应更一致。

- `--rgb-vmax-shear`
  含义：RGB 图里 Fx/Fy 横向剪切力通道的量程上界。
  默认值：`12.8`。
  怎么调：
  如果红绿变化太弱，看不出侧向摩擦，可以减小。
  如果红绿很容易饱和、噪声太明显，可以增大。

### `tactile_monitor.py` 独有参数

- `--output-dir`
  含义：保存 `png` 和 `npz` 文件的目录。
  怎么用：例如填 `tactile_logs`，就会在当前目录下创建这个文件夹并保存结果。

- `--save-every`
  含义：每收到多少帧保存一次。
  默认值：`0`，表示不保存。
  怎么调：值越小保存越密，数据也会更多。

- `--max-frames`
  含义：最多读取多少帧后自动退出。
  默认值：`0`，表示一直运行。
  怎么用：做冒烟测试、连通性测试、标定检查时非常有用。

### `tactile_preview_server.py` 独有参数

- `--host`
  含义：预览网页服务绑定的地址。
  默认值：`127.0.0.1`。
  怎么用：一般保持默认即可；只有你明确要让局域网其他机器直接访问时，才考虑改成 `0.0.0.0` 等地址。

- `--http-port`
  含义：预览网页服务使用的端口号。
  默认值：`8765`。
  怎么用：如果这个端口被占用了，就换成别的。

- `--refresh-ms`
  含义：浏览器页面刷新热力图的间隔，单位毫秒。
  默认值：`250`。
  怎么调：调小会更流畅，但浏览器和服务器负担会更大；调大会更省资源。

## 可视化调参建议

你可以从下面几组参数开始试：

- 弱接触增强型
  `--heatmap-vmax 8.0 --heatmap-gamma 0.5 --heatmap-colormap turbo`

- 大动态范围型
  `--heatmap-vmax 20.0 --heatmap-gamma 0.8 --heatmap-colormap inferno`

- 平滑稳健型
  `--heatmap-vmax 15.0 --heatmap-gamma 0.9 --heatmap-colormap viridis`

实际调参时可以遵循这几条经验：

- 如果轻微接触就整片发亮，增大 `--heatmap-vmax`
- 如果按得比较明显了图还是偏黑，减小 `--heatmap-vmax`
- 如果想让弱接触、边缘接触更明显，减小 `--heatmap-gamma`
- 如果 RGB 图里蓝色太强、红绿变化太不明显，减小 `--rgb-vmax-shear`
- 如果 RGB 图里红绿噪声太重、太容易跳动，增大 `--rgb-vmax-shear`

## 标定说明

当前还没有单独的“只做标定”的独立脚本，但两个运行脚本都内置了标定能力，只要加上 `--calibrate` 就会在启动后自动标定。

标准标定流程如下：

1. 确保触觉手指没有接触任何物体
2. 运行 `tactile_monitor.py` 或 `tactile_preview_server.py`，并带上 `--calibrate`
3. 在启动后的前几秒内保持传感器完全空载

建议你检查标定效果时重点看这些现象：

- 如果静止不接触时热力图仍然长期有亮斑，重新运行一次并带上 `--calibrate`
- 如果静止时基线总在慢慢漂，适当增大 `--calibration-samples`
- 如果刚启动时前几帧波动很大，增大 `--calibration-warmup-frames`
- 如果偶发尖峰会把标定带偏，优先使用 `--calibration-reducer median`
- 如果你觉得启动太慢，可以减小 `--calibration-samples` 或 `--calibration-interval`

一个比较稳妥的标定示例：

```bash
python tools/tactile_monitor.py \
  --port /dev/ttyACM0 \
  --mode auto_push \
  --calibrate \
  --calibration-warmup-frames 30 \
  --calibration-samples 80 \
  --calibration-interval 0.03 \
  --calibration-reducer median \
  --max-frames 30
```

现在这套标定会同时作用于：

- 终端里显示的 `Fx/Fy/Fz` 数值
- 热力图和 RGB 图使用的分布式触觉图
- 保存出来的 `npz` 数据，其中会包含标定后的分布式触觉数据和对应的零偏

## Python 调用方式

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

## 集成建议

当前建议先把这套触觉工具用于：

- 传感器联通测试
- 零点标定
- 运行时观察和调试
- 热力图和触觉数据采集
- 接触检测、规则判断等外围逻辑

暂时不建议直接去改 `mibot.utils.io.compose_state()` 或当前 XR0 模型输入维度。  
如果以后确定要把触觉接进模型，再把它作为一次单独的数据格式和训练流程变更来处理会更稳妥。
