# PCB AR 标注

显微镜正上方摄像头的 PCB 元件 AR 标注。当前仓库是**可运行壳 + 接口桩**，用来对齐 `docs/TECH_SPEC.md` 和 `docs/DATA_MODEL.md`。

**本轮不做打包（no packaging in this slice）。** 没有 PyInstaller，没有安装包，也没有 `ultralytics` / torch。

## 环境

Python 3.10+。核心依赖只有 `opencv-python`、`PyQt6`、`numpy`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

只跑程序、不跑测试时，把上一行换成 `pip install -r requirements.txt`。也可以 `pip install -e ".[dev]"`。

## 启动

在仓库根目录：

```bash
PYTHONPATH=src python -m app.main
```

等价入口：

```bash
cd src && python -m app.main
```

默认视频源是摄像头 `0`。没有摄像头时会发一条错误并改画合成 PCB 帧，窗口仍然开着。强制合成帧：

```bash
PYTHONPATH=src python -m app.main --source synthetic
```

窗口标题是「PCB AR 标注」。画面是底层视频 pixmap，上面是空的标注层，右边是「元件参数」占位栏。侧栏只标未核对红 `#E53935` 和已核对绿 `#43A047`。

## 已确认（v0.1.1）

写进桩和文档，跟踪界面还没做：

- D1：矩阵是全局像素 → 当前帧像素。
- D2：允许纯像素工程；`absolute_pos_mm` / `pixel_per_mm` 可空，空着时侧栏警告。
- D3：主路径 LocalGlobal，仿射是降级。
- D4：跟踪丢失 TTL 0.8s，之后应冻结叠加并提示。常量在 `vision.policy`，计时器未接入。
- D5：`doubt`（`#FDD835`）留在枚举里，本轮界面不提供。
- D9 / D12：顶层 `status` 才是核对状态；气泡键就是 `name`、`value`、`package`。

D6–D8、D10–D11、D13–D14 已在 v0.1.2 确认：默认相机 0 / 近 4K / 无相机走合成帧，样例 zip 进 `assets/samples/`，路径相对 `assets/projects/`，运行时不用 pydantic。清单见 `docs/MILESTONES.md`。

需要图形界面（`DISPLAY` 或 Wayland）。没有显示器时：

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m app.main --smoke
```

`--smoke` 收到第一帧后退出。成功时标准输出有一行 `smoke ok:`。

## 拼接对照

样例板 `SAMPLE-PARTIAL-001`：15 张 `20261109-*.jpg` 是局部瓦片，`微信图片_20261109184831_115_34.jpg` 是全局整板对照（按文件名，不按尺寸）。

```bash
PYTHONPATH=src python -m vision.stitch_compare \
  --zip /path/to/局部图片.zip \
  --board-id SAMPLE-PARTIAL-001 \
  --out artifacts/stitch_compare
```

已经落在 `assets/samples/SAMPLE-PARTIAL-001/` 时：

```bash
PYTHONPATH=src python -m vision.stitch_compare \
  --board-dir assets/samples/SAMPLE-PARTIAL-001 \
  --board-id SAMPLE-PARTIAL-001 \
  --out artifacts/stitch_compare
```

OpenCV Stitcher 只在加 `--try-stitcher` 时试一次，失败会写进说明，不作为唯一结果。对照分数不够时 `ok` 为 false。

## 测试

```bash
PYTHONPATH=src python -m pytest
```

计划与用例编号在 `docs/TEST_PLAN.md`。

## 本轮有什么

- `VideoCaptureThread`：`QThread`，信号 `frame_ready(object, float)`、`error(str)`、`stopped()`，`stop(wait_ms)` 可重复调用。`source` 可以是摄像头索引、视频路径，或 `"synthetic"`。
- `PCBAffineTracker` / `AffineResult`、`LocalGlobalTracker` / `HomographyResult`。`M` 是 2×3，`H` 是 3×3，方向都是**全局像素 → 当前帧像素**。LocalGlobal 是主路径，仿射是降级。`track()` 返回 `ok=False` 和单位矩阵。`project_boxes` 只做给定 `H` 的投影。
- `BoardProject` / `Component` / `Fiducial` / `ProjectStore`。权威状态在元件的顶层 `status`。气泡键固定为 `name` / `value` / `package`。样例工程：`assets/projects/DEMO-PCB-001.json`。
- 主窗口壳。不在画布上画框，也不编辑参数。

## 本轮没有什么

- 真实跟踪、拼接、检测权重
- 右侧栏表单和气泡编辑
- XYZ 机台标定（`src/calib/xyz.py` 调用会抛 `NotImplementedError`）
- 打包与可选检测依赖
- 样例全局图 PNG（JSON 里的 `global_map_path` 指向尚未提供的文件）

## 目录

```
src/app/main.py                 启动
src/app/ui/main_window.py       主窗口
src/app/ui/canvas.py            视频层 + 空标注层
src/app/ui/annotation_items.py  框 / 线 / 气泡类（未放上场景）
src/app/ui/sidebar.py           右侧占位
src/vision/capture.py           采集线程
src/vision/policy.py            D1/D3/D4 常量（TTL 0.8s）
src/vision/affine_tracker.py    仿射降级桩
src/vision/local_global.py      主路径单应桩与 project_boxes
src/model/schema.py             JSON 结构与 STATUS_COLORS
src/model/store.py              读写与可见性过滤
src/calib/xyz.py                标定占位
assets/projects/DEMO-PCB-001.json
docs/TECH_SPEC.md
docs/DATA_MODEL.md
docs/MILESTONES.md
docs/TEST_PLAN.md
tests/
```

规格原文也在仓库根的 `PROJECT_BRIEF.md` 和 `HANDOFF.md`。
