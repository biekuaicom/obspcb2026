# 测试计划 — PCB AR 标注（视觉脚手架）

| 字段 | 值 |
|------|-----|
| 版本 | v0.1 |
| 日期 | 2026-11-09 |
| 对齐 | `docs/TECH_SPEC.md` v0.1.1、`docs/DATA_MODEL.md` v0.1.1、`docs/MILESTONES.md` v0.1.1 |
| 本轮 | 可运行壳、采集线程、Tracker 桩、JSON 冒烟 |

本轮不测真实 Fiducial / ORB 跟踪、拼接、YOLO、打包、标定真机。配准成功只锁定「调用方拿到 `ok=True` 矩阵之后」的投影方向；`track()` 本身固定失败并返回单位矩阵。

已确认并写进桩：D1 全局像素→当前帧；D2 纯像素可缺毫米（界面要警告）；D3 LocalGlobal 主路径、Affine 降级；D4 丢失 TTL **0.8s** 后冻结并提示（只存常量，计时器未接）；D5 `doubt` 留枚举、不做 MVP 界面；D9 顶层 `status`；D12 气泡键 `name` / `value` / `package`。D6–D8、D10–D11、D13–D14 仍开放。

## 如何跑

仓库根目录，已安装 `requirements-dev.txt`：

```bash
PYTHONPATH=src python -m pytest
```

无显示器时 Qt 用例依赖 `tests/conftest.py` 里的 `QT_QPA_PLATFORM=offscreen`。

手工看窗口：

```bash
PYTHONPATH=src python -m app.main --source synthetic
```

无显示器时只做退出码检查：

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m app.main --smoke
```

成功时打印 `smoke ok: <宽>x<高> mode=synthetic`，进程退出码 0。

## 用例

### 配准

| ID | 场景 | 期望 | 本轮自动化 |
|----|------|------|------------|
| TC-REG-01 | 配准成功 | `AffineResult.ok` / `HomographyResult.ok` 为真；`M` 为 2×3、`H` 为 3×3；方向是**全局像素 → 当前帧像素**（与 `estimateAffinePartial2D(src_global, dst_frame)`、`perspectiveTransform` 一致）。用该矩阵投影 `base_pixel_box` 四角后，平移量落到当前帧。 | **部分**。`test_success_matrix_projects_global_box_into_frame` 注入一个成功矩阵并检查投影。`PCBAffineTracker.track` / `LocalGlobalTracker.track` **不会**返回成功，真实匹配未实现。 |
| TC-REG-02 | 配准失败 | 点数不足、空帧、或本轮桩：`ok=False`，置信度 0，矩阵为单位仿射/单位单应，`message` 非空。失败时不得假装已经跟上视口。模板与全局图只被保存/记下尺寸，不算匹配成功。 | **是**。`test_track_fail_closed_identity`、`test_stub_trackers_import_and_fail_closed`。 |
| TC-REG-03 | 跟踪丢失降级（TTL、冻结上一矩阵、切另一条 Tracker） | D3/D4：先 LocalGlobal，失败再 Affine，再沿用上一矩阵 **0.8s**，超时冻结并提示「跟踪丢失」。 | **常量**。`test_policy_constants_match_confirmed_decisions` 锁 TTL 和主/降级路径。计时器与冻结叠加未接入。 |

### 投影与可见性

| ID | 场景 | 期望 | 本轮自动化 |
|----|------|------|------------|
| TC-VIS-01 | 中心投影后落在视口多边形内 | `ProjectStore.list_visible` 保留该元件。边上的点算可见。 | **是**。`test_list_visible_filters_projected_centers`、`test_list_visible_includes_center_on_the_edge`。 |
| TC-VIS-02 | 中心落在视口外 | 不返回该元件。单位矩阵下，样例框中心在视口外即隐藏。 | **是**。同上，`OUT` 在单位矩阵下被滤掉。 |
| TC-VIS-03 | 平移单应后再滤 | 同一视口里，原先在外的元件可以被平移进来，原先在内的可以被平移出去。2×3 仿射按补行 `[0,0,1]` 后与 3×3 同一方向。 | **是**。 |
| TC-VIS-04 | 四角投影 | `LocalGlobalTracker.project_boxes` 返回每框 4×2 角点，顺序左上、右上、右下、左下。`H` 不是 3×3 则 `ValueError`。 | **是**。 |
| TC-VIS-05 | 无矩阵或视口为空 | 不画。`transform is None` 或多边形少于 3 点时返回 `[]`。 | **是**。 |
| TC-VIS-06 | 只更新可见 GraphicsItem、不每帧重建 | TECH_SPEC §3.5 / §4。 | **否**。本轮标注层保持为空，`set_annotations` 只记下 view model。 |

判定点是 `base_pixel_box` 的中心 `(x+w/2, y+h/2)`，不是透视四边形的重心。

### JSON

| ID | 场景 | 期望 | 本轮自动化 |
|----|------|------|------------|
| TC-JSON-01 | 样例工程往返 | `assets/projects/DEMO-PCB-001.json` 与 DATA_MODEL §6.1 一致。`BoardProject.from_dict` → `to_dict` 深相等。顶层 `status=unchecked`，`core_params` 恰好 `name` / `value` / `package`，不含 `status`。 | **是**。`test_sample_project_json_roundtrip`。 |
| TC-JSON-02 | 旧简报形状 | 仅有 `core_params.status` 时，加载提升到顶层并从 core 删除。缺的 `name` / `value` / `package` 补成空字符串。严格的 `validate_project` 在提升前仍报告缺顶层 status。 | **是**。`test_legacy_core_status_is_promoted`。 |
| TC-JSON-03 | 写盘 | `ProjectStore.update_component` 合并 core、改顶层 status；`save` / `load` 后仍在。扩展字段保留。 | **是**。`test_store_save_load_and_patch`。 |
| TC-JSON-04 | 拒绝坏文档 | `core_params` 出现 `name`/`value`/`package` 以外的键、宽或高 ≤ 0、路径含 `..`、未知 status 在严格校验里报错。加载未知 status 时降为 `unchecked` 并打日志。无毫米字段的纯像素工程可以加载。 | **是**。`test_pure_pixel_board_uses_exact_bubble_keys`。 |
| TC-JSON-05 | 颜色 | `unchecked #E53935`、`checked #43A047` 进侧栏。`doubt #FDD835` 仍在 `STATUS_COLORS`，MVP 不画存疑控件（D5）。气泡只显示 name / value / package。 | **是**。`test_status_colors_match_data_model`、`test_bubble_shows_at_most_three_core_lines`。 |
| TC-JSON-06 | 全局图文件必须存在 | DATA_MODEL §6.2。 | **否**。样例 PNG 未随本轮提供，路径根目录仍待确认。校验只要求相对路径且不越出 `..`。 |

### 采集与窗口

| ID | 场景 | 期望 | 本轮自动化 |
|----|------|------|------------|
| TC-CAP-01 | `source="synthetic"` | `frame_ready(ndarray, timestamp_s)`，BGR，尺寸与构造参数一致，时间戳为 `float`。`stop()` 可重复调用，线程退出并发 `stopped`。 | **是**。 |
| TC-CAP-02 | 无摄像头 | 索引对应设备不存在时发 `error` 并改合成帧，UI 仍有画面。`set_source` 在下一轮循环生效；在 `start` 之前设成 `synthetic` 则不会去打开摄像头。 | **是**（Linux 上跳过不存在的 `/dev/videoN`）。 |
| TC-CAP-03 | 主窗口壳 | 标题含「PCB AR 标注」。底层 pixmap 有帧，标注层子项为 0，右侧 Dock 标题「元件参数」。样例工程 `DEMO-PCB-001` 被载入。 | **是**。`test_main_window_shell`（offscreen）。 |
| TC-CAP-04 | 真摄像头 / 视频文件循环 | 打开设备或文件，文件结束后续读。 | **手工**。本环境没有 `/dev/video*`。 |
| TC-CAP-05 | 冷启动与干净退出 | TECH_SPEC §6.1 的 1、2、4。第 3 步「显示全局图」本轮不做。 | 开发机：`--smoke`。打包后的同一清单留给构建发布，本轮不打包。 |

## 明确不在本轮

- YOLO / `ultralytics` / torch
- 瓦片拼接与 OpenCV Stitcher
- Fiducial ROI 搜索、ORB、RANSAC 的真实实现
- 画布内联编辑、侧栏表单、红绿框叠加、选中同步
- `src/calib/xyz.py` 的机台公式（调用即 `NotImplementedError`）
- PyInstaller 与安装包

## 以后补测时的入口

- 成功配准：对已知平移的合成帧，`track` 的矩阵作用在全局角点上，误差在 ransac 阈值内，且 `ok` 为真。
- 失败配准：纹理空白帧 `ok` 为假，UI 不把单位矩阵当成「已跟上」除非产品把身份矩阵定义为锁定。
- 可见性：4K 帧上只给视口内的元件创建 Item；视口外的 id 不出现在场景里。
- 序列化：改 `checked` 后重开进程，颜色仍按 `#43A047`。
