# PCB 显微镜摄像头 AR 标注系统 — 项目简报（PROJECT BRIEF）

> 版本：v0.1（绿场立项）  
> 日期：2026-11-09（Asia/Shanghai）  
> 目标平台：先桌面 PC，后手机  
> 技术栈：Python + OpenCV + PyQt6（QGraphicsView 分层渲染）  
> 状态：全新项目，无历史数据对接

---

> **规格覆盖（2026-11-09 Asia/Shanghai）**：`docs/TECH_SPEC.md` **v0.2.0 / D16** 将 **Live 主路径** 改为 **PCB 结构化几何拓扑匹配**（灰度+自适应二值 → 轮廓/孔/焊盘 → Geometric Constellation / point-set 局部匹配 → 锁定位姿叠加气泡）。高清 mosaic / Mapping Mode / 散图 stitch **均为可选建图或对照旁路，非 Live 必经**。工程库以轻量 JSON/拓扑（`topology_db`）为主。本文 §1.1「拼接全局高清大图」与 §3 配准表述以 TECH_SPEC / DATA_MODEL / MILESTONES **v0.2.0** 为准。

---

## 1. 项目目标

打造一套**显微镜正上方摄像头**驱动的 PCB 元件识别与 AR 风格实时标注系统，实现：

1. **自动识别**电路板上的元件（电阻、电容、IC、连接器等），生成初始标注框与核心参数。
2. **人工核对与修改**：画布内联气泡（最多 3 个核心参数）+ 右侧栏完整扩展参数。
3. **类 AR 实时跟随**：摄像头/显微镜视口平移、缩放、微小旋转时，标注框、引导条线与参数气泡“粘”在电路板物理位置上同步移动。
4. **前期只做 PC 端**；验证稳定后再移植手机端，并平滑迁移到自动 XYZ 机台（编码器坐标映射）。

### 1.1 成功标准（MVP）

- 能半自动/离线拼接显微镜局部帧为 PCB 全局高清大图。
- 在全局图上完成自动检测种子标注 + 人工核对落库（JSON/SQLite）。
- 实时视频流中，通过局部↔全局配准，将视口内标注投影到当前帧并叠加显示。
- 画布点击编辑 ≤3 个核心参数；右侧栏编辑扩展字段；未核对红色 / 已核对绿色。
- 800W 像素相机下跟踪帧率可接受（目标 ≥15 FPS，争取 30 FPS）。

---

## 2. 约束与假设

| 类别 | 说明 |
|------|------|
| 硬件姿态 | 摄像头在电路板正上方；主要运动为 X/Y/Z 平移；手动显微镜阶段可能有微小旋转 θ |
| 不需要 | 复杂 6DoF SLAM / PnP 位姿解算（前期）；完整透视畸变建模可后置 |
| 相机 | 约 800 万像素（常见 4K，如 3840×2160） |
| 定位线索 | 定位孔（Fiducial）、焊盘、固定连接器/接线座、大尺寸固定元件 |
| 数据 | 绿场项目，无遗留 BOM/标注库；字段可自定义扩展 |
| 交互 | 画布气泡最多 3 个核心参数，避免遮挡邻元件；完整参数走右侧栏 |
| 后期扩展 | 自动 XYZ 机台：用编码器物理坐标 + 标定矩阵替代纯视觉配准 |

---

## 3. 系统架构

采用 **MVC** 思想，视觉处理在 Worker 线程，UI 在主线程，元件数据双向绑定。

```
[显微镜相机 USB/HDMI]
        │
        ▼
[VideoCaptureThread (OpenCV)] ──► 帧信号
        │                              │
        │                              ▼
        │                    [配准模块]
        │                    · Fiducial 模板匹配 + estimateAffinePartial2D
        │                    · LocalGlobalTracker: ORB + RANSAC Homography
        │                              │
        ▼                              ▼
[QGraphicsScene]
  ├─ 底层: QGraphicsPixmapItem（实时视频）
  ├─ 中层: QGraphicsRectItem / LineItem（标注框与引导线）
  └─ 顶层: QGraphicsTextItem（≤3 参数气泡）
        │
        ▼
[PyQt6 MainWindow]
  ├─ 中央: PCBCanvasView（点击拾取、内联编辑）
  └─ 右侧: QDockWidget（完整参数表单 / 批量核对）
        ▲
        │ 双向绑定
[Model: 元件库 JSON / SQLite]
  · absolute_pos_mm（PCB 物理绝对坐标）
  · base_pixel_box（全局图像素框）
  · core_params / extended_params / status
```

### 3.1 两阶段工作流

**阶段 A — 全局建图（主路径以 TECH_SPEC v0.1.3 为准）**

1. **主路径（Mapping Mode）**：显微镜弓字形匀速扫板，帧间重叠约 80%–90%；帧间 feature/光流增量拼成 **全局 mosaic**（见 TECH_SPEC `VideoMosaicBuilder`）。
2. **兜底**：散图瓦片 / 外部工具（Hugin / PTGui）导入 → 同一 Global Map；用于对照或无扫描条件。
3. 在全局 mosaic 上跑 YOLO 自动检测种子 + 人工框选/核对 → 写入全局标注库。

**阶段 B — 在线实时跟随**

1. Live Frame → `LocalGlobalTracker`（ORB + Lowe ratio + RANSAC Homography）对齐 Global Map。
2. 将全局坐标系下的标注投影到当前视口，过滤可见元件，叠加到 QGraphicsView。
3. 辅助：Fiducial 模板匹配 + `estimateAffinePartial2D` 做刚性仿射跟随（平移+旋转+等比缩放）。

### 3.2 后期 XYZ 机台平滑迁移

- **坐标系解耦**：元件位置始终存 **PCB 绝对物理坐标（mm）**，不绑死某一帧像素。
- 手动显微镜：视觉求矩阵 `M_manual`。
- 自动机台：串口/网口读 `(X,Y,Z)`，用标定公式  
  `Screen_Pos = (PCB_Phys_Pos − Machine_Current_Pos) × Scale(Z)`  
  替换视觉矩阵，架构几乎无需重构。

---

## 4. 数据模型

### 4.1 元件 JSON 示例

```json
{
  "component_id": "R1",
  "absolute_pos_mm": { "x": 12.5, "y": 34.2 },
  "base_pixel_box": [100, 150, 60, 30],
  "core_params": {
    "name": "R1",
    "value": "10K",
    "status": "unchecked"
  },
  "extended_params": {
    "package": "0805",
    "tolerance": "1%",
    "supplier": "",
    "sop_note": "注意焊接温度"
  },
  "meta": {
    "source": "yolo_seed",
    "updated_at": "2026-11-09T17:00:00+08:00"
  }
}
```

### 4.2 状态与颜色约定

| status | 含义 | 画布颜色 |
|--------|------|----------|
| `unchecked` | 待核对 | 红 |
| `checked` | 已核对 | 绿 |
| （可选）`doubt` | 存疑 | 黄（后续） |

### 4.3 持久化建议

- **MVP**：单板一个 JSON 文件 + 全局图路径引用；便于版本管理与 Diff。
- **v1**：SQLite（`boards` / `components` / `fiducials` / `sessions`），支持多板与核对历史。
- 画布只读写 `core_params`（≤3 键）；右侧栏读写 `extended_params` 全量。

### 4.4 全局工程清单结构（建议）

```json
{
  "board_id": "DEMO-PCB-001",
  "global_map_path": "assets/maps/DEMO-PCB-001.png",
  "pixel_per_mm": { "x": 42.0, "y": 42.0 },
  "fiducials": [
    { "id": "F1", "pixel": [120, 80], "mm": { "x": 0.0, "y": 0.0 } },
    { "id": "F2", "pixel": [3800, 90], "mm": { "x": 88.0, "y": 0.2 } }
  ],
  "components": [ "…见上例…" ]
}
```

---

## 5. 模块清单

| 模块 | 职责 | 建议路径 |
|------|------|----------|
| `app.main` | 启动入口、主窗口装配 | `src/app/main.py` |
| `app.ui.main_window` | QMainWindow、Dock、菜单/工具栏 | `src/app/ui/main_window.py` |
| `app.ui.canvas` | `PCBCanvasView`、点击拾取、内联编辑 | `src/app/ui/canvas.py` |
| `app.ui.annotation_items` | 框/线/气泡 GraphicsItem | `src/app/ui/annotation_items.py` |
| `app.ui.sidebar` | 右侧完整参数表单 | `src/app/ui/sidebar.py` |
| `vision.capture` | `VideoCaptureThread` | `src/vision/capture.py` |
| `vision.affine_tracker` | `PCBAffineTracker`（Fiducial + Affine） | `src/vision/affine_tracker.py` |
| `vision.local_global` | `LocalGlobalTracker`（ORB + Homography） | `src/vision/local_global.py` |
| `vision.mosaic` | **主路径** Mapping Mode / `VideoMosaicBuilder` | `src/vision/mosaic.py` |
| `vision.stitch` | 散图半自动拼接（**兜底/对照**） | `src/vision/stitch.py` |
| `vision.detect` | YOLO 自动检测种子（可选依赖 ultralytics） | `src/vision/detect.py` |
| `model.store` | JSON/SQLite 读写、状态色映射 | `src/model/store.py` |
| `model.schema` | 数据结构校验（pydantic 可选） | `src/model/schema.py` |
| `calib.xyz` | 后期机台标定矩阵（占位） | `src/calib/xyz.py` |
| `tests/` | 单元/冒烟：配准、投影、序列化 | `tests/` |
| `packaging/` | PyInstaller 规格、冻结依赖模板 | `packaging/` |

---

## 6. 文档阶段划分（供「文档架构」Bot）

1. **全局拼图 + 数据模型** — 瓦片采集规范、拼接质量验收、JSON/SQLite Schema。
2. **局部–全局配准 + 叠加渲染** — Tracker API、失败降级、可见性裁剪、性能预算。
3. **自动检测 + 人工核对 UX** — YOLO 种子、气泡≤3、侧栏扩展、状态色与批量核对。
4. **桌面应用封装** — PyInstaller、依赖分层（核心 vs ultralytics 可选）、安装与相机权限。
5. **后期：XYZ 机台标定矩阵** — 物理坐标解耦、编码器映射、Scale(Z)、串口协议草案。

产出建议文件：`docs/TECH_SPEC.md`、`docs/DATA_MODEL.md`、里程碑检查清单。

---

## 7. 里程碑：MVP → v1

### MVP（桌面可演示）

- [ ] Mapping Mode 视频扫描建图（主）+ 全局图/散图导入兜底（外部拼接结果可导入亦可接受）
- [ ] 数据模型落地（JSON）与样例板
- [ ] `VideoCaptureThread` + QGraphics 视频底图
- [ ] `LocalGlobalTracker` 或 `PCBAffineTracker` 至少一条可跑通路径
- [ ] 标注投影叠加；未核对红 / 已核对绿
- [ ] 画布内联改 ≤3 参数；侧栏扩展参数联动
- [ ] 基础测试计划与冒烟脚本

### v0.5

- [ ] YOLO 自动检测种子标注
- [ ] Fiducial + Affine 与 Local-Global 双路径可切换/融合
- [ ] 4K 下 ROI 模板匹配性能优化（≥15 FPS）
- [ ] SQLite 持久化与多板切换

### v1

- [ ] PyInstaller 一键打包安装包
- [ ] 拼接管线产品化（重叠提示、错位警告）
- [ ] XYZ 机台标定矩阵原型（可无真实机台用模拟坐标）
- [ ] 手机端可行性评估（非必须同期交付）

---

## 8. 开放风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| PCB 重复纹理导致 OpenCV Stitcher 误匹配/重影 | 全局图错位 → 全链路标注偏移 | **优先半自动网格抓图**；保留 20%–30% 重叠；必要时用 Hugin/PTGui 离线拼图再导入；禁止盲目全自动 Stitcher 作为唯一路径 |
| 4K 全图 `matchTemplate` 过慢 | 帧率掉到不可用 | **ROI 窗口匹配**（初始化记录孔位邻域，后续仅在 ~200×200 内搜）；可降采样粗匹配 + 原分辨率精修 |
| 单帧 Fiducial 出视野 | Affine 失败 | 降级使用上一帧矩阵；或切到 Local-Global ORB；UI 显示“跟踪丢失” |
| ORB 在低纹理区域匹配不足 | Homography 失败 | 提高特征数/换 AKAZE；强制依赖 Fiducial；提示用户移到特征更丰富区域 |
| YOLO 误检/漏检 | 人工核对负担大 | 检测仅作种子；状态默认 unchecked；提供批量通过/驳回 |
| Z 轴变焦导致尺度变化 | 框大小与线宽漂移 | Affine 含等比缩放；后期显式 Scale(Z)；气泡字号随尺度略调 |
| 打包体积（ultralytics/torch）过大 | 分发困难 | ultralytics 设为**可选依赖**；核心包不含检测权重；检测功能按需安装 |

---

## 9. 技术参考与实现要点

### 9.1 开源与可借鉴

- **OpenCV**：模板匹配、ORB/AKAZE、`estimateAffinePartial2D`、`findHomography`、（慎用）Stitcher。
- **Ultralytics YOLOv8 / YOLOv11**：元件检测种子；社区有 PCB Component Detection 数据集可微调。
- **PyQt6 QGraphicsView**：硬件加速分层、拾取、矩阵变换；可参考 Labelme / LabelImg 的标注交互。
- **视觉标定类开源**：搜索 `OpenCV projector camera calibration`、SMT/点胶机视觉叠加工件逻辑。

### 9.2 仿射跟随（手动显微镜）

- 基准 Fiducial 像素点 → 当前帧模板匹配中心 → `cv2.estimateAffinePartial2D` 得 2×3 矩阵 `M`。
- 将全局基准框四角 `cv2.transform` 后重算外接矩形；线宽/字号可按 `M` 的尺度项缩放。

### 9.3 局部–全局配准

- 全局图预提取 ORB 描述子；Live Frame KNN + Lowe ratio（如 0.75）+ RANSAC Homography。
- `cv2.perspectiveTransform` 投影全局标注；中心点落在视口内才渲染。

### 9.4 拼接工程建议

- **主路径**：视频扫描式增量 mosaic（Mapping Mode）；详见 `TECH_SPEC.md` v0.1.3。
- **不要**默认信任 OpenCV `Stitcher` 全自动结果（重复焊盘/走线易 Ghosting）。
- **兜底**：散图半自动抓图 / 离线专业拼图软件导入；用于对照或无扫描条件。
- 全局 mosaic 生成后；之后只维护标注库与 Live 配准。

### 9.5 性能（800W / 4K）

- 视频线程与 UI 解耦；重计算放 Worker，主线程只做 pixmap 与 item 更新。
- 模板匹配限 ROI；必要时金字塔粗到细。
- 仅更新可见标注项；避免每帧重建全部 GraphicsItem。

---

## 10. 协作 Bot 角色边界（摘要）

| Bot | 主责 | 典型产出 |
|-----|------|----------|
| **文档架构** | 需求澄清、技术规格、数据模型、里程碑与风险清单 | `TECH_SPEC.md`、`DATA_MODEL.md`、检查清单 |
| **视觉工程** | PyQt6/OpenCV 实现、Tracker、检测接入、测试 | 可运行脚手架、算法模块、测试计划 |
| **构建发布** | 依赖冻结、打包方案、CI/安装说明 | `requirements*.txt`、PyInstaller 规格、打包计划 |

三方以本简报为共同基线；规格变更经文档架构更新后再下沉到代码与打包。

---

## 11. 目录约定（建议仓库结构）

```
pcb-ar-annotate/
├── PROJECT_BRIEF.md          # 本文件
├── HANDOFF.md                # 三角色首批任务
├── docs/
│   ├── TECH_SPEC.md          # 待文档架构产出
│   └── DATA_MODEL.md         # 待文档架构产出
├── src/
│   ├── app/
│   ├── vision/
│   ├── model/
│   └── calib/
├── tests/
├── packaging/
├── assets/                   # 样例全局图、模板、权重（大文件另议）
└── requirements.txt
```

---

## 12. 非目标（本期不做）

- 手机端正式产品化（仅预留架构意识）。
- 完整 6DoF / SLAM。
- 与外部 ERP/MES/历史 BOM 对接。
- 生产级 SMT 闭环控制。

---

**下一步**：见同目录 `HANDOFF.md`，三角色按首批任务并行开工。
