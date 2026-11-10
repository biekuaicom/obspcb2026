# PCB 显微镜 AR 标注 — 技术规格（TECH_SPEC）

| 字段 | 值 |
|------|-----|
| 版本 | **v0.3.1** |
| 日期 | 2026-11-10（Asia/Shanghai） |
| 状态 | 草稿；**D1–D17 全部已确认**；拓扑主切片已落地（§3.8）；D17 建图骨架已通，**规格签名对齐落地实现** |
| 基线 | `PROJECT_BRIEF.md` v0.1、`HANDOFF.md` |
| 读者 | 视觉工程、构建发布、产品（用户） |
| 修订 | **v0.3.1**：D17 规格签名对齐落地 `src/vision/stable_capture.py`（视觉无需因文档改名/改签）。先前 **v0.3.0（D17）**：建图主路径改为「稳定式间隔拍照建图」。先前 **v0.2.1**：勾选拓扑下一刀；§3.8 签名对齐落地。先前 **v0.2.0（架构降维）**：D16。先前 v0.1.4：Mapping §3.6 对齐。先前 v0.1.3：D15。先前 v0.1.2 / v0.1.1：其余决策 |

---

## 1. 目标与主路径对比

### 1.1 目标（对齐简报 + D16 + D17）

显微镜正上方摄像头驱动的 PCB 元件识别与 AR 风格实时标注：自动/半自动种子入库 → 人工核对 → 视口平移/缩放/微旋转时标注框与气泡粘附跟随。本期仅桌面 PC。

**D16 降维（仍有效）**：工程库以**轻量 JSON / 拓扑**为主（元件 id、类型、逻辑坐标、周围过孔/焊盘相对几何指纹、`core_params`）。**不要求**像素级精确形变复原作为 Live 前提；目标鲁棒于新板/旧板/氧化烧黑。

**D17 覆盖建图主路径**：普通电脑跑 Video Mosaic 全图累加会卡死/报错，必须限制端侧算力。建库改为「稳定式间隔拍照」：只在画面清晰稳定时落盘局部瓦片（原图 + 二值 + 邻接相对位姿），运行时**不**生成超大全图像素画布。已拍区域以红色半透明色块叠在直播画面上（**本切片红区 = 上一张瓦片的投影**，见 §2.1 / §3.9）。**Live 叠加主路径仍是 D16** 的 `TopologyExtractor` + `ConstellationMatcher` / `GeometricLocalizer`（§3.8 已落地签名不因 D17 回改）。

### 1.2 主路径对比表（D15 / D16 / D17）

| 路径 | 角色 | 模块 | Live 是否必经 | 适用 |
|------|------|------|---------------|------|
| **稳定式间隔拍照建图** | **建图主路径（D17）** | `StableCaptureSession`（§3.9，`vision.stable_capture`） | **否**（建库入口） | 稳定才拍；瓦片 chain + 2×3 相对位姿；红区跟踪上一张；每拍抽点 |
| **Topology Live（几何星座 / point-set）** | **Live 主路径（D16）** | `TopologyExtractor` + `ConstellationMatcher` / `GeometricLocalizer`（§3.8） | **是** | 帧内几何 → 与库局部匹配 → 锁定位姿 → 叠气泡 |
| **Mapping Mode / Video Mosaic 全图累加** | **非默认 / 离线选配** | `VideoMosaicBuilder`（§3.6） | **否** | 须显式选择；普通电脑可能卡死/报错；不作为默认建图或 Live 前提 |
| **散图 zip / 瓦片 stitch** | **兜底 / 离线对照** | `SampleAssetImporter` + `SemiAutoStitchPipeline`（§3.7） | **否** | 无扫描条件、外部拼图、与用户全局整图目视对照（D8） |

> v0.3.0 草案曾用 `StabilityGate` / `IntervalScanController` / `TileGraph` / `vision.interval_scan` 作为主类名。**v0.3.1 以落地为准**，不再把这些作为主类名；视觉无需改名。

**推荐闭环（D16 + D17）**：

```
开始扫描 ──稳定门限──► 间隔拍照（原图+二值+邻接位姿）──每拍抽点──► Topology DB
                                                                      │
人工/检测在拓扑坐标上标注核对 ◄───────────────────────────────────────┘
                                                                      │
Live：当前帧几何抽取 → Constellation 匹配 → 锁定位姿 → 叠参数气泡
```

> Video Mosaic 全图累加**不在默认闭环内**。有既有 mosaic 时，仍可用 `LocalGlobalTracker` 作**旁路/对照配准**；**默认 Live 不依赖** `global_map_path`。散图 zip 仍为兜底。

### 1.3 MVP 成功标准（可演示即过）

| # | 标准 | 验收方式 |
|---|------|----------|
| M1 | 工程具备可用 **`topology_db`**（内嵌或相对路径）；至少若干元件含几何指纹 | 打开工程可见拓扑库；字段校验通过 |
| M2 | 间隔拍照建图可跑通：稳定才拍、重叠/失跟门禁、chain 落盘且运行时无全图画布 | 冒烟见 §3.9 / MILESTONES D17；模块级产物已在 `artifacts/stable_capture_map/` |
| M3 | 拓扑抽取 + 点集入库可跑通（含每拍追加） | 冒烟：帧/瓦片 → 特征点集 → 写入 topology |
| M4 | **Live** 帧对库做 constellation / point-set 局部匹配，锁定位姿后叠标注 | 至少一条拓扑匹配路径可演示；可见元件叠加 |
| M5 | 画布内联 ≤3 核心参数；侧栏扩展字段；红/绿状态色 | 双向绑定联调通过 |
| M6 | 800W/4K 下跟踪帧率可接受（目标 ≥15 FPS，争取 30） | 冒烟计时；参考机优先，各机实测（D14） |
| M7（可选） | 离线 Video Mosaic / 散图对照可打开 | **须显式选择**；**不阻塞**间隔拍照建图与 Topology Live MVP |

---

## 2. 工作流与状态机

### 2.0 应用状态机（建图 = 间隔拍照；Live = 拓扑匹配）

```
                         ┌──────────┐
                         │   Idle   │
                         └────┬─────┘
              ┌───────────────┼────────────────────┐
              │ 建图主路径    │                    │ 非默认 / 离线
              ▼               ▼                    ▼
     ┌────────────────┐  ┌──────────────┐  ┌────────────────┐
     │ StableCapture  │─►│ TopologyBuild│  │ Video Mosaic   │
     │ （§2.1 / §3.9）│  │ （抽特征/    │◄─│ 全图累加       │
     │ 稳定间隔拍照   │  │  入库/核对） │  │ （显式选配）   │
     └────────────────┘  └──────┬───────┘  └────────────────┘
                                │ topology_db 就绪
                                ▼
                         ┌──────────────┐
                         │ Annotating   │  ← 在拓扑坐标（或可选 mosaic）上框选/核对
                         └──────┬───────┘
                                │ 开始 Live
                                ▼
                         ┌──────────────┐
                         │ LiveTopology │  ← 帧→几何→Constellation 匹配→叠加
                         └──────────────┘
                                │ 丢失 / 停止
                                ▼
                           Idle / Annotating
```

| 事件 | 行为 |
|------|------|
| 开始扫描 | `StableCaptureSession.start()`，进入 §2.1 观察/拍摄循环（默认建图入口） |
| 进入 TopologyBuild | 可由间隔拍照每拍入库驱动；亦可从单帧/录像/（离线）mosaic 半自动抽特征；写 `topology_db` |
| 离线 Video Mosaic | **须显式选择**；不改变「Live 不依赖 mosaic」；成功后可辅助抽特征或对照 |
| 取消间隔扫描 / Mapping | 丢弃未提交会话；**已落盘瓦片与已有 topology_db 保留** |
| 无 `topology_db` 启动 Live | **失败降级**：提示先建库；可选回退到有 mosaic 时的 LocalGlobal（旁路）；禁止静默空叠加 |
| Live 跟踪丢失 | TTL 0.8 s（D4）；超时冻结并提示；不清除库 |

### 2.1 稳定式间隔拍照建图状态机（D17，建图主路径）

落地以 `start` / `observe` / `capture` / `render` / `save_session` 为循环，**没有**单独的 `stable_hold_ms`，也没有显式枚举状态机类。UI 提示以 `ObserveState.message` / `CaptureDecision.reason` 的**原句**为准：

| 条件 | 文案（保留原句） |
|------|------------------|
| 尚未 `start()` | 请先点击开始 |
| 已开始但不清晰 | 画面模糊，等待清晰 |
| 已开始、清晰，但还没有上一帧（`motion_px is None`） | 等待下一帧确认稳定 |
| 已开始、清晰，但 `motion_px > max_motion_px` | 画面仍在移动，等待稳定 |
| 清晰且稳定 | 稳定，可以拍摄 |
| `capture()` 成功 | 已拍摄 |
| 后续张重叠不足 | 红区重叠不足，请退回重拍 |
| 后续张跟踪失败 | 未能跟上已拍区域，请退回红色覆盖处重拍 |

```
Idle（未 start）
  observe / capture → 「请先点击开始」
  │ 点击开始 → start()
  ▼
逐帧 observe(frame)
  ├─ 不清晰 → 「画面模糊，等待清晰」
  ├─ 尚无上一帧 → 「等待下一帧确认稳定」
  ├─ 仍在移动 → 「画面仍在移动，等待稳定」
  └─ 清晰且稳定 → 「稳定，可以拍摄」
        │ capture()   ← 未就绪则按上表拒绝，不落盘
        ├─ 首张：无重叠检查；抽出拓扑；matcher 绑定该瓦片
        └─ 第二次及以后：
              ├─ 未能跟踪上一张 → 「未能跟上已拍区域，请退回红色覆盖处重拍」
              ├─ 视口与红区 overlap < min_overlap → 「红区重叠不足，请退回重拍」
              └─ 成功：「已拍摄」；写邻接边；matcher 改绑到新瓦片拓扑
        │
        └─ 停止 / save_session → 写出 tiles + chain.json + topology_db.json
           + summary.json（full_canvas: false）；可选 coverage.png
```

**稳定判定（落地）**：须已 `start`，且清晰（`frame_sharpness` ≥ `min_sharpness`），且已有上一帧，且 `interframe_motion_px` ≤ `max_motion_px`。第一帧 `observe` 之后 `motion_px` 为 `None`，提示「等待下一帧确认稳定」。

**门限（落地默认值，【可迭代】）**

| 参数 | 默认 | 说明 |
|------|------|------|
| 清晰度 `frame_sharpness` | Laplacian 方差 | `min_sharpness` **40.0**【可迭代】 |
| 运动 `interframe_motion_px` | `phaseCorrelate` 位移幅度，折回原图像素 | `max_motion_px` **8.0**【可迭代】 |
| `min_overlap` | **0.08**（仍在 0.08–0.15 区间） | `frame_overlap_ratio`：上一张投影四边形与当前视口的相交面积 / 视口面积。只要留一小部分红区即可，**禁止**要求大面积重叠【可迭代】 |
| `max_long_edge` | **640** | 本会话内 `TopologyExtractor` 降采样长边【可迭代】；不改变 §3.8 默认 1280 |
| （无）`stable_hold_ms` | — | 落地靠「须有上一帧」确认稳定，不另计时 |

**红区行为（本切片 vs 需求原文）**

- **本切片落地**：红区 = **上一张瓦片**经 `GeometricLocalizer.track` 得到的 `pose.M`（**2×3**，库/上一张 → 当前帧）投影到当前帧的四边形，再 `paint_red_coverage`（默认 `alpha=0.45`）。拍成后 matcher **改绑到新瓦片**拓扑，下一帧跟踪的是新的「上一张」。运行时无全景像素画布。
- 需求原文是「已拍区域」并集。**全部已拍足迹并集**标为【可迭代】随后项，**不要求**视觉为对齐文档在本刀改实现。
- 本帧跟踪失败：不盲目平移红区；`capture()` 走「未能跟上已拍区域…」拒绝。

### 2.2 阶段 A — 拓扑建库（主：间隔拍照）+ 离线 mosaic 选配

| 步骤 | 输入 | 输出 | 失败降级 |
|------|------|------|----------|
| **A0 间隔拍照（默认）** | Live，§2.1 | `chain.json` + 瓦片原图/二值 + `topology_db.json`；工程 `map_source=interval_tiles`，`tile_graph` 可指向 chain | 不稳/无重叠/失跟 → 拒拍重拍；取消保留已落盘 |
| A0′（离线选配）Video Mosaic | 显式选择后的弓字形扫板 | `assets/maps/<board_id>.png` | 普通电脑可能卡死/报错；失败不阻塞 A0 |
| A0b（可选）散图兜底 | 瓦片 zip + `global_ref` | 对照 / 兜底 | 见 §3.7 |
| A1 拓扑抽取 | 当前瓦片帧（或离线 mosaic ROI） | 孔/焊盘/轮廓点集 | 特征过少 → 提示换区/调阈值/重拍 |
| A2 点集入库 | 各瓦片本地像素系点集（id 带瓦片前缀） | `topology_db`（`kind=stable_capture_chain`） | 写盘失败 → 提示重试 |
| A3 种子标注 | 拓扑坐标 +（可选）YOLO | `components[]`，`status=unchecked` | 无 YOLO → 纯人工 |
| A4 人工核对 | 用户编辑 | JSON 写回 | 校验失败拒绝写回 |

**阶段 A Done when**：工程 JSON 可打开；**`topology_db` 可加载**；至少 1 个元件有逻辑坐标 + 指纹（或明确「待抽」占位）与 `core_params`。默认建图产出是 **chain**（不是全景 mosaic）。`global_map_path` **非必须**，也不是间隔扫描或 Live 的前提。将各瓦片点变换到统一参考系、以及用合并库做 Live 全板匹配，见随后项（§3.9）。

### 2.3 阶段 B — Live 拓扑匹配叠加（主路径，D16，不变）

| 步骤 | 输入 | 输出 | 失败降级 |
|------|------|------|----------|
| B1 取帧 | `VideoCaptureThread` | `frame` + 时间戳 | 无相机 → synthetic/录像（D6） |
| B2 预处理 | BGR 帧 | 灰度 + **自适应二值** | 曝光极端 → 调 blockSize/C 或 ROI |
| B3 几何抽取 | 二值图 | 轮廓 / 孔 / 焊盘候选点集 | 点数不足 → 本帧跳过匹配 |
| B4 局部匹配 | 帧点集 + `topology_db` | 位姿（相似/仿射）+ 置信度 + 命中元件 | 见 §2.4 |
| B5 投影过滤 | 库内逻辑框/坐标 + 位姿 | 视口像素框；中心在视口内才渲染 | 无有效位姿 → 不画或冻结 |
| B6 叠加渲染 | 可见元件 | QGraphics 更新（不重建全部 Item） | 主线程过载 → 跳帧 |
| B7 编辑回写 | 气泡/侧栏 | Model → 持久化 | 校验失败提示 |

**旁路 B′（有 mosaic 时）**：`LocalGlobalTracker` / `PCBAffineTracker` 仍可作对照或应急；**默认不启用为 Live 主路径**（D16 覆盖 D3 的「默认主配准」角色：D3 仍描述 mosaic 旁路优先级，Live 默认走拓扑）。

### 2.4 Live 匹配失败降级

```
优先序（D16 + D4）：
1. ConstellationMatcher / GeometricLocalizer 成功且置信度达标 → 用估计位姿
2. （可选旁路）若工程有 global_map 且启用旁路：LocalGlobal → Affine（D3）
3. 否则沿用上一帧位姿（TTL = 0.8 s）
4. TTL 过期 → 冻结标注；UI「跟踪丢失」；可提示移到孔/焊盘更丰富区域
5. 无 topology_db → 禁止进入 Live 主路径；明确提示建库
```

---

## 3. 模块接口草案

路径对齐简报第 5 节。签名级约定。**§3.8 已落地，签名与实现一致，D17 不回改。§3.9 已按落地 `stable_capture.py` 对齐（v0.3.1）。**

### 3.1 `vision.capture` — `VideoCaptureThread`

路径：`src/vision/capture.py`

```python
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np

class VideoCaptureThread(QThread):
    """OpenCV 采集线程；禁止在主线程做 read()。"""
    frame_ready = pyqtSignal(object, float)  # (np.ndarray BGR, timestamp_s)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, source: int | str = 0, width: int | None = None,
                 height: int | None = None, parent=None): ...
    def run(self) -> None: ...
    def stop(self, wait_ms: int = 2000) -> None: ...
    def set_source(self, source: int | str) -> None: ...
```

约定（**D6 已确认**）：默认索引 0；近 4K 优先；无相机必须 synthetic/录像。

### 3.2–3.3 Tracker（**可选旁路**，有 mosaic 时）

`PCBAffineTracker` / `LocalGlobalTracker` 签名保持 v0.1.4；**角色改为可选旁路/对照**，非 Live 默认。详见仓库既有实现与 docstring。D1 方向（全局→当前）在旁路启用时仍适用。

### 3.4 `model.store` / `model.schema`

```python
class BoardProject: ...          # 见 DATA_MODEL.md（含 topology_db；tile_graph 可指向 chain）
class Component: ...
class Fiducial: ...
class TopologyDB: ...            # 板级拓扑库
STATUS_COLORS = {"unchecked": "#E53935", "checked": "#43A047", "doubt": "#FDD835"}

class ProjectStore:
    def load(self, path: str) -> BoardProject: ...
    def save(self, project: BoardProject, path: str | None = None) -> None: ...
    def update_component(self, component_id: str, *, core: dict | None = None,
                         extended: dict | None = None, status: str | None = None) -> None: ...
    def load_topology(self, project: BoardProject) -> TopologyDB: ...
    def save_topology(self, project: BoardProject, topo: TopologyDB) -> None: ...
    def list_visible(self, viewport_polygon, transform) -> list[Component]: ...
```

读写边界：画布只读写 `core_params`（`name`/`value`/`package`，D12）；侧栏 `extended_params`；顶层 `status` 权威（D9）。路径相对工程目录（D11）。

> D17 工程字段 `tile_graph`（指向 `chain.json` 或内嵌 chain）与 `map_source=interval_tiles` 的 **ProjectStore / UI 接入尚未落地**，由后续补齐；本切片骨架未改 store。

### 3.5 UI

**默认建图入口**为间隔拍照扫描（开始 / 提示文案见 §2.1 / 红区 `render` / 拒拍重拍 / 停止保存）。骨架已提供 `observe` 文案与 `render` 红区；**产品窗口接入待做**。Video Mosaic Mapping UI 为**非默认 / 离线选配入口**（须显式选择）。强调：Topology 建库进度、Live 匹配置信度、无拓扑库时的阻断提示。其余气泡/侧栏/双向绑定约定不变（§ 历史 v0.1.x）。

### 3.6 `vision.mosaic` — `VideoMosaicBuilder`（**非默认 / 离线选配**，D15→D16→D17）

> **角色（D17）**： **非默认 / 离线选配**。默认建图走 §3.9 间隔拍照，**默认不再做全图累加**。已落地实现（v0.1.4 签名）**保留**；**不阻塞**间隔拍照与 Topology Live。启用须显式选择，并接受普通电脑上可能卡死/报错。

路径：`src/vision/mosaic.py`（签名对齐落地：`MappingController`、`push_frame(ndarray)`、`progress()`、`finalize(..., board_id, repo_root)`、`cancel()->MappingState`）。

写出：`assets/maps/<board_id>.png`；工程可写 `global_map_path`、`map_source=video_mosaic`、`meta.mapping`。**Live 主依赖仍为 `topology_db`**。

### 3.7 `vision.stitch` / 样例导入 — 散图兜底（**非主路径**）

角色不变：对照/兜底/离线（D8）。接口草案同 v0.1.4 §3.7（`SampleAssetImporter` / `SemiAutoStitchPipeline`）。

### 3.8 `vision.topology` — 拓扑抽取与星座匹配（**Live 主路径**，D16）

路径：`src/vision/topology.py`（**已落地**；下列签名与实现一致，文档跟代码）。**D17 复用本 API，无需因本文回改。**

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class GeoPoint:
    id: str
    kind: str                 # hole | pad | slot
    xy: tuple[float, float]
    area: float = 0.0
    circularity: float = 0.0
    # to_dict() → {id, kind, xy:[x,y], area, circularity}

@dataclass
class TopologyExtractResult:
    ok: bool
    points: list[GeoPoint]
    width: int
    height: int
    scale: float = 1.0
    message: str = ""
    board_id: str = ""
    source: str = ""
    schema_version: str = "0.2-topo"
    # counts() / xy() / to_dict() / save(path) / from_dict(data)

@dataclass
class Pose2D:
    """Library → frame. M 为 2×3；精神同 D1（库/全局像素 → 当前帧）。"""
    M: np.ndarray             # 2×3 float64
    inliers: int
    confidence: float
    ok: bool
    message: str = ""
    elapsed_ms: float = 0.0
    scale: float = 1.0
    angle_deg: float = 0.0
    query_points: int = 0
    # 可选调试：pose_error_px / pose_error_deg

class TopologyExtractor:
    """灰度 + 自适应二值；长边 > max_long_edge 先降采样。"""

    def __init__(self, max_long_edge: int = 1280,
                 adaptive_block: int = 31, adaptive_C: int = 5): ...

    def extract(
        self,
        frame_bgr: np.ndarray,
        roi_xywh=None,
        *,
        working_scale: float | None = None,
    ) -> TopologyExtractResult: ...

class ConstellationMatcher:
    """帧星座 ↔ 库点集局部匹配；须先 set_topology。"""

    def __init__(self, min_inliers: int = 8, conf_thresh: float = 0.35): ...

    def set_topology(self, topology: TopologyExtractResult | dict | None) -> None:
        """换板/清库；None 清空。dict 经 TopologyExtractResult.from_dict。"""

    def localize(self, frame_bgr_or_result) -> Pose2D:
        """接受 frame_bgr 或 TopologyExtractResult；无库 → ok=False。"""

    def project_components(self, components, pose: Pose2D) -> list[dict]:
        """可选：经 pose.M 投影 library base_pixel_box → 帧角点。"""

class GeometricLocalizer:
    def __init__(self, extractor: TopologyExtractor,
                 matcher: ConstellationMatcher): ...
    def track(self, frame_bgr: np.ndarray) -> Pose2D: ...

def geo_fingerprint(
    points: list[GeoPoint],
    anchor_xy: tuple[float, float],
    *,
    neighbors: int = 6,
) -> dict:
    """元件级相对星座；返回 {kind:'constellation', anchor_xy, neighbors:[{id,kind,dxy},...]}。"""
```

**无拓扑库时**：`set_topology` 未调用 / `None` / 库空 → `localize`/`track` 返回 `ok=False`，`message` 为「未设置拓扑库，无法定位」或「拓扑库没有可用点」；UI 阻断 Live 主路径。

**建库辅助**：间隔拍照每张成功瓦片会 `extract` 并写入 chain / `topology_db`（D17，见 §3.9）。扫描过程中 matcher **只绑定上一张（刚拍成的）瓦片**，供红区跟踪。亦可从（离线）mosaic 或扫描帧批量 `extract`，聚类/关联到 `component_id` 后写 `geo_fingerprint`（半自动；人工可改）。

**本切片 Done when**：已满足（见 MILESTONES v0.2.1）— 抽取 → `assets/topology/SAMPLE-PARTIAL-001.json`（`0.2-topo`）→ localize ok + `C-LIVE` overlay；**不要求** mosaic。

### 3.9 `vision.stable_capture` — 稳定式间隔拍照建图（**建图主路径**，D17）

路径：`src/vision/stable_capture.py`（**骨架已落地**；下列签名与实现一致，文档跟代码。视觉无需因 v0.3.1 改名/改签。）

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def frame_sharpness(frame: np.ndarray) -> float:
    """Laplacian 方差。"""

def interframe_motion_px(prev: np.ndarray, curr: np.ndarray) -> float:
    """phaseCorrelate 位移幅度，结果折回原图像素。同尺寸帧。"""

def binary_lines(frame: np.ndarray) -> np.ndarray:
    """自适应阈值二值线条。"""

def frame_overlap_ratio(quad: np.ndarray, width: int, height: int) -> float:
    """上一张瓦片投影四边形与当前视口的相交面积 / 视口面积。"""

def paint_red_coverage(frame_bgr: np.ndarray, quad: np.ndarray, alpha: float = 0.45) -> np.ndarray: ...

def project_tile_quad(matrix_2x3: np.ndarray, width: int, height: int) -> np.ndarray:
    """用 2×3 矩阵把瓦片四角映到当前帧。"""

@dataclass
class ObserveState:
    started: bool
    sharp: bool
    sharpness: float
    motion_px: float | None
    stable: bool
    message: str
    overlap: float | None = None
    track_ms: float | None = None
    tracked: bool = False

@dataclass
class CaptureDecision:
    ok: bool
    reason: str
    overlap: float | None = None
    tile_id: str | None = None
    elapsed_ms: float = 0.0

class StableCaptureSession:
    def __init__(
        self,
        *,
        min_sharpness: float = 40.0,     # 【可迭代】
        max_motion_px: float = 8.0,      # 【可迭代】
        min_overlap: float = 0.08,       # 【可迭代】仍在 0.08–0.15
        max_long_edge: int = 640,        # 本会话 Extractor；【可迭代】
    ): ...

    def start(self) -> None: ...
    def observe(self, frame: np.ndarray) -> ObserveState: ...
    def capture(self) -> CaptureDecision: ...
    def render(self, frame: np.ndarray | None = None) -> np.ndarray:
        """在当前帧画红区（上一张投影四边形）。"""
    def chain_dict(self) -> dict: ...
    def topology_db(self, *, board_id: str, source: str) -> dict: ...

def save_session(
    session: StableCaptureSession,
    out_dir: Path,
    *,
    board_id: str,
    source: str,
) -> dict:
    """写出 tiles/{id}.png、{id}-binary.png、chain.json、topology_db.json、
    summary.json（full_canvas: false）；可选 coverage.png。"""
```

**约定**

- `observe`：每帧做清晰/稳定判定；若已有瓦片，则用 `GeometricLocalizer.track` 跟踪**上一张**并计算 `overlap`。返回叠加与提示所需状态。**不**拼全景画布。
- `capture`：未 `start`、不清晰、尚无上一帧、仍在移动 → 拒绝（文案见 §2.1），记入 `rejects`。首张不做重叠检查。后续张须 `tracked` 且 `overlap ≥ min_overlap`，否则分别走失跟 / 重叠不足拒绝。成功时瓦片 id 为 `cap-000` 形式，并写邻接边。
- 边的 `M` 为 **2×3**（与 `Pose2D.M` 同形），`direction="previous_tile_to_current_frame"`（上一张瓦片像素 → 当前帧）。可选 `track_ms`、`overlap`。不要再强制 3×3 `M_rel`。
- 红区与 matcher 绑定见 §2.1：本切片只投影上一张；拍成后改绑新瓦片。
- `save_session`：运行时**不**生成全图像素画布；`summary.full_canvas` 恒为 `false`。实现侧目前没有全图画布函数，规格也不虚构 `assemble_full_canvas`。禁止默认全图累加。
- 矢量字段继续预留，**本切片未做**；二值 PNG 已有（`{id}-binary.png`）。
- 工程目标目录建议 `assets/scans/<board_id>/`（tiles + `chain.json` + `topology_db.json`）。模块冒烟产物已在 `artifacts/stable_capture_map/`（3 张成功 `cap-000`..`cap-002`，2 次拒绝：模糊 / 跟踪丢失，`full_canvas=false`）。`map_source` 语义默认仍为 **`interval_tiles`**；实现侧目录/产物名现为 `stable_capture` / `artifacts/stable_capture_map`。
- `topology_db`：`kind="stable_capture_chain"`，`schema_version` 仍为 `"0.2-topo"`；点 id 为 `{tile_id}:{point_id}` 并带 `tile_id`；**点坐标在各瓦片自己的像素系**；`image_size` 可为 `null`；附注「没有全板像素画布」。扫描中 matcher 只绑定上一张，故「入库后对本张跟踪 / localize」在骨架里可用。把各瓦片点变到统一参考系、再用合并库做 Live 全板匹配，标为【可迭代】随后项（需工程接入，不要求本刀改签）。

**本切片官方勾选**：见 `MILESTONES.md` D17。骨架与模块冒烟已通；工程 / UI / 并集红区 / 矢量 / ProjectStore 接入仍待对照后勾选。**不要求** mosaic 全图。

---

## 4. 性能预算

| 项 | 预算 / 约定 |
|----|-------------|
| 相机 | ~800 万像素；近 4K 优先；失败用实际分辨率（D6） |
| 目标帧率 | **Live 拓扑跟踪** ≥15 FPS，争取 30（D14）；间隔扫描预览同样按此约束端侧算力 |
| **二值/轮廓** | **禁止默认全分辨率 4K 重形态学**；策略：**长边降采样**（§3.8 默认 `max_long_edge≈1280`；间隔拍照会话默认 **640**）和/或 **ROI 窗口**；位姿估在降采样域，投影回原分辨率 |
| **间隔扫描（默认建图）** | 运行时**不生成超大全图像素画布**；每帧只做稳定判定 + 上一张拓扑跟踪 + 红区投影。会话内可保留已拍瓦片原图以便 `save_session` 写出，这不是全景画布累加 |
| Video Mosaic 全图累加 | **非默认 / 离线选配**；普通电脑已知会卡死/报错（见 §6.2）；若启用须有进度且勿卡 UI，失败可中止 |
| 基准机 | `DESKTOP-J4I5P0U` 优先参考；各机实测记录（D14） |
| 旁路 ORB | 仅当启用 mosaic 旁路时；全局描述子预计算 |
| 线程 | Capture + 抽取/匹配 + `StableCaptureSession` 判定/提交（+ 可选离线 Mosaic Building）在 Worker；主线程只更新 UI |
| 标注更新 | 仅可见项；禁止每帧销毁重建全部 Item |
| 降载 | 匹配跟不上则跳帧匹配、仍显示上一姿态（带 TTL） |

---

## 5. 文档阶段 1–5 验收标准

| 阶段 | 范围 | 验收标准 |
|------|------|----------|
| **1** 拓扑库 + 数据模型 + 瓦片 chain | `topology_db`、chain / `tile_graph`、指纹 schema、mosaic 为离线选配 | `DATA_MODEL.md` 定稿；主/选配角色清晰 |
| **2** 间隔建图 + Live 拓扑匹配 + 叠加 | §2–§4、§3.8 / §3.9、失败降级、性能 | 间隔拍照可演示；Topology Live 可演示 |
| **3** 检测 + 核对 UX | YOLO 种子、气泡≤3、侧栏、状态色 | §3.5；检测可选 |
| **4** 桌面封装 | PyInstaller、依赖分层 | §6；torch 不进核心包 |
| **5** XYZ 机台 | 物理坐标解耦、自动扫板可选 | `calib/xyz` 占位；本期不真机 |

---

## 6. 发布约定（D6 / D7 / D14 已确认）

> 供「构建发布」对齐；可运行代码 + 冒烟通过前**不执行正式打包**。

| 项 | 约定 |
|----|------|
| 平台 | 优先 Windows x64；参考机 `DESKTOP-J4I5P0U`（非唯一） |
| 产物 | PyInstaller **onedir** 便携为先；VERSION 单源；禁止无版本号（D7） |
| 核心依赖 | `opencv-python`、`PyQt6`、`numpy`；运行时不含 pydantic（D13） |
| 资源外置 | **topology JSON**、**scans / chain**、可选 mosaic、模板、权重、样例外置 `assets/` |

### 6.1 冒烟验收项

1. 冷启动无未捕获异常  
2. 打开主窗口  
3. 加载样例工程 + **`topology_db`**（`map_source=interval_tiles` 时 chain / `tile_graph` 须可解析；mosaic 可选）  
4. 无相机：synthetic/录像可出帧（D6）  
5. 有相机（可选）：索引 0；近 4K 失败仍可用实际分辨率  
6. 正常退出无崩溃  
7. 性能笔记：机名 + FPS；参考机优先 ≥15 FPS（D14）

D17 模块级冒烟已有（稳定门限、重叠/失跟拒绝、三张以上瓦片+边、上一张红区跟随、`full_canvas=false`），见 `artifacts/stable_capture_map/`。工程与 UI 冒烟仍待接入。

### 6.2 已知限制（增补）

| 依赖/限制 | 说明 |
|-----------|------|
| **Video Mosaic 全图累加导致卡死/报错** | 普通电脑端侧算力不足；**故 D17 将全图累加降为非默认 / 离线选配**，默认走间隔拍照。若用户显式启用离线全图累加，允许因内存失败并须明确报错 |
| **chain 点坐标分属各瓦片像素系** | 合并 `topology_db`（`kind=stable_capture_chain`）尚无全板统一坐标系；扫描跟踪只绑上一张。全板 Live 匹配待随后 |
| 拓扑误匹配 | 重复局部结构 → 需指纹维度/几何校验（见 Gaps） |
| 氧化烧黑 | 孔/焊盘缺失 → 自适应阈值 + 不要求像素全景 |
| 大 mosaic | 仅离线选配；OOM 风险仍在；可限制边长 |
| Mapping 漂移 | 离线选配问题；不作为 Live 或间隔扫描前提 |

### 6.3 样例资产（D8；对照/兜底）

源 `E:\局部图片.zip`；落盘 `assets/samples/`；**不**作为 Live 或默认建图主依赖。主依赖：`assets/topology/<board_id>.json`（或工程内嵌），以及默认建图产出的 `assets/scans/<board_id>/`（模块冒烟现位于 `artifacts/stable_capture_map/`）。

---

## 7. Gaps / Risks 与决策摘要

### 7.1 风险

| 风险 | 对策（规格层） |
|------|----------------|
| **普通电脑全图 mosaic 卡死/报错** | D17：运行时不生成超大全图像素画布；默认间隔拍照；全图仅显式离线选配 |
| **重复局部结构误匹配** | 指纹含相对向量/极坐标直方图 + 元件类型门控；拒识低置信度 |
| **氧化导致孔缺失** | 自适应二值；指纹允许多假设；不依赖像素全景 |
| **指纹维度不足/过拟合** | schema 标【可迭代】；先相对邻域 K 近邻向量 |
| **红区只跟上一张 / 跟踪丢失** | 本切片明确为上一张投影；并集为随后项；失败不盲目平移，宁可拒拍 |
| Mapping 长扫漂移 | 离线选配问题；高重叠/Rescan；不阻塞 Live / 间隔扫描 |
| 4K 全图二值过慢 | 降采样 / ROI（§4） |
| YOLO 误检 | 仅种子；默认 unchecked |
| 打包体积 | ultralytics 可选 |

### 7.2 决策状态摘要

见 `MILESTONES.md`。**D1–D17 已确认**（D17：2026-11-10 Asia/Shanghai）。要点：

- D1–D14：同前（矩阵方向、纯像素 MVP、旁路 Tracker 序、TTL、doubt、相机、版本、样例、status、轴向、工程目录、core 三键、手写校验、FPS 机名）
- **D15**：曾引入 Mapping / video mosaic 建图；其后角色被后续决策收窄
- **D16**：Live 主路径 = **结构化几何拓扑**；mosaic **非** Live 必经；工程以 `topology_db` 为主（本结论仍有效）
- **D17**：建图主路径 = **稳定式间隔拍照**（落地模块 `StableCaptureSession`）；Video Mosaic 全图累加 = **非默认 / 离线选配**；运行时不生成超大全图像素画布。D17 **覆盖建图主路径**，不推翻 D16 的 Live 匹配。v0.3.1 仅把规格签名与落地对齐。

**仍开放**：无阻塞立项的决策。实现/产品随后项见 §3.9 与 MILESTONES（并集红区、统一逻辑坐标、矢量、工程接入），标【可迭代】。

---

## 8. 非目标（本期不做）

- 手机端正式产品化  
- 完整 6DoF / SLAM / 完整透视畸变建模  
- **以像素级全景精确复原作为 Live 前提**（D16）  
- **运行时拼整张全景像素画布**（D17）  
- **默认开启 Video Mosaic 全图累加**（D17）  
- 外部 ERP/MES/历史 BOM 对接  
- 生产级 SMT 闭环  
- OpenCV `Stitcher` 作为唯一建图路径  
- 散图 zip 作为唯一/主建图路径  
- 核心包捆绑 torch/ultralytics  
- 无版本号发布产物（D7）  
- Mapping 真机 XYZ 自动扫板（接口预留）

---

## 9. 本轮协作提示

- **视觉工程**：D17 骨架已通（`stable_capture.py`）。v0.3.1 **文档跟实现对齐**，**无需因本文改名/改签**。按对齐后的 §3.9 继续补齐工程接入与验收。不要为红区/建图去拼运行时全景画布。本切片红区 = 上一张投影；全部已拍并集为随后项。已落地的 §3.8 拓扑 API 继续复用。Video Mosaic 保留作非默认/离线选配。  
- **构建发布**：§6；拓扑 JSON 与 scans/chain 外置；可运行 + 冒烟通过前不打包。  
- 规格变更：先改本文档与 DATA_MODEL，再通知视觉/构建。  
- 与 `PROJECT_BRIEF.md` / 旧文冲突时，**以本文 v0.3.1（D17）为准**。
