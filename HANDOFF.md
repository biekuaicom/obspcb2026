# HANDOFF — 三角色首批任务

> 共同基线：`/workspace/pcb-ar-annotate/PROJECT_BRIEF.md`  
> **现行规格**：`docs/TECH_SPEC.md` **v0.3.1**（**D17**：建图主路径 = 稳定式间隔拍照，规格已对齐落地 `stable_capture.py`；**D16**：Live 主路径 = 结构化几何拓扑，拓扑主切片已落地）  
> 仓库根：`/workspace/pcb-ar-annotate/`  
> 原则：只改文档先行；视觉按对齐后的 TECH_SPEC §3.9 补齐工程接入与验收（**无需因 v0.3.1 改名/改签**）；**构建发布在可运行代码 + 冒烟通过前只做计划与依赖模板，不打包。**  
> 更新日期：2026-11-10（Asia/Shanghai）— **v0.3.1 / D17：规格签名对齐落地 `src/vision/stable_capture.py`**。

---

## 0. 主路径对齐（全员，D16 + D17）

| 路径 | 角色 | 模块 |
|------|------|------|
| **稳定式间隔拍照建图** | **建图主路径（D17）** | `StableCaptureSession`（`TECH_SPEC` §3.9，`src/vision/stable_capture.py`，**骨架已通**） |
| **Topology Live** | **Live 主路径（D16）** | `TopologyExtractor` + `ConstellationMatcher` / `GeometricLocalizer`（§3.8，**已落地**） |
| Mapping Mode / Video Mosaic 全图累加 | **非默认 / 离线选配** | `VideoMosaicBuilder`（§3.6，已落地可保留；须显式选择） |
| 散图 zip / 半自动瓦片拼接 | **兜底 / 离线对照** | `SampleAssetImporter` + `SemiAutoStitchPipeline`（§3.7） |

**闭环**：开始扫描 → 稳定门限 → 间隔拍照（原图+二值+邻接 2×3 位姿；本切片红区跟踪上一张）→ 每拍抽点 → 人工核对 → **Live 几何星座匹配叠加**。  
**非目标**：不以像素级全景精确复原作为 Live 前提；**运行时不拼整张全景像素画布**；**默认不开启** Video Mosaic 全图累加。

若 `PROJECT_BRIEF.md` 或旧文仍写「拼接全局高清 + LocalGlobal 为 Live 前提」或「默认 Video Mosaic 建图」，**以 TECH_SPEC v0.3.1 为准**。

---

## 1. 文档架构 Bot

**角色定位**：需求/架构文档、数据模型、里程碑与风险。

### 首批任务（已完成并持续修订）

1. 通读 BRIEF，维护 gaps。  
2. **`docs/TECH_SPEC.md`**：间隔拍照建图主路径（对齐 `StableCaptureSession`）、Topology Live、Video Mosaic 离线选配、性能、发布约定。  
3. **`docs/DATA_MODEL.md`**：`tile_graph` 指向 `chain.json`、`map_source=interval_tiles`、`kind=stable_capture_chain`、与 `global_map_path` 关系。  
4. **`docs/MILESTONES.md`**：可勾选；**D16 拓扑主项已勾选**；**D17 官方项保持未勾选**（骨架已通，待工程与 UI）。  
5. Gaps/Risks（全图卡死、误匹配、氧化、红区只跟上一张、chain 坐标不统一等）。

### 交付路径

- `/workspace/pcb-ar-annotate/docs/TECH_SPEC.md`（当前 **v0.3.1**）
- `/workspace/pcb-ar-annotate/docs/DATA_MODEL.md`（当前 **v0.3.1**）
- `/workspace/pcb-ar-annotate/docs/MILESTONES.md`（当前 **v0.3.1**）

### 协作约定

- 规格变更先改文档，再通知视觉/构建。  
- BRIEF / 旧文冲突时以 TECH_SPEC **v0.3.1** 覆盖。  
- 本文档轮次**只改 Markdown**；不改 `src/`、测试、`VERSION`、`PROJECT_BRIEF.md` 正文。

---

## 2. 视觉工程 Bot

**角色定位**：PyQt6 + OpenCV 实现、算法与测试。

### 空壳 — 已完成

脚手架、Capture、主窗口壳、双 Tracker 桩、测试计划、可运行入口。

### Mapping / Video Mosaic — 主切片已落地（现为非默认 / 离线选配）

`VideoMosaicBuilder` / `MappingController` / VIDEO-MAP-001 冒烟已完成。**保留代码**；**默认不再走全图累加**。不阻塞 D17。

### 拓扑切片（Live 主路径，D16）— **已落地（v0.2.1 勾选）**

1. [x] **`TopologyExtractor`**  
2. [x] **`topology_db` + `geo_fingerprint` 落盘**（`assets/topology/SAMPLE-PARTIAL-001.json`，`0.2-topo`）  
3. [x] **`ConstellationMatcher` / `GeometricLocalizer`**  
4. [x] **Live 最小叠加**（`C-LIVE` overlay）；无库明示「未设置拓扑库，无法定位」  
5. mosaic / 散图：仍不阻塞；现为离线选配 / 兜底  

已落地的 §3.8 拓扑 API **继续复用、无需因 v0.3.1 回改**。

### 本轮下一刀（建图主路径，D17）— **骨架已通，按对齐后的 §3.9 继续补齐工程接入与验收**

视觉**无需因 v0.3.1 改名**。不要再按已废弃的 `interval_scan.py` / `StabilityGate` / `IntervalScanController` / `TileGraph` 去改代码。

模块已有：

- `StableCaptureSession`：`start` / `observe` / `capture` / `render` / `chain_dict` / `topology_db`
- 辅助：`frame_sharpness`、`interframe_motion_px`、`binary_lines`、`frame_overlap_ratio`、`paint_red_coverage`、`save_session`
- 测试：`tests/test_stable_capture.py`
- 产物：`artifacts/stable_capture_map/`（3 张 `cap-000`..`002`，2 次拒绝：模糊 / 跟踪丢失，`full_canvas=false`）

仍待补齐（官方勾选仍为 `[ ]`）：

1. [ ] **工程与 UI 接入**：`ProjectStore` 的 `tile_graph` / `map_source=interval_tiles`；落盘到 `assets/scans/<board_id>/`；主窗口开始扫描 / 文案 / 红区 / 拒拍  
2. [ ] **全部已拍足迹并集红区**（本切片红区只跟上一张，并集为随后项，不要求本刀改实现）  
3. [ ] **矢量线条**（本切片未做；二值 PNG 已有）  
4. [ ] **统一逻辑坐标**与合并库的 Live 全板 localize（点目前在各瓦片本地系）  
5. [ ] 工程级冒烟与验收勾选  

### 交付路径（已有）

- `/workspace/pcb-ar-annotate/src/vision/topology.py`
- `/workspace/pcb-ar-annotate/src/vision/stable_capture.py`
- `/workspace/pcb-ar-annotate/assets/topology/`
- `/workspace/pcb-ar-annotate/artifacts/stable_capture_map/`
- `tests/test_topology.py`、`tests/test_stable_capture.py`

### 协作约定

- 不引入打包逻辑；ultralytics 不强制。  
- 状态色红/绿；气泡 ≤3。  
- **禁止**把「必须先拼出全景 mosaic」当成 Live 或默认建图前提。  
- **禁止**为红区或建图维护不断增大的全景像素画布。

---

## 3. 构建发布 Bot

**角色定位**：依赖、封装、发布说明。

### 首批任务

1. 打包计划：PyInstaller onedir；VERSION 单源（D7）。应用版本仍由本角色管理，**文档 v0.3.1 不改仓库根 `VERSION`**。  
2. 核心：`opencv-python`、`PyQt6`、`numpy`；pydantic 仅 dev（D13）。  
3. 资源外置：**topology JSON**、**scans / chain**、可选 mosaic、权重。  
4. Gate：可运行 + 冒烟通过前不正式打包。

### 交付路径

- `packaging/PACKAGING_PLAN.md`、`requirements*.txt`、可选 `.spec`

---

## 并行顺序建议

```
文档架构 ──► TECH_SPEC / DATA_MODEL / 清单（**v0.3.1 / D17 签名已对齐落地**；拓扑主切片已勾选）
     │
视觉工程 ──► [空壳✓] ──► [Video Mosaic 离线选配✓] ──► [拓扑 Live✓]
     │                                              ──► [D17 间隔拍照骨架✓]
     │                                              ──► [补齐工程接入与验收 ← 下一刀]
     │                                              ──► MVP 打磨
     │                      └── 散图兜底仍不阻塞
构建发布 ──► 计划 + requirements ──► [可运行 + 冒烟通过后] 打包试验
```

---

## 完成定义（本轮）

| 角色 | Done when |
|------|-----------|
| 文档架构 | TECH_SPEC + DATA_MODEL + 里程碑 + gaps（**v0.3.1**）已落盘；§3.9 与 `stable_capture.py` 对齐；D17 官方项未提前勾选 |
| 视觉工程 | 骨架已通；按对齐后的 §3.9 补齐工程接入与验收；无需因 v0.3.1 改名；§3.8 不回改 |
| 构建发布 | 计划与依赖模板就绪；暂不打包；拓扑与 scans/chain 资产外置 |

更新本文件状态时请注明日期（Asia/Shanghai）与负责人 Bot。
