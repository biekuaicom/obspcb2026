# PCB 显微镜 AR 标注 — 里程碑检查清单（MILESTONES）

| 字段 | 值 |
|------|-----|
| 版本 | **v0.3.1** |
| 日期 | 2026-11-10（Asia/Shanghai） |
| 对齐 | `PROJECT_BRIEF.md`、`HANDOFF.md`、`TECH_SPEC.md` v0.3.1 |
| 用法 | 勾选 `[x]`；每阶段满足 **Done when** 再进入下一阶段 |
| 修订 | **v0.3.1**：D17 规格对齐落地 `stable_capture.py`；骨架已通但官方勾选仍保持未完成（待工程与 UI）。先前 **v0.3.0（D17）**：新增间隔拍照下一刀。先前 **v0.2.1**：勾选拓扑主项；§3.8 签名对齐落地。先前 **v0.2.0**：D16。先前 v0.1.4：勾选 Mapping 主项。先前 v0.1.3：D15。先前 v0.1.2 / v0.1.1：其余决策 |

---

## 本轮视觉工程应先实现（空壳 — 历史）

> **状态：已完成**。

- [x] 脚手架 `src/app`、`src/vision`、`src/model`、`tests/`
- [x] `VideoCaptureThread`（含无相机假帧/视频）
- [x] PyQt6 主窗口壳
- [x] `PCBAffineTracker` / `LocalGlobalTracker` 桩
- [x] 测试计划 + 可运行入口

---

## 已完成：拓扑抽取 + 指纹入库 + Live Constellation 匹配（主路径，D16）

对齐 `TECH_SPEC.md` §2.3–§2.4、**§3.8** 与 `DATA_MODEL.md` §2.1 / §2.7。

**状态：拓扑主切片已落地（2026-11-09 Asia/Shanghai）。**

### 可勾选任务

- [x] `TopologyExtractor`：灰度 + 自适应二值 → 轮廓/孔/焊盘点集（含 4K 降采样或 ROI）
- [x] `geo_fingerprint` / 板级 `topology_db` 写入（内嵌或 `assets/topology/<board_id>.json`）
- [x] 工程字段：`topology_db`；`ProjectStore` 加载/保存拓扑
- [x] `ConstellationMatcher` / `GeometricLocalizer`：帧点集 ↔ 库局部匹配 → 位姿
- [x] Live 最小叠加：锁定位姿后至少 1 个元件气泡/框（合成或样例帧）
- [x] 无 `topology_db` 时阻断/明确提示（失败降级）
- [x] 冒烟测试：抽取 → 入库 → localize 返回 ok + 投影

### Done when

1. 能从帧抽出几何点集；  
2. 指纹/拓扑库可落盘并被工程引用；  
3. matcher 在合成或样例场景给出位姿；  
4. Live（或回放）能叠至少 1 个标注；  
5. **不要求**先有 mosaic。

**Done when 1–5 均已满足**（冒烟 7/7 + overlay `C-LIVE`；`assets/topology/SAMPLE-PARTIAL-001.json` schema `0.2-topo`；**不要求** mosaic）。

**一句话切片**：`TopologyExtractor` + `topology_db` + `ConstellationMatcher` + Live 最小叠加。

---

## 本轮下一刀：稳定式间隔拍照建图（D17）

对齐 `TECH_SPEC.md` §2.1、**§3.9**（落地 `src/vision/stable_capture.py`）与 `DATA_MODEL.md` §2.8 / `map_source=interval_tiles`。

**状态（2026-11-10 Asia/Shanghai）：骨架已通**（`src/vision/stable_capture.py` + `tests/test_stable_capture.py` + `artifacts/stable_capture_map`），规格已跟落地 API 对齐；视觉无需因 v0.3.1 改名/改签。**工程接入 / 全已拍并集红区 / 矢量 / ProjectStore 的 `map_source` 等仍待对照后勾选。** 下列官方勾选保持 `[ ]`。

### 可勾选任务

- [ ] 稳定门限：清晰 + 稳定才允许拍（落地：`frame_sharpness` / `interframe_motion_px`，默认 `min_sharpness=40.0`、`max_motion_px=8.0`；无 `stable_hold_ms`）— **骨架已有**，待工程与 UI 勾选
- [ ] 首张：原图 + 二值 PNG + `TopologyExtractor` 抽点入库（矢量本切片未做）— **骨架已有**（`save_session` 写出 `{id}.png` / `{id}-binary.png`）
- [ ] Live 红区半透明叠加，随板移动 — **骨架已有，且仅跟踪上一张投影四边形**（不是全部已拍并集；并集为随后项）。禁止为红区维护全景像素画布
- [ ] 移板提示；后续拍照重叠门禁与失跟拒绝（`min_overlap=0.08`；文案已落地）— **骨架已有**，待 UI 接入勾选
- [ ] chain 落盘且运行时无全图（`tiles/` + `chain.json` + `topology_db.json` + `summary.full_canvas=false`；`map_source=interval_tiles`）— **模块级产物已有**（`artifacts/stable_capture_map/`，3 张 `cap-000`..`002`）；工程目录 `assets/scans/` 与 ProjectStore 接入仍缺
- [ ] 与 `topology_db` / `ConstellationMatcher` 同步：每拍抽点，`kind=stable_capture_chain`，点在各瓦片本地系，扫描中 matcher 改绑新瓦片 — **骨架已写出**；统一逻辑坐标 / 用合并库做 Live 全板 localize / 工程引用仍缺
- [ ] 冒烟：稳定门限、重叠/失跟拒绝、两张以上瓦片+边、上一张红区跟随、入库后对本张跟踪仍可用 — **模块测试与 artifacts 已有**（2 次拒绝：模糊 / 跟踪丢失）；工程与 UI 冒烟待勾选

### Done when

1. 不清晰或不稳定时拒绝拍照，须有上一帧确认稳定后允许首张落盘；  
2. 首张写出原图 + 二值，并抽拓扑点；矢量不作为本切片完成条件；  
3. 直播画面叠红色半透明区，且红区随电路板移动（本切片 = 上一张投影，非全景画布；并集不作为本刀完成条件）；  
4. 后续拍照在失跟或视口内红区不足时被拒绝并提示重拍；重叠且稳定时可落盘，并与上一张写 2×3 边；  
5. `chain.json` 可落盘；工程可通过 `tile_graph` 引用；运行时不生成超大全图像素画布（`full_canvas=false`）；  
6. 扫描过程中对新帧跟踪上一张仍可用；合并库的全板 Live localize 与 ProjectStore 接入列为随后验收，不阻塞本刀骨架；  
7. **不要求** Video Mosaic 全图。

当前：实现骨架已具备上述 1–5、7 的模块级证据，**在未见完整 ProjectStore / UI 接入前保持未勾选**，待工程与 UI 对照后勾选。

**一句话切片**：`StableCaptureSession`（`observe` / `capture` / `render` / `save_session`）+ 上一张红区跟踪 + chain 落盘 + 每拍同步点集。

---

## 非默认 / 离线选配：Mapping Mode / `VideoMosaicBuilder`（已落地可保留，D15→D16→D17）

> **角色（D17）**：**非默认 / 离线选配**。已落地主项勾选保留，代码可留；**默认不再走全图累加**。须显式选择才启用。普通电脑上全图累加可能卡死/报错。**不阻塞** D17。

### 可勾选任务（历史主切片 — 主项已完成）

- [x] `MappingController` 状态机
- [x] `VideoMosaicBuilder`：`push_frame` / `finalize` / `reset` / `cancel`
- [x] 写出 `assets/maps/<board_id>.png` + `map_source=video_mosaic` + `meta.mapping`（VIDEO-MAP-001）
- [x] 合成冒烟 8/7/1，画布 600×420
- [ ] 可选：`global_ref` 对照入口 — **未交付**
- [ ] 可选：mosaic 半自动抽特征 → 写入 `topology_db` — **待与拓扑 / 间隔拍照衔接**
- [ ] 旁路 Live `LocalGlobalTracker.set_global_map` 全联调 — **不阻塞**间隔拍照与拓扑主路径

**状态：旁路/离线主切片已落地（2026-11-09）；保留代码；默认建图改走间隔拍照（D17）；Live 默认走拓扑（D16）。**

---

## 并行/随后：散图兜底（非主路径）

同 v0.1.4：`SampleAssetImporter` / `SemiAutoStitchPipeline` / D8 zip；**不阻塞**间隔拍照建图与拓扑 Live。

### 并行记下（用户已确认，尚未写入本规格的官方完成条）

- 暗光 / 氧化鲁棒：目标工况为常规到中等变暗、轻度氧化；**极端 stress 仅为最坏情况，非目标工况**。
- Live 画面参数气泡（与既有「气泡 ≤3 核心参数」产品要求一致，产品化联调仍待官方勾选）。
- 仓库已有实现侧产物：`artifacts/topology_stress/`、`artifacts/topology_live_loop/`。**待对照规格勾选**；本稿不把未写入本规格验收条的项标为官方完成。
- **D17 随后项**：全部已拍足迹并集红区；chain 点变换到统一参考系并与 `logical_pos` / Live 全板匹配对齐；矢量线条；`ProjectStore` + `assets/scans/` + `map_source=interval_tiles` 工程接入。

---

## MVP（桌面可演示）— 间隔拍照建库 + 拓扑 Live

### 拓扑与数据

- [ ] `topology_db` 可加载；至少若干元件含 `logical_pos` + `geo_fingerprint` + `core_params`
- [ ] `ProjectStore` 读写工程 + 拓扑 + chain（`tile_graph`）；重开保持
- [ ] **默认建库入口**为间隔拍照扫描（`map_source=interval_tiles`，模块 `StableCaptureSession`），而不是默认 Video Mosaic
- [ ] （可选）离线 Video Mosaic / 散图对照可打开，标明非默认 / 离线选配

### 采集与 UI 壳

- [ ] `VideoCaptureThread` + Canvas + Dock
- [ ] 无相机降级（D6）；默认相机索引 0 / 近 4K（D6）
- [ ] 间隔拍照建库入口 + Topology Live 入口；无库提示
- [ ] 扫描中红区叠加、稳定/移板/拒拍文案（对齐 §2.1 原句）

### 匹配与叠加

- [ ] **主路径** `GeometricLocalizer` / `ConstellationMatcher` 可演示
- [ ] 视口内投影；中心外不渲染
- [ ] TTL **0.8 s**；超时冻结并提示（D4）
- [ ] 状态色红/绿（`doubt` 不进 MVP）

### 编辑与绑定

- [ ] 气泡 ≤3 `core_params`；侧栏 extended；status 写回

### 质量

- [ ] 冒烟：冷启动 → 间隔拍照建库（或载已有拓扑+chain 工程）→ Live/回放叠加 → 退出
- [ ] ≥15 FPS 记机名（D14）；参考机优先
- [ ] 运行时无超大全图像素画布；普通电脑扫描过程不因 mosaic 累加卡死

### MVP Done when

用户可：**开始间隔拍照扫描建库（稳定门限 + 重叠/失跟门禁 + chain 落盘）→ 拓扑库可用 → Live/回放几何匹配叠加 → 改核心参数与状态并保存 → 红绿正确**；帧率主观可接受。全景 mosaic **不是**演示前提，也**不是**默认建图路径。

---

## v0.5

- [ ] YOLO 种子（optional）
- [ ] 检测未装仍可启动
- [ ] mosaic 离线选配 Tracker 可切换对照
- [ ] 指纹维数/拒识策略迭代；氧化场景回归（极端 stress 非目标工况）
- [ ] 4K ROI/降采样达标有测量
- [ ] SQLite + 多板
- [ ] `doubt`（黄）
- [ ] （可选）离线 Mapping 漂移抑制增强
- [ ] 间隔扫描：全部已拍并集红区；统一逻辑坐标；矢量线条（若需）

### v0.5 Done when

种子检测可跑；拓扑匹配更稳；多板可用；性能有证据。

---

## v1

- [ ] VERSION + onedir 便携（D7）；冒烟通过
- [ ] 规划安装包/带版本更新路径
- [ ] 间隔拍照建库向导产品化；Video Mosaic / 散图作离线工具入口
- [ ] XYZ 标定原型 + 可选自动扫板
- [ ] 发布清单

### v1 Done when

可分发 Windows 便携包（带版本号）+ 间隔拍照建库 + 拓扑 Live 演示；检测仍可选。

---

## 文档阶段门禁

| 阶段 | 文档就绪标志 |
|------|----------------|
| 1 拓扑+数据模型+chain | `DATA_MODEL` 含 `topology_db` 与 chain / `tile_graph`；mosaic 角色=非默认/离线选配 |
| 2 间隔建图+Live 拓扑+叠加 | `TECH_SPEC` §2–§4 / §3.8 / §3.9 与落地 `stable_capture.py` 对齐 |
| 3 检测+UX | 气泡/侧栏/状态色规则明确 |
| 4 桌面封装 | `TECH_SPEC` §6 |
| 5 XYZ | 坐标解耦；扫板可选 |

---

## Gaps / Risks

| ID | 风险/缺口 | 影响 | 建议 | 状态 |
|----|-----------|------|------|------|
| R21 | **普通电脑 Video Mosaic 全图累加卡死/报错** | 无法默认建图 | D17：间隔拍照；运行时不生成全图画布 | **已确认（D17）** |
| R17 | **重复局部结构误匹配** | 标错元件 | 指纹+类型门控；低置信拒识 | 开放（D16） |
| R18 | **氧化孔缺失** | 匹配失败 | 自适应二值；多假设；不依赖全景 | 开放（D16） |
| R19 | **指纹维度【可迭代】** | 过拟合/欠区分 | 先 K 近邻相对向量 + 可选极坐标直方图 | 开放（可迭代） |
| R22 | **红区只跟上一张 / 跟踪丢失 / 重叠误拒** | 扫不下去或与「已拍并集」观感有差 | 本切片明确为上一张；并集随后；失败不盲目平移 | 开放（D17） |
| R23 | **chain 点不在统一坐标系** | 合并库不能直接 Live 全板匹配 | 边链变换到参考瓦片；工程接入后验收 | 开放（D17 随后） |
| R1 | 重复纹理拼接误匹配 | 离线 mosaic | 短基线；禁唯一 Stitcher | 开放（离线选配） |
| R1b | Mapping 长扫漂移 | 离线 mosaic | 重叠/Rescan；**不阻塞 Live / 间隔扫描** | 开放（离线选配） |
| R2 | 4K 全图过慢 | FPS | 降采样/ROI | 开放 |
| R5 | YOLO 误检 | 人工负担 | 仅种子 | 开放 |
| R8–R16 | 历史决策项 | — | D1–D15 已确认 | **已确认** |
| R20 | 文档仍写 mosaic 为 Live 前提 | 实现走偏 | **D16/D17 已确认**；BRIEF 以 v0.3.1 覆盖 | **已确认** |

---

## 待用户确认的决策点

1–14. **D1–D14** — **已确认**（2026-11-09 Asia/Shanghai）：同前版（矩阵方向、纯像素 MVP、旁路 Tracker 序、TTL、doubt、相机、VERSION、样例、status、轴向、工程目录、core 三键、手写校验、FPS/机名）。  
15. **D15** — **已确认**：曾引入 Mapping / video mosaic；**D17 后角色 = 非默认 / 离线选配**（非默认建图，非 Live 必经）。  
16. **D16** — **已确认**（2026-11-09 Asia/Shanghai）：  
    - 工程库以轻量 JSON/拓扑为主（id、类型、逻辑坐标、过孔/焊盘相对指纹、`core_params`）；  
    - Live：帧→灰度+自适应二值→轮廓/孔/焊盘→ Geometric Constellation / point-set 局部匹配→锁定位姿→叠气泡；  
    - 目标鲁棒新板/旧板/氧化烧黑；**不要求**像素级精确形变复原；  
    - **匹配主路径为拓扑**；字段 **`topology_db`** 与建图产出并存。本结论不被 D17 推翻。  
17. **D17** — **已确认**（2026-11-10 Asia/Shanghai）：  
    - **覆盖建图主路径**为稳定式间隔拍照（落地 `StableCaptureSession`）：稳定才拍；本切片红区跟踪**上一张**（并集为随后项）；后续拍照须留一小部分红区否则重拍；  
    - 每张局部存原图 + 二值 + 邻接 2×3 位姿；运行时**不**生成超大全图像素画布；  
    - 每拍抽点入 `topology_db`（`kind=stable_capture_chain`，点在各瓦片本地系）；`map_source` 默认 `interval_tiles`；  
    - Video Mosaic 全图累加降为**非默认 / 离线选配**。  
    - v0.3.1：规格签名与落盘 schema 跟实现对齐，视觉无需改名/改签。

**仍开放**：无阻塞决策（并集红区、统一逻辑坐标、矢量、工程接入标【可迭代】）。

---

## 角色门禁（本轮）

| 角色 | Done when |
|------|-----------|
| 文档架构 | TECH_SPEC + DATA_MODEL + 本清单（**v0.3.1**；D17 官方项未勾选、D16 主项已勾选；规格已跟 `stable_capture.py` 对齐）已落盘 |
| 视觉工程 | D17 **骨架已通**；按对齐后的 §3.9 继续补齐工程接入与验收；无需因 v0.3.1 改名；§3.8 不回改；Video Mosaic 保留为非默认/离线选配 |
| 构建发布 | 打包计划对齐；拓扑 JSON 与 scans/chain 外置；可运行+冒烟前暂不打包 |

更新状态时请注明日期（Asia/Shanghai）与负责人 Bot。
