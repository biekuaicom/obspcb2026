# PCB 显微镜 AR 标注 — 数据模型（DATA_MODEL）

| 字段 | 值 |
|------|-----|
| 版本 | **v0.3.1** |
| 日期 | 2026-11-10（Asia/Shanghai） |
| 状态 | 草稿；**D1–D17 已确认**；拓扑主切片已落地；D17 chain / `stable_capture` 规格对齐落地 |
| 配套 | `TECH_SPEC.md` v0.3.1、`PROJECT_BRIEF.md` §4 |
| 修订 | **v0.3.1**：D17 落盘 schema 对齐 `chain.json` / `stable_capture_chain`（2×3 `M`，点在各瓦片像素系）；视觉无需因文档改 schema。先前 **v0.3.0（D17）**：引入间隔拍照建图与 `map_source=interval_tiles`。先前 **v0.2.1**：板级拓扑 / `geo_fingerprint` 与落地 `0.2-topo` 对齐。先前 **v0.2.0**：D16 新增 `topology_db` 等。先前 v0.1.4 / v0.1.3 / v0.1.2 / v0.1.1 |

---

## 1. 实体总览

| 实体 | 含义 | MVP 载体 | v1 |
|------|------|----------|-----|
| **Board** | 一块 PCB 工程（**拓扑库** + **间隔拍照 chain** + 可选 mosaic + 标定 + 元件集） | 单文件 JSON | `boards` 表 |
| **TopologyDB** | 板级几何拓扑（点、指纹、索引） | 内嵌或 `assets/topology/<board_id>.json`；间隔拍照会话也可写出 `topology_db.json` | 可入库 |
| **CaptureChain** | 间隔拍照瓦片 + 邻接 2×3 位姿（D17 默认建图产出；落地文件名 `chain.json`） | 内嵌或路径；工程字段仍可用 `tile_graph` 指向该 chain | 可入库 |
| **Component** | 单个元件标注 + 几何指纹 | `components[]` | `components` 表 |
| **Fiducial** | 定位孔/基准点 | `fiducials[]` | `fiducials` 表 |
| **Session** | 核对/跟踪会话 | 可选 | `sessions` 表 |
| **MappingRun** | 离线选配 Video Mosaic 扫描摘要 | `meta.mapping` | 可入库 |
| **TileSet / Sample** | 散图 + 全局对照（兜底） | `assets/samples/` | 可入库 |

> v0.3.0 草案曾用类名 / schema 名 `TileGraph`、`0.3-tiles`、3×3 `M_rel`。**v0.3.1 以落地 `chain.json` 为准**，不再强制这些旧名。

---

## 2. 字段说明

### 2.1 Board（工程根）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `schema_version` | string | 建议 | 含间隔拍照 chain 时建议 `"0.3"`；读侧兼容 `"0.2"`（缺 `tile_graph` / chain 视为无间隔扫描图） |
| `board_id` | string | 是 | 稳定 ID |
| `name` | string | 否 | 显示名 |
| **`topology_db`** | object \| string | **Live 主依赖** | **内嵌**拓扑对象，或**相对工程目录**的路径，如 `../topology/<board_id>.json`（D11/D16）。间隔拍照会话也会单独写出 `topology_db.json`（`kind=stable_capture_chain`） |
| **`tile_graph`** | object \| string | **`map_source=interval_tiles` 时须可解析** | 指向 **chain**（`chain.json`）或直接内嵌 chain 对象，如 `../scans/<board_id>/chain.json`（D17）。名称沿用工程字段，内容对齐落地 chain，而不是 v0.3.0 的 0.3-tiles 草案 |
| `global_map_path` | string | **否（D16/D17）** | 可选 mosaic；相对工程目录；**不是**间隔扫描或 Live 的前提 |
| `global_ref_path` | string | 否 | 全局对照图（D8） |
| `map_source` | string | 否 | **`interval_tiles`**（**默认**，语义保留） \| `video_mosaic`（非默认 / 离线选配） \| `tile_stitch` \| `external_import` \| `none`。缺省且存在可解析 chain / `tile_graph` 时按 `interval_tiles` 理解。实现侧目录/产物名现为 `stable_capture` / `artifacts/stable_capture_map` |
| `pixel_per_mm` | `{x,y}` | 条件 | **D2** 可省略；UI 警告 |
| `fiducials` | Fiducial[] | 建议 | 旁路 Affine / 标定用 |
| `components` | Component[] | 是 | 可空数组 |
| `meta` | object | 否 | 见下；间隔扫描工程摘要接入待定【可迭代】，模块级摘要见 `summary.json` |

**`topology_db` ↔ `tile_graph`（chain）↔ `map_source` / `global_map_path`（D16 + D17）**

| 字段 | Live 角色 | 建图角色 |
|------|-----------|----------|
| `topology_db` | **主依赖**；无则不可进 Topology Live | 每张成功间隔拍照追加点集（本切片点在各瓦片本地系，`kind=stable_capture_chain`） |
| `tile_graph`（chain） | **非** Live 强依赖 | **默认建图产出**（`map_source=interval_tiles`） |
| `global_map_path` + `map_source=video_mosaic` | **可选旁路**（LocalGlobal 对照/应急） | **非默认 / 离线选配**；不要默认写超大 `assets/maps/*.png` |

`ProjectStore` 对 `tile_graph` / `map_source=interval_tiles` 的读写**尚未落地**，由后续工程接入。

### 2.2 Component

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `component_id` | string | 是 | 板内唯一，如 `R1` |
| `component_type` | string | 建议 | 如 `resistor` \| `capacitor` \| `ic` \| `connector` \| `via_cluster` \| `other`（【可迭代】） |
| `logical_pos` | `{x, y}` | 建议 | **拓扑/逻辑坐标**（板平面）。本切片间隔拍照点集尚未变换到统一参考系（见 §4），与 `logical_pos` 的对齐为随后项。若与 `absolute_pos_mm` / 历史 `base_pixel_box` 并存：逻辑匹配优先用 `logical_pos`，mm 仍服务机台（D2/D10） |
| `absolute_pos_mm` | `{x, y}` | 条件 | 物理 mm；D2 可省略 |
| `base_pixel_box` | `[x,y,w,h]` | 条件 | 可选；**仅当存在全局 mosaic 像素画布时**有意义。间隔扫描默认**没有**全局 mosaic 画布，**不依赖**本字段 |
| `geo_fingerprint` | object | **拓扑 Live 建议** | 周围过孔/焊盘相对几何指纹，见 §2.7【可迭代】 |
| `core_params` | object | 是 | 最多 3 键；默认 `name`/`value`/`package`（D12） |
| `extended_params` | object | 否 | 侧栏全量 |
| `status` | enum | 是 | 顶层权威（D9） |
| `meta` | object | 否 | `source`: `yolo_seed` \| `manual` \| `topology_seed` \| `interval_scan` \| `stable_capture` \| `import` |

### 2.3 Fiducial

同 v0.1.x：`id` / `pixel` / `mm` / `template_path`。有 mosaic 时 `pixel` 对齐 mosaic；纯拓扑 / 间隔拍照工程可用 `logical_pos` 等价字段（【可迭代】扩展）。

### 2.4 Session

同前；MVP 可不落。

### 2.5 TileSet / 样例 manifest（兜底）

同 v0.1.4；`role: "fallback_or_reference"`。与 D17 的 capture **chain** 不是同一实体：前者是散图 zip 对照，后者是间隔拍照默认建图产出。

### 2.6 MappingRun（`meta.mapping`，非默认 / 离线选配）

同 v0.1.4 落地字段（`mode`/`state`/帧计数/`canvas_size`/…）。**不**再作为 Live 或间隔扫描前置。仅当用户显式启用 Video Mosaic 全图累加时写入。

间隔拍照模块级摘要由 `save_session` 写成 `summary.json`（`shots` / `rejects` / `reject_reasons` / `overlap_mean` / `track_ms_median` / **`full_canvas`: false** 等）。工程 `meta` 如何引用该摘要【可迭代】，待 ProjectStore 接入。

### 2.7 TopologyDB 与 `geo_fingerprint`（D16，【可迭代】）

#### 2.7.1 板级 `topology_db` 对象（内嵌或独立文件）

单帧 / 对照图抽取对齐落地：`assets/topology/SAMPLE-PARTIAL-001.json`（`TopologyExtractResult.to_dict()`）。

间隔拍照会话另写出 `topology_db.json`，在 `0.2-topo` 形状上增加 chain 字段（见下）。**Live 读入既有 `from_dict` 仍只强依赖 `points` + `image_size` + `detect_scale`**；`kind=stable_capture_chain` 且 `image_size` 为 `null`、点分属各瓦片坐标系时，**不宜直接当作单一全板库做 Live localize**（随后项）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | string | **`"0.2-topo"`**（落地常量；D17 不改该值） |
| `board_id` | string | 与工程一致 |
| `kind` | string | 间隔拍照会话为 **`"stable_capture_chain"`**；单帧抽取文件可无此键 |
| `image_size` | object \| null | 单帧抽取为 `{ "width", "height" }`；**chain 库可为 `null`**（没有全板像素画布） |
| `source` | string | 来源路径或标签 |
| `detect_scale` | number | 单帧抽取工作尺度；chain 库可缺省 |
| `counts` | object | 单帧抽取：`{ "hole", "pad", "slot", "points" }`；chain 库可缺省 |
| `points` | array | 见下 |
| `edges` | array | chain 库会带上与 `chain.json` 相同的邻接边 |
| `slots` | array | 可选冗余（单帧 `to_dict` 会写出） |
| `note` | string | chain 库落地原句：**「点坐标在各瓦片自己的像素系里，没有全板像素画布」** |
| `coord_frame` / `built_at` / `meta` | — | 可选 |

**`points[]` 元素**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 单帧如 `pad-0246`。chain 库为 **`{tile_id}:{point_id}`**（如 `cap-000:slot-0001`） |
| `kind` | string | **`hole` \| `pad` \| `slot`** |
| `xy` | `[number, number]` | 单帧：该图坐标。**chain 库：该点所在瓦片自己的像素系**，不是统一参考系 |
| `area` / `circularity` | number | 同前 |
| `tile_id` | string | chain 库必有，如 `cap-000` |

> 读入兼容：`TopologyExtractResult.from_dict` 只依赖 `points` + `image_size` + `detect_scale`；旧草案若仍写 `x`/`y`/`via` 须迁移到本形状。Live **只强依赖**可解析的 `topology_db`（通常仍是单一坐标系的板级库）。

#### 2.7.2 `geo_fingerprint`（元件级，【可迭代】）

目标：用**周围过孔/焊盘相对几何**做局部可区分签名；不要求像素全景。  
落地辅助：`geo_fingerprint(points, anchor_xy, neighbors=6)`。

```json
{
  "kind": "constellation",
  "anchor_xy": [423.61, 213.53],
  "neighbors": [
    { "id": "hole-0268", "kind": "hole", "dxy": [-25.43, -0.62] },
    { "id": "pad-0246", "kind": "pad", "dxy": [-9.31, 25.01] }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `kind` | 固定 `"constellation"`（落地） |
| `anchor_xy` | `[x, y]` 元件锚点 |
| `neighbors` | K 近邻：`id` / `kind` / `dxy:[dx,dy]` |

【可迭代】可增：`polar_hist`、半径门控、类型权重等；读侧应容忍未知键。匹配主路径当前以板级点集星座 + RANSAC 相似变换为主，元件指纹作局部特征/种子。

### 2.8 Capture chain（D17，间隔拍照默认建图产出）

落地文件：**`chain.json`**（由 `StableCaptureSession.chain_dict()` / `save_session` 写出）。工程可用 `tile_graph` 指向该文件或内嵌同一对象。落地 **没有** 独立的 `schema_version` 字段，不必再写 `0.3-tiles`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `tiles` | array | 见下 |
| `edges` | array | 见下 |
| `rejects` | array | 被拒绝的 `capture()`：`reason`、`overlap`（可 null） |

**`tiles[]` 元素（chain 内为摘要；像素文件在 `tiles/`）**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 如 **`cap-000`**、`cap-001` |
| `image_size` | `{width, height}` | 该瓦片像素尺寸 |
| `points` | number | 该瓦片抽取到的点数（不是点数组；点数组在 `topology_db.json`） |

对应文件（相对会话输出目录）：

| 文件 | 说明 |
|------|------|
| `tiles/{id}.png` | 原图（必选） |
| `tiles/{id}-binary.png` | 二值/黑白线条 PNG（必选；本切片已做） |
| 矢量 | schema/产品可随后预留；**本切片未写出** |
| `coverage.png` | 可选；当前帧上一张红区叠加预览 |
| `summary.json` | 会话摘要，含 `full_canvas: false` |

**`edges[]` 元素**

| 字段 | 类型 | 说明 |
|------|------|------|
| `src` | string | 上一张瓦片 `id` |
| `dst` | string | 当前瓦片 `id` |
| `M` | number[2][3] | **2×3** 矩阵，与 `Pose2D.M` 同形 |
| `direction` | string | 固定 **`"previous_tile_to_current_frame"`**（上一张瓦片像素 → 当前帧） |
| `overlap` | number | 该次提交时的 `frame_overlap_ratio` |
| `track_ms` | number | 可选；跟踪耗时毫秒 |

首张无边。后续每张成功拍照与**上一张**建立一条边。运行时**不**从 chain 生成超大全图像素画布；`summary.full_canvas` 为 `false`。实现侧没有全图画布装配函数，规格亦不虚构。

---

## 3. status 枚举与画布颜色

| status | 含义 | 颜色 | MVP |
|--------|------|------|-----|
| `unchecked` | 待核对 | 红 `#E53935` | 必须 |
| `checked` | 已核对 | 绿 `#43A047` | 必须 |
| `doubt` | 存疑 | 黄 `#FDD835` | 不进 MVP（D5 → v0.5） |

---

## 4. 坐标系约定

### 4.1 `logical_pos` / 拓扑坐标（D16 主）

- 板平面逻辑坐标；默认与图像同向（x 右、y 下，D10）。
- Live：`ConstellationMatcher` 估计 **库逻辑 → 当前帧** 位姿后，将 `logical_pos` / 框投影到视口。
- **没有全局 mosaic 像素画布时不依赖 `base_pixel_box`**。

### 4.2 间隔拍照：各瓦片本地像素 + 2×3 邻接（D17 落地）

- 每张瓦片有自己的局部像素系（原点为该瓦片图像左上）。
- **本切片**写入 `topology_db` 的点**留在各瓦片自己的像素系**，用 `{tile_id}:{point_id}` 区分；`image_size` 可为 `null`；注明没有全板像素画布。
- `edges[].M`（2×3，`previous_tile_to_current_frame`）把上一张坐标映到拍摄该边时的当前帧，精神与 D1 / `Pose2D.M`（库 → 帧）一致。
- **本切片红区**只用上一张的四角经该 `M` 投影，不是全部已拍足迹并集（并集为【可迭代】随后项）。
- 把点沿边链变换到统一参考系（例如首张 `cap-000`），并与 `logical_pos` 对齐、供 Live 全板匹配，标为【可迭代】随后项；不要求本刀改实现。

### 4.3 `base_pixel_box`（可选 mosaic 像素）

- 仅当存在全局 mosaic 时有意义；原点 mosaic 左上；`[x,y,w,h]`。
- 旁路 LocalGlobal 时沿用 D1（全局→当前）。
- 默认间隔扫描路径下本字段可缺省。

### 4.4 `absolute_pos_mm`（D10 / D2）

- 长边横向 x、短边竖向 y；原点建议 F1；暂不预对齐机械系。
- MVP 可无 mm。

### 4.5 变换方向

- **拓扑 Live**：库逻辑坐标 → 当前帧像素（与 D1「库/全局 → 帧」精神一致）。
- **间隔扫描红区 / 邻接边**：上一张瓦片像素 → 当前帧（2×3 `M`）。
- **mosaic 旁路**（仅离线选配）：全局 mosaic 像素 → 当前帧（D1）。

---

## 5. `core_params` / `extended_params` 读写边界

同前：气泡 ≤3 键 `name`/`value`/`package`（D12）；侧栏 extended + status；顶层 status 权威（D9）。

---

## 6. JSON 示例

下列片段均为**示例**，风格对齐冒烟产物 `artifacts/stable_capture_map/`（DEMO / SAMPLE），不是工程目录里必须已存在的资产。

### 6.1 间隔拍照工程（默认建图，示例）

```json
{
  "schema_version": "0.3",
  "board_id": "DEMO-PCB-001",
  "name": "演示板",
  "topology_db": "../scans/DEMO-PCB-001/topology_db.json",
  "tile_graph": "../scans/DEMO-PCB-001/chain.json",
  "map_source": "interval_tiles",
  "components": [
    {
      "component_id": "R1",
      "component_type": "resistor",
      "logical_pos": { "x": 520.0, "y": 310.0 },
      "status": "unchecked",
      "core_params": {
        "name": "R1",
        "value": "10K",
        "package": "0805"
      },
      "extended_params": { "tolerance": "1%" },
      "geo_fingerprint": {
        "kind": "constellation",
        "anchor_xy": [520.0, 310.0],
        "neighbors": [
          { "id": "hole-0001", "kind": "hole", "dxy": [12.5, -8.0] },
          { "id": "pad-0014", "kind": "pad", "dxy": [-20.0, 5.0] }
        ]
      },
      "meta": {
        "source": "stable_capture",
        "updated_at": "2026-11-10T10:43:00+08:00"
      }
    }
  ],
  "meta": {
    "created_at": "2026-11-10T10:00:00+08:00",
    "updated_at": "2026-11-10T10:43:00+08:00"
  }
}
```

### 6.2 `chain.json` 片段（示例，对齐落地）

```json
{
  "tiles": [
    { "id": "cap-000", "image_size": { "width": 480, "height": 360 }, "points": 143 },
    { "id": "cap-001", "image_size": { "width": 480, "height": 360 }, "points": 158 },
    { "id": "cap-002", "image_size": { "width": 480, "height": 360 }, "points": 166 }
  ],
  "edges": [
    {
      "src": "cap-000",
      "dst": "cap-001",
      "M": [
        [0.999899, -0.000284, -47.888721],
        [0.000284, 0.999899, -0.041129]
      ],
      "overlap": 0.8953,
      "direction": "previous_tile_to_current_frame",
      "track_ms": 16.17
    }
  ],
  "rejects": [
    { "reason": "画面模糊，等待清晰", "overlap": null },
    { "reason": "未能跟上已拍区域，请退回红色覆盖处重拍", "overlap": null }
  ]
}
```

同目录还有 `tiles/cap-000.png`、`tiles/cap-000-binary.png`（以及 cap-001 / cap-002）、`topology_db.json`、`summary.json`（`"full_canvas": false`）、可选 `coverage.png`。无矢量文件。

### 6.3 间隔拍照 `topology_db.json` 片段（`kind=stable_capture_chain`，示例）

```json
{
  "schema_version": "0.2-topo",
  "board_id": "SAMPLE-PARTIAL-001",
  "source": "assets/samples/SAMPLE-PARTIAL-001/global_ref.png",
  "kind": "stable_capture_chain",
  "image_size": null,
  "points": [
    {
      "id": "cap-000:slot-0001",
      "kind": "slot",
      "xy": [249.89, 126.2],
      "area": 381.0,
      "circularity": 0.102,
      "tile_id": "cap-000"
    }
  ],
  "edges": [],
  "note": "点坐标在各瓦片自己的像素系里，没有全板像素画布"
}
```

### 6.4 `assets/topology/SAMPLE-PARTIAL-001.json` 片段（D16 落地单帧库，不变）

```json
{
  "schema_version": "0.2-topo",
  "board_id": "SAMPLE-PARTIAL-001",
  "image_size": { "width": 1080, "height": 1919 },
  "source": "assets/samples/SAMPLE-PARTIAL-001/global_ref.png",
  "detect_scale": 1.0,
  "counts": { "hole": 43, "pad": 293, "slot": 39, "points": 375 },
  "points": [
    { "id": "slot-0001", "kind": "slot", "xy": [673.59, 1856.44], "area": 1188.5, "circularity": 0.088 },
    { "id": "pad-0246", "kind": "pad", "xy": [414.3, 238.54], "area": 120.0, "circularity": 0.75 }
  ],
  "slots": [
    { "id": "slot-0001", "kind": "slot", "xy": [673.59, 1856.44], "area": 1188.5, "circularity": 0.088 }
  ]
}
```

### 6.5 无瓦片 chain、无 mosaic 的最小 Live 工程

```json
{
  "schema_version": "0.3",
  "board_id": "TOPO-ONLY-001",
  "topology_db": {
    "schema_version": "0.2-topo",
    "board_id": "TOPO-ONLY-001",
    "image_size": { "width": 0, "height": 0 },
    "detect_scale": 1.0,
    "counts": { "hole": 0, "pad": 0, "slot": 0, "points": 0 },
    "points": [],
    "source": "live_frames"
  },
  "map_source": "none",
  "components": [],
  "meta": { "updated_at": "2026-11-10T10:00:00+08:00" }
}
```

> Live 仍只检查 `topology_db` 可解析。本例无 chain，不能把 `map_source` 写成 `interval_tiles`。

### 6.6 离线 Video Mosaic 选配字段（示例，非默认）

若用户显式启用全图累加，可同时存在（不替代 `topology_db`，也不是 Live 前提）：

```json
{
  "global_map_path": "../maps/DEMO-PCB-001.png",
  "map_source": "video_mosaic",
  "meta": {
    "mapping": {
      "mode": "video_mosaic",
      "state": "MosaicReady",
      "frames_total": 8,
      "frames_used": 7,
      "frames_skipped": 1,
      "canvas_size": [600, 420],
      "overlap_hint": 0.85,
      "motion": "feature"
    }
  }
}
```

### 6.7 校验规则（增补）

| 规则 | 说明 |
|------|------|
| Live 主路径 | 加载后须能解析 `topology_db`（对象或可读文件） |
| `map_source=interval_tiles` | `tile_graph`（chain）必须可解析；`tiles` 至少 1 条，含 `id` + `image_size`；对应 `tiles/{id}.png` 与 `tiles/{id}-binary.png` 在工程接入后应能解析 |
| 邻接边 | 除首张外，成功瓦片应出现在 `edges` 中；`M` 为 2×3，`direction=previous_tile_to_current_frame` |
| `global_map_path` | 若存在则文件应存在；缺失不阻止拓扑 Live，也不阻止间隔扫描 |
| `map_source` | `interval_tiles` \| `video_mosaic` \| `tile_stitch` \| `external_import` \| `none`；默认指向间隔拍照 |
| `geo_fingerprint` | 落地为 `kind=constellation` + `anchor_xy` + `neighbors[]`；未知键容忍 |
| `core_params` / `status` | 同前（D9/D12） |
| 路径 | 工程字段相对**工程目录**解析（D11）；chain 所引用的瓦片文件相对**该 chain / 会话输出目录** |

---

## 7. 持久化策略

### 7.1 MVP：JSON + 可选独立拓扑 / chain 文件

- 工程：`assets/projects/<board_id>.json`
- 拓扑：内嵌 **或** `assets/topology/<board_id>.json`；间隔拍照会话亦写 `topology_db.json`
- **默认建图（工程目标）**：`assets/scans/<board_id>/`（`chain.json` + `tiles/{id}.png` + `tiles/{id}-binary.png` + `topology_db.json` + `summary.json`）
- **模块冒烟产物（已有）**：`artifacts/stable_capture_map/`（同上结构，3 张成功、2 次拒绝）
- 可选离线 mosaic：`assets/maps/`（**不要默认写出超大全图 PNG**）

### 7.2 v1：SQLite（增补列草案）

`boards` 增加 `topology_path TEXT` / `topology_json TEXT` / `chain_path TEXT`；`components` 增加 `component_type`、`logical_x`、`logical_y`、`fingerprint_json`。其余同 v0.1.4。

---

## 8. 校验（D13）

运行时手写最小校验；pydantic 仅 `requirements-dev`。`validate_project` 须检查：Live 场景下 `topology_db` 可解析；`map_source=interval_tiles` 时 chain / `tile_graph` 可解析（**ProjectStore 接入后**强制）。

---

## 9. 文件路径约定（D11 / D8 / D16 / D17）

```
pcb-ar-annotate/
├── assets/
│   ├── topology/                # 【Live 主依赖】板级拓扑 JSON <board_id>.json
│   ├── scans/<board_id>/        # 【默认建图，工程目标】
│   │   ├── chain.json
│   │   ├── topology_db.json
│   │   ├── summary.json
│   │   ├── coverage.png         # 可选
│   │   └── tiles/               # {id}.png + {id}-binary.png
│   ├── maps/                    # 【非默认 / 离线选配】mosaic 全图；不要默认写入
│   ├── samples/<board_id>/      # 对照/兜底/扫描
│   ├── templates/
│   ├── projects/                # 工程 JSON
│   ├── weights/
│   └── db/
├── artifacts/
│   └── stable_capture_map/      # 模块冒烟产物（已有）
├── VERSION
├── src/vision/stable_capture.py
└── docs/...
```

| 约定 | 说明 |
|------|------|
| Live 主依赖 | `topology_db` → `assets/topology/` 或内嵌（单一坐标系板级库） |
| 默认建图 | `tile_graph` 指向 `assets/scans/<board_id>/chain.json`；`map_source=interval_tiles` |
| 实现产物名 | 模块/冒烟目录现为 `stable_capture` / `artifacts/stable_capture_map` |
| mosaic 全图 | 非默认 / 离线选配；`map_source=video_mosaic` |
| 散图源 | `E:\局部图片.zip`；对照/兜底（D8） |
| 解析根 | 工程字段相对**工程目录**（D11）；瓦片文件相对会话输出目录 |

---

## 10. 与简报 / 前版差异

1. **D17 / v0.3.1**：默认建图为稳定式间隔拍照；落盘对齐 `chain.json` + `kind=stable_capture_chain`；边为 2×3 `M`；点在各瓦片像素系；运行时不生成超大全图像素画布；Video Mosaic 为非默认/离线选配。  
2. **D16 仍有效**：工程库以拓扑为主；mosaic 非 Live 必经；Live 只强依赖可解析的（通常是单一坐标系的）`topology_db`。  
3. 工程字段仍用 `tile_graph` 指向 chain；`map_source` 默认 `interval_tiles`（语义），实现目录名为 `stable_capture`。  
4. 无全局 mosaic 画布时不依赖 `base_pixel_box`。统一参考系 / `logical_pos` 对齐为随后项。  
5. D1–D16 确认项保留；D15/D16 对 mosaic 角色被 D17 收窄为离线选配。  
6. `PROJECT_BRIEF` / 旧文冲突时以 TECH_SPEC/DATA_MODEL **v0.3.1** 为准。  
7. 单帧拓扑点 / `geo_fingerprint` 形状仍对齐落地 `0.2-topo`（v0.2.1，D17 不改）。

**仍开放决策**：无（并集红区、统一逻辑坐标、矢量、工程接入等标【可迭代】）。
