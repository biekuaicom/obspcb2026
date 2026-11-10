# PCB AR 标注 — 打包与发布计划（PACKAGING PLAN）

> 版本：0.1.0（模板占位）  
> 日期：2026-11-09（Asia/Shanghai）  
> 角色：构建发布  
> 基线：`PROJECT_BRIEF.md`、`HANDOFF.md`「构建发布」小节、**`docs/TECH_SPEC.md` v0.1.2 §6「发布约定」**  
> **门禁：可运行应用 + 冒烟（§6.1）通过之前，禁止执行正式打包（勿跑 PyInstaller 产出安装包）。禁止无版本号产物（D7）。**

---

## 0. 与 TECH_SPEC 对齐

本计划**显式遵循** `docs/TECH_SPEC.md`：

| 规格位置 | 本计划落点 |
|----------|------------|
| **§6 发布约定（v0.1.2）** | 平台、产物形态、版本、核心/可选依赖、资源外置（下文 §2–§7） |
| **§6.1 冒烟验收项** | 下文 §10 与规格 **7 项**对齐（含无相机/有相机/FPS） |
| **§6.2 运行时依赖与已知限制** | 下文 §9 相机/PyQt6/OpenCV/杀毒/版本目录坑 |
| **§6.3 样例资产** | 资源外置含 samples 瓦片与 global_ref；不打进核心包 |
| **§5 文档阶段 4「桌面封装」** | 本文件 + requirements 模板 + `.spec` 草稿即本阶段构建侧交付 |
| **D7 / D13 / D14【已确认】** | 见 §3 / §4 / §7；`VERSION` 单源；先 onedir 并预留安装包/更新；pydantic 不进核心；参考机优先 + ≥15 FPS 记各机实测 |

规格中写的 `requirements-detect.txt` 与本仓库 **`requirements-optional.txt` 等价**（见 §4）；不另拆冲突的第二套 detect 文件。

---

## 1. 目标与非目标

### 1.1 本期目标

- 为桌面端（**Windows x64 优先**）准备可复现的依赖分层与 PyInstaller **onedir** 规格草稿。
- 核心运行时不含 YOLO/torch，控制分发体积（对齐 TECH_SPEC §6 / §8）。
- 全局图、Fiducial 模板、检测权重**外置**，不打进核心包。

### 1.2 非目标（本期）

- 手机端 / 移动端打包。
- 本期不交付正式安装包/代码签名/自动更新实现；**D7 已确认须预留**安装包与更新路径（见 §7），门禁通过后先出 onedir，再迭代安装包。
- 将 `ultralytics` / `torch` 打进默认核心产物。

---

## 2. 工具选型

| 选项 | 结论 | 说明 |
|------|------|------|
| **PyInstaller** | **首选** | 与 Python + PyQt6 + OpenCV 桌面栈匹配；本仓库提供 `packaging/pcb_ar_annotate.spec` 草稿 |
| Briefcase / cx_Freeze / Nuitka | 备选调研 | 暂不采用 |
| 模式起步 | **onedir** | 与 TECH_SPEC §6「先 PyInstaller onedir 便携版」一致；冷启动与 DLL/插件排查更友好 |
| onefile | **可选后期** | 单文件便于拷贝，但解压启动更慢、杀毒误报更高；需单独验收 |

推荐命令（**仅在门禁通过后**）：

```bash
# 在仓库根、已激活 venv、且 src 可导入之后
pyinstaller packaging/pcb_ar_annotate.spec
```

---

## 3. 目标平台

| 平台 | 优先级 | 用途 |
|------|--------|------|
| **Windows x64** | P0 | 正式分发目标；参考机 `DESKTOP-J4I5P0U`（优先冒烟/计时） |
| **Linux x64** | P1 | CI / 构建机 / 冒烟 |
| macOS | 暂缓 | 未列入本期 |
| 手机端 | 非目标 | 简报 §12 / TECH_SPEC §8 |

**【已确认 · D14】**：`DESKTOP-J4I5P0U` **优先参考、非唯一**。冒烟/性能须记录**机名与 FPS**；参考机优先达到跟踪链路 **≥15 FPS**（争取 30）。其他机型结果可并列，不得因未在该机跑过而单独否决。

跨平台注意：默认在**目标 OS 本机构建**（Linux 构建机不出 Windows 安装包，除非另行约定交叉方案）。

---

## 4. 依赖分层

仓库根三份模板（**未锁死精确版本**；首次可跑通后改为 `pip freeze` 锁定）：

| 文件 | 内容 | 对应 TECH_SPEC §6 |
|------|------|-------------------|
| `requirements.txt` | 核心：`opencv-python`、`PyQt6`、`numpy`（**不含** pydantic / ultralytics） | 「核心依赖」；**D13【已确认】** |
| `requirements-optional.txt` | 检测：`ultralytics`（torch 体积风险说明） | 「可选依赖」；**≡ 规格中的 `requirements-detect.txt`** |
| `requirements-dev.txt` | `-r requirements.txt` + `pytest` + `pyinstaller`；pydantic 可选注释 | 开发/测试/打包；**D13** 允许 pydantic 仅在此 |

### 4.1 `requirements-detect.txt` 命名说明

TECH_SPEC §6 表格使用名称 `requirements-detect.txt`。本仓库**统一文件名为** `requirements-optional.txt`（与 `HANDOFF.md` 首批任务一致），语义完全等价：按需安装 YOLO/torch，**不进核心包、不含检测权重**。

- 文档/对话中若写 `requirements-detect.txt`，请指向 `requirements-optional.txt`。
- **不要**再复制一份同内容的 `requirements-detect.txt`，以免双文件漂移。
- 若产品日后强要求文件名一致，可改为单文件重命名或一行 `-r requirements-optional.txt` 的薄包装（需另开一轮，本轮不做）。

安装示例：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt          # 开发/测试/打包工具
pip install -r requirements-optional.txt     # 仅需 YOLO 种子时（= detect）
```

未装 optional 时：主程序仍须可启动；检测入口灰显或提示（TECH_SPEC §6.2）。

---

## 5. 入口约定

与简报 / TECH_SPEC 模块清单一致：

- 开发入口：`PYTHONPATH=src python -m app.main`（亦支持 `--source synthetic`、`--smoke`）
- 布局：`src/app/main.py`（src layout）
- 版本展示/元数据以仓库根 `VERSION` 为准（D7）

### 5.1 开发时让 `app` 可导入

任选其一（视觉工程落地脚手架后固定一种）：

1. **PYTHONPATH**：
   ```bash
   # Linux / macOS
   export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
   python -m app.main

   # Windows PowerShell
   $env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
   python -m app.main
   ```
2. **可编辑安装**（中长期）：日后 `pip install -e .`（需 pyproject；本轮不强制创建）。
3. **PyInstaller**：`.spec` 的 `Analysis` 入口指向 `src/app/main.py`，`pathex` 含 `src`。

---

## 6. 资源外置策略

对齐 TECH_SPEC §6「资源外置」：以下**不**打进核心 onedir（或仅极小占位）：

| 资源 | 建议位置 | 配置方式 |
|------|----------|----------|
| PCB 全局高清图 | `assets/maps/` 或用户工程目录 | 工程 JSON 的 `global_map_path` |
| Fiducial 模板图 | `assets/fiducials/` | 应用配置 / 工程文件 |
| YOLO 权重 | `assets/weights/`（可选功能） | 检测模块路径；缺省禁用检测 |
| 样例瓦片 + global_ref | `assets/samples/<board_id>/`（TECH_SPEC §6.3） | 导入后外置；不打进核心包 |
| 工程 JSON | `assets/projects/` 或用户目录 | 相对路径解析（D11） |

查找顺序建议：

1. 环境变量 **`PCB_AR_ASSETS`**（assets 根绝对路径）
2. 可执行文件旁的 **`assets/`**（onedir 根旁）
3. 开发树仓库根下的 `assets/`

大文件用 Git LFS 或另行分发；勿默认塞进 PyInstaller `datas`。

---

## 7. 版本号与产物形态（D7【已确认】）

- 约定：**semver**（`MAJOR.MINOR.PATCH`），与 TECH_SPEC §6 一致
- **必须有版本号**，便于后续更新安装与发布对照
- **单源文件：仓库根 `VERSION`**（当前 `0.1.0`）。发版时先改 `VERSION`，再同步：
  - `pyproject.toml` → `[project].version`
  - `src/app/__init__.py` → `__version__`
  - 包内元数据 /「关于」对话框展示的版本字符串
  - 安装包/更新元数据（预留）
- **禁止无版本号产物**（TECH_SPEC §6 / D7）：便携目录建议带版本区分（如 `pcb_ar_annotate-0.1.0/` 或目录内附带 `VERSION`）
- **产物路径**：先交付 **PyInstaller onedir 便携版**；**预留**安装包（Inno/NSIS 等）与**带版本号**的更新安装路径，门禁通过并 onedir 冒烟稳定后再落地安装包
- 本期仍：只维护 onedir `.spec` + 计划层预留说明；**不**在门禁前执行正式打包

---

## 8. 门禁：暂不打包（写死）

**在同时满足以下条件之前，禁止执行正式 `pyinstaller` 构建与对外分发：**

1. 视觉工程交付可启动入口（`python -m app.main` 或等价）——主窗口可开。
2. TECH_SPEC **§6.1 冒烟项**在开发态通过（见 §10.1；至少 1–4、6；5/7 按硬件与性能笔记执行）。
3. 基础测试（pytest 或约定冒烟脚本）至少不红。
4. （建议）安装说明已并入 TECH_SPEC §6 或独立 `docs/INSTALL.md` 草稿可引用。

本仓库 `packaging/pcb_ar_annotate.spec` 为**草稿**：**无代码 / 未过冒烟时请勿执行。**

---

## 9. Windows / Linux 相机与常见打包坑

（对齐 TECH_SPEC §6.2）

### 9.1 相机权限与后端

| 平台 | 注意 |
|------|------|
| Windows | 隐私 → 相机；企业策略；OpenCV 常用 `CAP_DSHOW` / `CAP_MSMF`；索引可配置 |
| Linux | `video` 组或 `/dev/video*` 读权限 |

打开失败 → synthetic/录像回放（TECH_SPEC §6.1/§6.2 · D6），UI 提示「无相机」，**不得**启动即崩；默认相机索引 0，近 4K 失败则用实际分辨率。

### 9.2 PyQt6

- 平台插件：`platforms/qwindows.dll`（Win）/ `libqxcb.so`（Linux）；缺失 →「Could not find the Qt platform plugin」。
- `.spec`：`hiddenimports` 含 `PyQt6.QtWidgets` / `QtGui` / `QtCore`；必要时 `collect_all('PyQt6')` 作后备。
- 尽量避免非 ASCII 安装路径（Windows）。

### 9.3 OpenCV

- 原生库漏收 → `DLL load failed` / `ImportError`。
- 对策：`collect_dynamic_libs('cv2')` 或补 `binaries`；干净机冒烟。

### 9.4 路径

- 打包后勿依赖 CWD；资源用 §6 策略 + onedir 旁 `assets/`。
- 工程 JSON 内路径建议相对工程文件。

### 9.5 杀毒 / SmartScreen（Windows）

- 未签名易拦截；优先 onedir；本草稿默认不用 UPX；日后代码签名。

### 9.6 可选 torch / ultralytics

- 核心 `.spec` 的 `excludes` 排除 `torch`、`ultralytics`。
- 缺 optional **不得**导致主程序无法启动（§6.2）。

---

## 10. 冒烟清单与验收大纲

### 10.1 冒烟项（与 TECH_SPEC v0.1.2 §6.1 一致）

开发运行与打包后验收均适用：

1. **冷启动**无未捕获异常  
2. **打开主窗口**（视频区 + 侧栏 Dock 可见）  
3. **加载样例工程** JSON + 全局图（或 `global_ref` / mosaic）  
4. **无相机**：synthetic 或录像回放可出帧（D6）  
5. **有相机**（可选）：索引 0 打开；近 4K 请求失败时仍能用实际分辨率（D6）  
6. **正常退出**无崩溃（Capture `stop` 干净）  
7. **性能笔记**：记录机名与 FPS；参考机优先达跟踪链路 **≥15 FPS**（争取 30）（D14）  

> 无硬件 CI / 演示以第 4 项为硬门槛；第 5 项有相机时测。打包验收须带版本号（目录或内嵌 `VERSION`）。

### 10.2 正式打包后验收大纲（门禁打开之后）

1. 干净 Windows x64 解压**带版本标识**的 onedir（**优先**参考 `DESKTOP-J4I5P0U`；可在其他机复测）。
2. 执行 §10.1 全项（无相机环境至少 1–4、6；有相机加 5；性能记第 7 项：机名 + FPS，参考机目标 ≥15）。
3. 确认无「缺 Qt 插件 / 缺 cv2 DLL」。
4. 设置 `PCB_AR_ASSETS` 或旁路 `assets/`，样例工程 / samples 可读。
5. 确认默认包**不含** torch / pydantic；optional / dev 另测。
6. 核对包内版本与仓库 `VERSION` 一致（关于对话框或元数据）。
7. 记录杀毒/SmartScreen 情况，回写附录。

---

## 11. 与文档 / 视觉协作

| 协作方 | 等待 / 对齐项 |
|--------|----------------|
| **文档架构** | 已对齐 TECH_SPEC §6；规格变更后同步本计划；安装章可继续留在 §6 或拆 `docs/INSTALL.md` |
| **视觉工程** | `src/` 入口就绪 + §6.1 四项通过 → **通知构建发布开打包试验** |
| **产品 / 小明** | **D7 / D13 / D14【已确认】**（见 §3/§4/§7）；其余开放决策仍跟文档架构 |
| **构建发布** | 计划与模板已按确认项更新；**门禁未放行前仍不正式打包** |

---

## 12. 本轮交付物清单

| 路径 | 说明 |
|------|------|
| `/workspace/pcb-ar-annotate/packaging/PACKAGING_PLAN.md` | 本文件（对齐 TECH_SPEC §6） |
| `/workspace/pcb-ar-annotate/requirements.txt` | 核心依赖模板 |
| `/workspace/pcb-ar-annotate/requirements-optional.txt` | 检测可选（≡ 规格 `requirements-detect.txt`） |
| `/workspace/pcb-ar-annotate/requirements-dev.txt` | 开发/测试/打包工具 |
| `/workspace/pcb-ar-annotate/packaging/pcb_ar_annotate.spec` | PyInstaller 草稿（**暂勿执行**） |
| `/workspace/pcb-ar-annotate/VERSION` | semver 单源（D7） |

---

## 13. 修订记录

| 日期（Asia/Shanghai） | 版本 | 说明 |
|----------------------|------|------|
| 2026-11-09 | 0.1.0 | 首批：计划 + 依赖模板 + spec 草稿；门禁写死；对齐 TECH_SPEC §6 / §6.1；标注 D7、D14【待用户确认】；detect≡optional |
| 2026-11-09 | 0.1.0 | **D7/D13/D14【已确认】**：`VERSION` 单源 + 先 onedir 并预留安装包/更新；pydantic 不进核心；基准机非唯一、参考机优先并记 FPS；仍不正式打包 |
| 2026-11-09 | 0.1.0 | 对齐 TECH_SPEC **v0.1.2** §6：冒烟扩为 7 项（含 D6 无相机/有相机与 ≥15 FPS）；禁止无版本号产物；样例 assets §6.3 |
