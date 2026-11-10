# 开发与运行环境

仓库**不包含**虚拟环境目录（`.venv`）。本地 `.venv` 约 558 MB、七千多个文件，且绑定具体操作系统和 Python 版本，直接上传别人也无法使用。请按下面步骤在本机创建。

## 要求

- Python 3.10 或更高（本仓库验证过 3.13.5）
- 能创建虚拟环境（标准库 `venv` 即可）
- 打开图形窗口需要显示器（`DISPLAY` 或 Wayland）。没有显示器时可用文末的 offscreen 冒烟命令

核心依赖只有 `numpy`、`opencv-python`、`PyQt6`。默认不要安装 `ultralytics` / `torch`（体积可能到数 GB）。打包交给单独流程，本仓库不附带安装包。

## 创建环境

在仓库根目录执行。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-dev.txt
```

Windows（PowerShell）：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt
```

只跑程序、不跑测试时，把最后一行换成：

```bash
pip install -r requirements.txt
```

也可以：

```bash
pip install -e ".[dev]"
```

想尽量复现本仓库验证过的核心与测试版本，改用锁定文件（仍不含 YOLO / torch）：

```bash
pip install -r requirements-lock.txt
```

## 可选：YOLO 检测

仅在需要自动检测种子时安装，且应在核心依赖之后单独装：

```bash
pip install -r requirements-optional.txt
```

`ultralytics` 会拉取 `torch` 等，体积很大。若有 NVIDIA GPU，请先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应的 `torch`，再安装 `ultralytics`，避免被默认 CPU 包覆盖。未安装时，主程序仍应能启动。

## 本仓库验证过的版本

记录于 `requirements-lock.txt`，对应 Python 3.13.5（Linux）：

| 包 | 版本 |
| --- | --- |
| numpy | 2.5.4 |
| opencv-python | 5.0.0.93 |
| PyQt6 / PyQt6-Qt6 | 6.12.0 |
| PyQt6_sip | 13.13.0 |
| pytest | 9.2.0 |

`requirements-dev.txt` 里的 `pyinstaller` 未装进上述验证环境，需要打包时再自行安装。

## 常用命令

均在仓库根目录、已激活虚拟环境时使用。

启动主窗口（默认摄像头 `0`；没有摄像头会改用合成帧，窗口仍打开）。打开后用工具栏或菜单「扫描建图…」（Ctrl+M）测试扫描生成整板图；没有摄像头时点「演示/合成帧」或「一键演示扫描」：

```bash
PYTHONPATH=src python -m app.main
```

强制合成帧：

```bash
PYTHONPATH=src python -m app.main --source synthetic
```

无显示器冒烟（收到第一帧后退出，成功时标准输出有 `smoke ok:`）：

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m app.main --smoke
```

无显示器跑通扫描并写出整板总览 `overview.png`（用样例 `SAMPLE-PARTIAL-001`，不需要摄像头）：

```bash
PYTHONPATH=src python -m app.main --demo-scan
```

也可以指定输出目录：

```bash
PYTHONPATH=src python -m app.main --demo-scan --out artifacts/scan_maps/demo
```

跑测试：

```bash
PYTHONPATH=src python -m pytest
```

稳定间隔拍照建图示例（需已有样例 `SAMPLE-PARTIAL-001`，会写出 `overview.png`）：

```bash
PYTHONPATH=src python -m vision.stable_capture \
  --global assets/samples/SAMPLE-PARTIAL-001/global_ref.png \
  --out artifacts/stable_capture_map
```

更多模块说明见仓库根目录 [README.md](../README.md)。规格与数据模型在 `docs/TECH_SPEC.md`、`docs/DATA_MODEL.md`。

## 常见问题

- **`python` 找不到或版本过低**：改用 `python3` 或 `py -3`，先执行 `python3 --version` 确认不低于 3.10。
- **创建 `venv` 失败**：Debian/Ubuntu 需安装 `python3-venv`（例如 `sudo apt install python3-venv`）。
- **窗口打不开 / Qt 平台插件报错**：确认有图形界面；无显示器时加 `QT_QPA_PLATFORM=offscreen`。
- **测试或导入找不到模块**：在仓库根目录执行，并设置 `PYTHONPATH=src`（`pyproject.toml` 已为 pytest 配置了 `pythonpath`）。
- **不要把 `.venv` 提交进 Git**：已在 `.gitignore` 中忽略。
