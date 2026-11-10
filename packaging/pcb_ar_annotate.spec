# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# PCB AR 标注 — PyInstaller 规格草稿（onedir）
# 路径：packaging/pcb_ar_annotate.spec
# 日期：2026-11-09（Asia/Shanghai）
# 对齐：docs/TECH_SPEC.md §6；packaging/PACKAGING_PLAN.md
#
# 【门禁 / 暂勿执行】
#   在可运行应用 + TECH_SPEC §6.1 冒烟四项（及约定测试）通过、且构建发布明确放行之前，
#   不要运行：  pyinstaller packaging/pcb_ar_annotate.spec
#   入口已存在：src/app/main.py（开发：PYTHONPATH=src python -m app.main）
#
# 用法（门禁通过后，在仓库根、venv 已装 requirements-dev.txt）：
#   pyinstaller packaging/pcb_ar_annotate.spec
# 产物：dist/pcb_ar_annotate/ （onedir 便携目录）；安装包/更新路径见 PACKAGING_PLAN §7（D7 预留）
#
# 版本单源：仓库根 VERSION（D7【已确认】）；当前 0.1.0；发版先改 VERSION 再同步 pyproject / __version__ / 关于框
# 对齐 TECH_SPEC v0.1.2 §6：产物须带版本；datas 可附带根目录 VERSION（门禁通过后启用）
# =============================================================================

import sys
from pathlib import Path

# 仓库根 = 本 spec 所在 packaging/ 的上一级
ROOT = Path(SPECPATH).resolve().parent  # noqa: SPECPATH 由 PyInstaller 注入
SRC = ROOT / "src"

# -----------------------------------------------------------------------------
# 入口：src/app/main.py
# 等价于开发态 `PYTHONPATH=src python -m app.main`
# 若日后改为 package 安装入口，可改 scripts 列表或改用 runpy 包装脚本。
# -----------------------------------------------------------------------------
entry_script = str(SRC / "app" / "main.py")

# 应用名 / onedir 目录名
APP_NAME = "pcb_ar_annotate"

block_cipher = None

# -----------------------------------------------------------------------------
# datas / binaries 占位
# - 大资源（全局图、Fiducial 模板、YOLO 权重）外置，见 PACKAGING_PLAN §6
#   与 TECH_SPEC §6「资源外置」；不要默认把 assets/maps、weights 打进包。
# - 若需随包极小占位（如空目录说明 README），可取消注释下方示例。
# datas 格式：[(src, dest_in_bundle), ...]
# binaries 格式：[(src, dest_in_bundle), ...]
# -----------------------------------------------------------------------------
datas = [
    # 示例（默认关闭）：随包附带简短说明，不附大图
    # (str(ROOT / "packaging" / "ASSETS_README.txt"), "."),
]

binaries = [
    # 若某台机器 collect_dynamic_libs('cv2') 仍漏 DLL，在此手动追加，例如：
    # (r"C:\\path\\to\\opencv_worldxxxx.dll", "."),
]

# -----------------------------------------------------------------------------
# hiddenimports：PyQt6 / 常见动态导入
# OpenCV 多数情况可被分析到；若运行报缺模块再追加 cv2 子模块。
# -----------------------------------------------------------------------------
hiddenimports = [
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    # 若启用 Qt 多媒体/OpenGL 等再打开：
    # "PyQt6.QtOpenGL",
    # "PyQt6.QtOpenGLWidgets",
    "cv2",
    "numpy",
    # src layout 下的包名（入口就绪后按实际子模块增减）
    "app",
    "app.main",
    "app.ui",
    "vision",
    "model",
]

# -----------------------------------------------------------------------------
# excludes：核心包不含检测栈（TECH_SPEC §6 / §8）
# -----------------------------------------------------------------------------
excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    # 常见无关重型库，减小体积（若代码未用可保持排除）
    "tkinter",
    "matplotlib",
    "scipy",
    "IPython",
    "notebook",
]

# -----------------------------------------------------------------------------
# Analysis
# pathex 含 src，使 app.* / vision.* 可被收集
# -----------------------------------------------------------------------------
a = Analysis(
    [entry_script],
    pathex=[str(SRC), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 可选：若缺 Qt 平台插件或 cv2 动态库，取消注释（体积显著增大）：
# from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
# _qt_datas, _qt_bins, _qt_hidden = collect_all("PyQt6")
# a.datas += _qt_datas
# a.binaries += _qt_bins
# a.hiddenimports += _qt_hidden
# a.binaries += collect_dynamic_libs("cv2")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------------
# EXE：onedir 模式下 console=False 为 GUI；调试首包时可临时 True 看控制台日志
# 不用 UPX（减少杀毒误报）
# -----------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir：二进制交给 COLLECT
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI；冒烟排障可改为 True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(ROOT / "packaging" / "icon.ico"),  # 有图标再启用
)

# -----------------------------------------------------------------------------
# COLLECT：onedir 便携目录 → dist/pcb_ar_annotate/
# TECH_SPEC §6 + D7【已确认】：先 onedir；预留 Inno/NSIS 等安装包与更新路径（本期不在此 spec 实现）
# -----------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# =============================================================================
# 备注
# 1. onefile：若日后需要，另写 spec 或改 EXE 为单文件模式并单独验收（SmartScreen）。
# 2. 资源：运行时读 PCB_AR_ASSETS 或可执行文件旁 assets/，勿依赖 sys._MEIPASS 放大图。
# 3. 本文件在无 src/app/main.py 时执行会失败——属预期；请先等视觉工程入口就绪。
# =============================================================================
