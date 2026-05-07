# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Doc-Diff-Agent
# Build: pyinstaller build/doc_diff_agent.spec
#
# Requirements (run from repo root):
#   pip install pyinstaller
#   pyinstaller build/doc_diff_agent.spec

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

# ── Data files ──────────────────────────────────────────────────────────────
datas = [
    (str(ROOT / "assets"), "assets"),
]

# PySide6 WebEngine resources
try:
    from PyInstaller.utils.hooks import collect_data_files as _cdf
    datas += _cdf("PySide6", subdir="Qt/resources")
    datas += _cdf("PySide6", subdir="Qt/translations")
except Exception:
    pass

# docling data files
try:
    datas += collect_data_files("docling")
except Exception:
    pass

# sentence-transformers / transformers cached tokenizer data
try:
    datas += collect_data_files("sentence_transformers")
    datas += collect_data_files("transformers")
except Exception:
    pass

# ── Hidden imports ──────────────────────────────────────────────────────────
hiddenimports = [
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebChannel",
    "sqlite3",
    "faiss",
    "cryptography",
    "openai",
    "langchain_openai",
    "langchain_core",
    "langgraph",
    "rank_bm25",
    "docx",
    "fitz",  # PyMuPDF
]
hiddenimports += collect_submodules("app")

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DocDiffAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icons" / "docdiff.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DocDiffAgent",
)
