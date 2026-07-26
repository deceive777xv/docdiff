from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "build" / "doc_diff_agent.spec"
FA_BUNDLE_DIR = Path("assets/fonts/fontawesome-free-7.2.0-desktop")


def _load_spec_datas(monkeypatch) -> list[tuple[str, str]]:
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda *args, **kwargs: []
    hooks.collect_submodules = lambda *args, **kwargs: []

    pyinstaller = ModuleType("PyInstaller")
    utils = ModuleType("PyInstaller.utils")
    pyinstaller.utils = utils
    utils.hooks = hooks

    monkeypatch.setitem(sys.modules, "PyInstaller", pyinstaller)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", utils)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)

    captured: dict[str, list[tuple[str, str]]] = {}

    def analysis(*args, **kwargs):
        captured["datas"] = kwargs["datas"]
        return SimpleNamespace(
            pure=[],
            zipped_data=[],
            scripts=[],
            binaries=[],
            zipfiles=[],
            datas=kwargs["datas"],
        )

    runpy.run_path(
        str(SPEC_PATH),
        init_globals={
            "SPECPATH": str(SPEC_PATH.parent),
            "Analysis": analysis,
            "PYZ": lambda *args, **kwargs: object(),
            "EXE": lambda *args, **kwargs: object(),
            "COLLECT": lambda *args, **kwargs: object(),
        },
    )
    return captured["datas"]


def _expand_bundle_paths(datas: list[tuple[str, str]]) -> set[Path]:
    bundled: set[Path] = set()
    for source_value, destination_value in datas:
        source = Path(source_value)
        destination = Path(destination_value)
        if source.is_dir():
            bundled.update(
                destination / file.relative_to(source)
                for file in source.rglob("*")
                if file.is_file()
            )
        elif source.is_file():
            bundled.add(destination / source.name)
    return bundled


def test_fontawesome_bundle_contains_only_runtime_font_and_license(monkeypatch):
    bundled = _expand_bundle_paths(_load_spec_datas(monkeypatch))
    fontawesome_files = {
        path.relative_to(FA_BUNDLE_DIR)
        for path in bundled
        if path.is_relative_to(FA_BUNDLE_DIR)
    }

    assert fontawesome_files == {
        Path("LICENSE.txt"),
        Path("otfs/Font Awesome 7 Free-Solid-900.otf"),
    }


def test_non_fontawesome_runtime_assets_remain_bundled(monkeypatch):
    bundled = _expand_bundle_paths(_load_spec_datas(monkeypatch))

    assert Path("assets/diff_template.html") in bundled
    assert Path("assets/icons/docdiff.svg") in bundled
    assert any(
        path.suffix == ".ttf"
        for path in bundled
        if path.is_relative_to(Path("assets/fonts/HarmonyOS_Sans"))
    )
