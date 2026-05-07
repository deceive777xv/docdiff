"""Backup and restore application data."""
from __future__ import annotations

import os
import zipfile
from datetime import datetime
from pathlib import Path


def _config_path() -> Path:
    appdata = os.environ.get("APPDATA", str(Path.home()))
    return Path(appdata) / "DocDiffAgent" / "config.json"


def backup(data_dir: str, dest_dir: str) -> Path:
    """Create a timestamped zip of data_dir + config.json. Returns the zip path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(dest_dir) / f"DocDiffAgent_backup_{timestamp}.zip"
    data = Path(data_dir)
    config = _config_path()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if config.exists():
            zf.write(config, "config.json")
        for rel in ("app.db", "faiss", "parsed"):
            p = data / rel
            if p.is_file():
                zf.write(p, f"data/{rel}")
            elif p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file():
                        zf.write(child, f"data/{child.relative_to(data)}")

    return zip_path


def restore(zip_path: str, data_dir: str) -> None:
    """Restore from a backup zip, overwriting existing files."""
    config = _config_path()
    data = Path(data_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name == "config.json":
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_bytes(zf.read(name))
            elif name.startswith("data/") and not name.endswith("/"):
                rel = name[len("data/"):]
                dest = data / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
