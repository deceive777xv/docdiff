"""Tests for backup_service."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.services.backup_service import backup, restore


def _make_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "app.db").write_text("db-content")
    docs = data / "docs"
    docs.mkdir()
    (docs / "source.pdf").write_bytes(b"%PDF source")
    faiss = data / "faiss"
    faiss.mkdir()
    (faiss / "index.faiss").write_bytes(b"\x00\x01\x02")
    parsed = data / "parsed"
    parsed.mkdir()
    (parsed / "doc1.json").write_text('{"title":"test"}')
    return data


def test_backup_creates_zip(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{"providers":[]}')
    monkeypatch.setattr("app.services.backup_service._config_path", lambda: config)

    dest = tmp_path / "backups"
    dest.mkdir()
    zip_path = backup(str(data), str(dest))

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "config.json" in names
    assert "data/app.db" in names
    assert "data/docs/source.pdf" in names
    assert any(n.startswith("data/faiss/") for n in names)
    assert any(n.startswith("data/parsed/") for n in names)


def test_backup_skips_missing_config(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    monkeypatch.setattr(
        "app.services.backup_service._config_path",
        lambda: tmp_path / "nonexistent.json",
    )
    dest = tmp_path / "backups"
    dest.mkdir()
    zip_path = backup(str(data), str(dest))

    with zipfile.ZipFile(zip_path) as zf:
        assert "config.json" not in zf.namelist()


def test_restore_overwrites_data(tmp_path, monkeypatch):
    data = _make_data_dir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{"providers":[]}')
    monkeypatch.setattr("app.services.backup_service._config_path", lambda: config)

    dest = tmp_path / "backups"
    dest.mkdir()
    zip_path = backup(str(data), str(dest))

    # Corrupt the original data
    (data / "app.db").write_text("corrupted")

    restore(str(zip_path), str(data))

    assert (data / "app.db").read_text() == "db-content"
    assert config.read_text() == '{"providers":[]}'


def test_restore_rejects_zip_entries_outside_data_dir(tmp_path, monkeypatch):
    """Restore refuses data entries that would write outside the data directory."""
    data = _make_data_dir(tmp_path)
    config = tmp_path / "config.json"
    monkeypatch.setattr("app.services.backup_service._config_path", lambda: config)
    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/../evil.txt", "owned")

    with pytest.raises(ValueError, match="Unsafe backup entry"):
        restore(str(zip_path), str(data))

    assert not (tmp_path / "evil.txt").exists()
