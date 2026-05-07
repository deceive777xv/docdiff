"""Tests for update_checker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.update_checker import APP_VERSION, _parse_version, check_for_update


def test_parse_version():
    assert _parse_version("1.2.3") == (1, 2, 3)
    assert _parse_version("0.0.1") == (0, 0, 1)
    assert _parse_version("bad") == (0,)


def test_check_for_update_returns_new_version(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = b"99.0.0"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = check_for_update()

    assert result == "99.0.0"


def test_check_for_update_returns_none_when_current(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = APP_VERSION.encode()

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = check_for_update()

    assert result is None


def test_check_for_update_returns_none_on_network_error():
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        result = check_for_update()
    assert result is None
