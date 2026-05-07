"""In-app update checker — compares current version against a remote version file."""
from __future__ import annotations

import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"

# URL to a plain-text file containing only the latest version string, e.g. "1.1.0"
_VERSION_URL = (
    "https://raw.githubusercontent.com/deceive777xv/doc-diff-agent/main/VERSION"
)


def _parse_version(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.strip().split("."))
    except ValueError:
        return (0,)


def check_for_update(timeout: int = 5) -> Optional[str]:
    """Return the latest version string if a newer version is available, else None."""
    try:
        with urllib.request.urlopen(_VERSION_URL, timeout=timeout) as resp:
            latest = resp.read().decode().strip()
    except Exception as exc:
        logger.debug("Update check failed: %s", exc)
        return None

    if _parse_version(latest) > _parse_version(APP_VERSION):
        return latest
    return None
