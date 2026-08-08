"""AnyDoc-based parser for supported non-PDF document formats."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.parser.markdown_cleanup import remove_dispimg_artifacts
from app.core.parser.markdown_ir import parse_markdown
from app.core.types import DocumentIR


def is_available() -> bool:
    try:
        import anydoc  # noqa: F401
        return True
    except ImportError:
        return False


def extract(file_path: str) -> DocumentIR:
    path = Path(file_path)
    if not is_available():
        raise RuntimeError("firecrawl-anydoc is not installed")

    import anydoc

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    markdown = anydoc.to_markdown(str(path))
    cleaned_markdown = remove_dispimg_artifacts(markdown)
    return parse_markdown(cleaned_markdown, path.stem, file_hash)
