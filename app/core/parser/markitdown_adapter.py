"""MarkItDown compatibility parser for formats AnyDoc does not support."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.parser.markdown_cleanup import remove_dispimg_artifacts
from app.core.parser.markdown_ir import parse_markdown
from app.core.types import DocumentIR


def is_available() -> bool:
    try:
        import markitdown  # noqa: F401
        return True
    except ImportError:
        return False


def extract(
    file_path: str,
    llm_client=None,
    llm_model: str = "",
) -> DocumentIR:
    path = Path(file_path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix.lower() in {".md", ".markdown"}:
        markdown = path.read_text(encoding="utf-8")
    else:
        if not is_available():
            raise RuntimeError("markitdown is not installed")

        from markitdown import MarkItDown

        converter = MarkItDown(
            enable_plugins=bool(llm_client),
            llm_client=llm_client or None,
            llm_model=llm_model or None,
        )
        markdown = converter.convert(file_path).markdown

    cleaned_markdown = remove_dispimg_artifacts(markdown)
    return parse_markdown(cleaned_markdown, path.stem, file_hash)
