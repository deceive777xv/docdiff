"""Narrow cleanup for parser-generated Markdown artifacts."""
from __future__ import annotations

import re


_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_DISPIMG_PATTERN = re.compile(r"=DISPIMG\([^)]*\)", re.IGNORECASE)


def remove_dispimg_artifacts(md_text: str) -> str:
    """Remove ``=DISPIMG(...)`` noise from Markdown table cells only.

    Missing-value-looking strings such as ``NaN``, ``None`` and ``N/A`` are
    source content and are deliberately preserved.
    """
    result_lines: list[str] = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            result_lines.append(line)
            continue
        if _TABLE_SEPARATOR_PATTERN.match(stripped):
            result_lines.append(line)
            continue

        cells = line.split("|")
        processed_cells: list[str] = []
        for index, cell in enumerate(cells):
            if index == 0 or index == len(cells) - 1:
                processed_cells.append(cell)
                continue

            content = _DISPIMG_PATTERN.sub("", cell.strip())
            if cell.startswith(" ") and cell.endswith(" "):
                processed_cells.append(f" {content} ")
            elif cell.startswith(" "):
                processed_cells.append(f" {content}")
            elif cell.endswith(" "):
                processed_cells.append(f"{content} ")
            else:
                processed_cells.append(content)
        result_lines.append("|".join(processed_cells))

    return "\n".join(result_lines)
