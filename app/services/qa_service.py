"""QA service — retrieval-augmented question answering."""
from __future__ import annotations
import logging
import sqlite3

from app.core.model.base_provider import BaseProvider
from app.core.retrieval.context_format import format_hits_context
from app.core.retrieval.searcher import search
from app.core.types import ChunkHit, RetrievalScope
from app.db import document_repo

logger = logging.getLogger(__name__)

_QA_PROMPT = """你是一个专业的文档问答助手。请根据以下参考资料回答用户问题。

参考资料：
{context}

用户问题：{question}

回答要求：
1. 只根据参考资料中的内容回答，不要编造信息
2. 如果参考资料中找不到答案，请明确说明"文档中未找到相关内容"
3. 引用具体章节、页码或段落位置（如资料中有）
4. 回答简洁、准确
"""


def _resolve_version_ids(
    conn: sqlite3.Connection,
    scope: RetrievalScope,
    current_version_ids: list[str],
) -> list[str]:
    """Resolve scope to a list of version_ids to search."""
    if scope in (RetrievalScope.CURRENT_DOC, RetrievalScope.BASELINE, RetrievalScope.TARGET):
        return current_version_ids

    if scope == RetrievalScope.STANDARD_LIB:
        return [
            row["id"]
            for row in document_repo.list_latest_versions(conn, source_type="standard")
        ]

    if scope == RetrievalScope.ALL:
        ids = list(current_version_ids)
        for version in document_repo.list_latest_versions(conn, source_type="standard"):
            vid = version["id"]
            if vid not in ids:
                ids.append(vid)
        return ids

    return current_version_ids


def answer(
    conn: sqlite3.Connection,
    data_dir: str,
    question: str,
    provider: BaseProvider,
    embedder: BaseProvider,
    scope: RetrievalScope = RetrievalScope.CURRENT_DOC,
    current_version_ids: list[str] | None = None,
    top_k: int = 5,
) -> tuple[str, list[ChunkHit]]:
    """
    Answer a question using RAG.
    Returns (answer_text, list_of_chunk_hits_used_as_citations).
    """
    if current_version_ids is None:
        current_version_ids = []

    version_ids = _resolve_version_ids(conn, scope, current_version_ids)

    if not version_ids:
        return "没有可检索的文档。请先导入文档或选择检索范围。", []

    hits = search(data_dir, conn, question, embedder, version_ids, top_k=top_k)

    if not hits:
        return "文档中未找到与问题相关的内容。", []

    context = format_hits_context(hits)
    prompt = _QA_PROMPT.format(context=context, question=question)

    answer_text = provider.chat([{"role": "user", "content": prompt}])
    logger.info("QA answered question (scope=%s, hits=%d)", scope.value, len(hits))
    return answer_text, hits
