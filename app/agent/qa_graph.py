"""LangGraph StateGraph for the QA (retrieval-augmented answering) workflow."""
from __future__ import annotations

import logging
from collections import Counter

from langchain_core.messages import AIMessage, BaseMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agent.sqlite_checkpointer import SQLiteCheckpointSaver
from app.agent.states import QAState
from app.core.retrieval.searcher import search
from app.db import compare_repo, document_repo

logger = logging.getLogger(__name__)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from app.core.types import Chunk, ChunkHit
serde = JsonPlusSerializer(allowed_msgpack_modules=[Chunk, ChunkHit])
_checkpointer = SQLiteCheckpointSaver(serde=serde)

_QA_SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据以下参考资料回答用户问题。

参考资料：
{context}

回答要求：
1. 只根据参考资料中的内容回答，不要编造信息
2. 如果参考资料中找不到答案，请明确说明"文档中未找到相关内容"
3. 引用具体章节或页码（如资料中有）
4. 回答简洁、准确
"""

_RISK_LABELS = {"high": "高风险", "medium": "中风险", "low": "低风险", "none": "无风险"}


def _route(state: QAState) -> str:
    return "end" if state.get("error") else "continue"


def _format_context(hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk
        ref = f"[{i}] "
        if chunk.section_path:
            ref += f"章节：{chunk.section_path}，"
        if chunk.page_no:
            ref += f"第{chunk.page_no}页，"
        ref += f"内容：{chunk.text}"
        parts.append(ref)
    return "\n\n".join(parts)


def _clip_text(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_compare_context(result, limit: int = 20) -> str:
    """Format persisted diff items as direct QA context for compare questions."""
    counts = Counter(item.diff_type for item in result.items)
    count_text = "，".join(f"{diff_type}{count}处" for diff_type, count in counts.items())
    parts = [
        "对比结果：",
        f"差异总数：{len(result.items)}" + (f"（{count_text}）" if count_text else ""),
    ]

    for index, item in enumerate(result.items[:limit], 1):
        risk = _RISK_LABELS.get(item.risk_level, item.risk_level)
        item_parts = [
            f"[{index}] 章节：{item.section_path or '未命名章节'}",
            f"类型：{item.diff_type}",
            f"风险：{risk}",
            f"相似度：{item.similarity_score:.3f}",
        ]
        if item.baseline_text:
            item_parts.append(f"基准：{_clip_text(item.baseline_text)}")
        if item.target_text:
            item_parts.append(f"目标：{_clip_text(item.target_text)}")
        if item.explanation:
            item_parts.append(f"说明：{_clip_text(item.explanation)}")
        parts.append("；".join(item_parts))

    if len(result.items) > limit:
        parts.append(f"其余 {len(result.items) - limit} 处差异未展开。")

    return "\n".join(parts)


def resolve_scope(state: QAState, config: RunnableConfig) -> dict:
    """Map scope string to concrete version_id list."""
    try:
        scope = state.get("scope", "current_doc")
        conn = config["configurable"]["conn"]

        if scope in ("current_doc", "compare"):
            ids = list(state.get("current_version_ids") or [])
            if not ids:
                label = "对比文档" if scope == "compare" else "当前文档"
                return {"error": f"{label}范围未指定版本，请先选择文档。", "status": "failed"}
            return {"_version_ids": ids, "status": "scope_resolved"}

        if scope == "standard_lib":
            docs = document_repo.list_documents(conn, source_type="standard")
            ids = [document_repo.list_versions(conn, d["id"])[0]["id"]
                   for d in docs
                   if document_repo.list_versions(conn, d["id"])]
            if not ids:
                return {"error": "标准文档库中没有可检索的文档。", "status": "failed"}
            return {"_version_ids": ids, "status": "scope_resolved"}

        # "all"
        ids = list(state.get("current_version_ids") or [])
        for doc in document_repo.list_documents(conn, source_type="standard"):
            versions = document_repo.list_versions(conn, doc["id"])
            if versions and versions[0]["id"] not in ids:
                ids.append(versions[0]["id"])
        if not ids:
            return {"error": "没有可检索的文档。", "status": "failed"}
        return {"_version_ids": ids, "status": "scope_resolved"}

    except Exception as e:
        logger.exception("resolve_scope failed")
        return {"error": str(e), "status": "failed"}


def retrieve_chunks(state: QAState, config: RunnableConfig) -> dict:
    """Vector search for relevant chunks and attach compare-result context."""
    try:
        conn = config["configurable"]["conn"]
        embedder = config["configurable"]["embedder"]
        compare_task_id = state.get("compare_task_id")
        try:
            hits = search(
                state["data_dir"],
                conn,
                state["question"],
                embedder,
                state["_version_ids"],
                top_k=5,
            )
        except Exception:
            if state.get("scope") == "compare" and compare_task_id:
                logger.warning("chunk retrieval failed for compare QA; using diff context", exc_info=True)
                hits = []
            else:
                raise

        result = {"_hits": hits, "status": "retrieved"}
        if state.get("scope") == "compare" and compare_task_id:
            diff_result = compare_repo.get_task_result(conn, compare_task_id)
            result["_compare_context"] = _format_compare_context(diff_result)
        return result
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        return {"error": str(e), "status": "failed"}


async def generate_answer(state: QAState, config: RunnableConfig) -> dict:
    """Generate answer via streaming LangChain model, accumulate into session messages."""
    try:
        hits = state.get("_hits", [])
        compare_context = state.get("_compare_context", "")
        if not hits and not compare_context:
            return {"answer": "文档中未找到与问题相关的内容。", "status": "answered"}

        lc_model = config["configurable"].get("lc_model")
        if not lc_model:
            return {"answer": "请先在设置页面配置模型", "status": "answered"}

        context_parts = []
        if compare_context:
            context_parts.append(compare_context)
        if hits:
            context_parts.append("检索片段：\n" + _format_context(hits))
        context = "\n\n".join(context_parts)
        system_msg = SystemMessage(content=_QA_SYSTEM_PROMPT.format(context=context))

        history = list(state.get("messages", []))[-6:]
        messages_to_send = [system_msg] + history

        chunks: list[BaseMessageChunk] = []
        async for chunk in lc_model.astream(messages_to_send):
            chunks.append(chunk)
        answer = "".join(c.content for c in chunks if isinstance(c.content, str))

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "answered",
        }
    except Exception as e:
        logger.exception("generate_answer failed")
        return {"error": str(e), "status": "failed"}


def attach_citations(state: QAState) -> dict:
    """Package chunk hits as citation list."""
    return {"citations": list(state.get("_hits", [])), "status": "completed"}


def _build_qa_graph():
    graph = StateGraph(QAState)
    graph.add_node("resolve_scope",    resolve_scope)
    graph.add_node("retrieve_chunks",  retrieve_chunks)
    graph.add_node("generate_answer",  generate_answer)
    graph.add_node("attach_citations", attach_citations)

    graph.set_entry_point("resolve_scope")
    graph.add_conditional_edges("resolve_scope",   _route, {"continue": "retrieve_chunks", "end": END})
    graph.add_conditional_edges("retrieve_chunks", _route, {"continue": "generate_answer",  "end": END})
    graph.add_conditional_edges("generate_answer", _route, {"continue": "attach_citations", "end": END})
    graph.add_edge("attach_citations", END)
    return graph.compile(checkpointer=_checkpointer)


qa_graph = _build_qa_graph()
