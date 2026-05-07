"""LangGraph StateGraph for the QA (retrieval-augmented answering) workflow."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.states import QAState
from app.core.retrieval.searcher import search
from app.db import document_repo

logger = logging.getLogger(__name__)

_checkpointer = MemorySaver()

_QA_SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据以下参考资料回答用户问题。

参考资料：
{context}

回答要求：
1. 只根据参考资料中的内容回答，不要编造信息
2. 如果参考资料中找不到答案，请明确说明"文档中未找到相关内容"
3. 引用具体章节或页码（如资料中有）
4. 回答简洁、准确
"""


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
    """Vector search for relevant chunks."""
    try:
        conn = config["configurable"]["conn"]
        embedder = config["configurable"]["embedder"]
        hits = search(
            state["data_dir"],
            conn,
            state["question"],
            embedder,
            state["_version_ids"],
            top_k=5,
        )
        return {"_hits": hits, "status": "retrieved"}
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        return {"error": str(e), "status": "failed"}


async def generate_answer(state: QAState, config: RunnableConfig) -> dict:
    """Generate answer via streaming LangChain model, accumulate into session messages."""
    try:
        hits = state.get("_hits", [])
        if not hits:
            return {"answer": "文档中未找到与问题相关的内容。", "status": "answered"}

        lc_model = config["configurable"].get("lc_model")
        if not lc_model:
            return {"answer": "请先在设置页面配置模型", "status": "answered"}

        context = _format_context(hits)
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
