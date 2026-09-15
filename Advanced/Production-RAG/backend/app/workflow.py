"""
workflow.py
-----------
The actual LangGraph workflow for this RAG system. This is not a
decorative dependency — the graph below is compiled and executed for
every query, with a typed state (`RAGState`) threaded through each node,
and the sequence of executed node names returned to the caller for
observability.

Graph shape:

    START
      |
    validate_query --(invalid)--> END
      |(valid)
    rewrite_query
      |
    retrieve_documents
      |
    evaluate_retrieval
      |
    rerank_improve
      |
    generate_answer
      |
    validate_answer
      |
     END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.config import get_settings
from app.generation import generate_answer as run_generation
from app.llm_client import LLMClient
from app.logging_config import get_logger
from app.models import RAGState
from app.prompts import build_rewrite_prompt
from app.reranking import rerank as run_rerank
from app.retrieval import retrieve as run_retrieve
from app.validation import validate_answer as run_answer_validation
from app.validation import validate_query as run_query_validation

logger = get_logger(__name__)


def _record(state: RAGState, node: str) -> None:
    state.setdefault("steps", [])
    state["steps"].append(node)
    logger.debug("[workflow] executed node=%s state_keys=%s", node, list(state.keys()))


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def node_validate_query(state: RAGState) -> RAGState:
    ok, err = run_query_validation(state["question"])
    state["query_valid"] = ok
    state["query_error"] = err
    _record(state, "validate_query")
    return state


def node_rewrite_query(state: RAGState) -> RAGState:
    client = LLMClient()
    question = state["question"]
    rewritten = question
    if client.is_configured():
        try:
            system, user = build_rewrite_prompt(question)
            candidate = client.chat(system, user, temperature=0.0, max_tokens=100)
            # guard against degenerate/empty rewrites
            if candidate and len(candidate) <= 500:
                rewritten = candidate.strip().strip('"')
        except Exception as exc:
            logger.warning("Query rewrite failed, using original question: %s", exc)
    state["rewritten_query"] = rewritten
    _record(state, "rewrite_query")
    return state


def node_retrieve_documents(state: RAGState) -> RAGState:
    settings = get_settings()
    top_k = state.get("top_k") or settings.retrieval_top_k
    retrieved = run_retrieve(
        state.get("rewritten_query") or state["question"],
        top_k=top_k,
        doc_ids=state.get("doc_ids"),
    )
    state["retrieved"] = retrieved
    _record(state, "retrieve_documents")
    return state


def node_evaluate_retrieval(state: RAGState) -> RAGState:
    retrieved = state.get("retrieved", [])
    # A simple, explainable sufficiency heuristic: at least one chunk
    # with reasonable similarity to the query.
    sufficient = any(r["similarity"] >= 0.15 for r in retrieved) if retrieved else False
    state["retrieval_sufficient"] = sufficient
    _record(state, "evaluate_retrieval")
    return state


def node_rerank_improve(state: RAGState) -> RAGState:
    settings = get_settings()
    retrieved = state.get("retrieved", [])
    reranked = run_rerank(
        state.get("rewritten_query") or state["question"],
        retrieved,
        top_k=settings.rerank_top_k,
    )
    state["reranked"] = reranked
    _record(state, "rerank_improve")
    return state


def node_generate_answer(state: RAGState) -> RAGState:
    client = LLMClient()
    context = state.get("reranked") or state.get("retrieved") or []
    if not client.is_configured():
        state["answer"] = (
            "⚠️ The LLM is not configured (missing LLM_API_KEY / LLM_BASE_URL / "
            "LLM_MODEL). No answer could be generated."
        )
        state["grounded"] = False
    else:
        try:
            answer, grounded = run_generation(client, state["question"], context)
            state["answer"] = answer
            state["grounded"] = grounded
        except Exception as exc:
            state["answer"] = f"⚠️ Answer generation failed: {exc}"
            state["grounded"] = False
    _record(state, "generate_answer")
    return state


def node_validate_answer(state: RAGState) -> RAGState:
    client = LLMClient()
    context = state.get("reranked") or state.get("retrieved") or []
    if client.is_configured() and state.get("grounded"):
        is_valid, notes = run_answer_validation(
            client, state["question"], context, state["answer"], state["grounded"]
        )
    else:
        is_valid, notes = True, "Validation skipped (LLM not configured or answer not grounded)."
    state["answer_valid"] = is_valid
    state["validation_notes"] = notes
    _record(state, "validate_answer")
    return state


# ---------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------

def route_after_validate_query(state: RAGState) -> str:
    return "rewrite_query" if state.get("query_valid") else "END"


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("validate_query", node_validate_query)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve_documents", node_retrieve_documents)
    graph.add_node("evaluate_retrieval", node_evaluate_retrieval)
    graph.add_node("rerank_improve", node_rerank_improve)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("validate_answer", node_validate_answer)

    graph.add_edge(START, "validate_query")
    graph.add_conditional_edges(
        "validate_query",
        route_after_validate_query,
        {"rewrite_query": "rewrite_query", "END": END},
    )
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "evaluate_retrieval")
    graph.add_edge("evaluate_retrieval", "rerank_improve")
    graph.add_edge("rerank_improve", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("validate_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_workflow(question: str, doc_ids: list[str] | None, top_k: int) -> RAGState:
    graph = get_graph()
    initial_state: RAGState = {
        "question": question,
        "doc_ids": doc_ids,
        "top_k": top_k,
        "steps": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state
