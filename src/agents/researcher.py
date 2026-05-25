"""Researcher: ejecuta búsquedas Tavily en paralelo. Sin LLM.

`TavilyClient.search` es sincrónico; lo envolvemos con `asyncio.to_thread`
para usarlo con `asyncio.gather`.
"""
import asyncio
import os

from tavily import TavilyClient

from src.agents._trace_util import trace
from src.config import TAVILY_MAX_RESULTS
from src.state import AgentState, NodeError, ResearchResult, TraceEvent


def _normalize(raw_result: dict, query: str) -> ResearchResult:
    """Pasa de un dict crudo de Tavily a ResearchResult."""
    return ResearchResult(
        url=raw_result.get("url", ""),
        title=raw_result.get("title", ""),
        content=raw_result.get("content", ""),
        snippet=raw_result.get("snippet") or (raw_result.get("content", "")[:200]),
        query=query,
    )


async def _search_one(client: TavilyClient, query: str) -> list[dict]:
    """Wrapper async sobre TavilyClient.search."""
    return await asyncio.to_thread(
        client.search,
        query,
        max_results=TAVILY_MAX_RESULTS,
        search_depth="basic",
    )


async def researcher_node(state: AgentState) -> dict:
    """Ejecuta las queries del state en paralelo con Tavily."""
    queries = state["queries"]
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    coros = [_search_one(client, q) for q in queries]
    results = await asyncio.gather(*coros, return_exceptions=True)

    new_results: list[ResearchResult] = []
    new_errors: list[NodeError] = []
    trace_events: list[TraceEvent] = []

    for query, result in zip(queries, results, strict=False):
        if isinstance(result, Exception):
            new_errors.append(
                NodeError(
                    node="researcher",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(result).__name__,
                    message=str(result),
                )
            )
            trace_events.append(trace("researcher", "warn", f"falló query '{query}': {type(result).__name__}"))
            continue

        raw_items = result.get("results", []) if isinstance(result, dict) else []
        for item in raw_items:
            new_results.append(_normalize(item, query))
        trace_events.append(trace("researcher", "info", f"query '{query}' devolvió {len(raw_items)} resultados"))

    diff: dict = {
        "research_results": new_results,
        "research_rounds": state["research_rounds"] + 1,
        "iteration_count": state["iteration_count"] + 1,
        "trace": trace_events,
    }
    if new_errors:
        diff["errors"] = new_errors
    return diff
