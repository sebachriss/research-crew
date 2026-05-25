# tests/test_researcher.py
import asyncio
from unittest.mock import patch

import pytest

from src.agents.researcher import researcher_node
from src.state import make_initial_state


@pytest.fixture
def state_with_queries():
    s = make_initial_state("¿Q?")
    s["queries"] = ["query A", "query B", "query C"]
    return s


def _make_tavily_response(query_to_results: dict):
    """Helper: devuelve función mock que simula TavilyClient.search."""
    def fake_search(query, **kwargs):
        if isinstance(query_to_results.get(query), Exception):
            raise query_to_results[query]
        return {
            "results": query_to_results.get(query, []),
        }
    return fake_search


def test_researcher_executes_all_queries_in_parallel(state_with_queries):
    responses = {
        "query A": [{"url": "uA", "title": "tA", "content": "cA"}],
        "query B": [{"url": "uB", "title": "tB", "content": "cB"}],
        "query C": [{"url": "uC", "title": "tC", "content": "cC"}],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert len(diff["research_results"]) == 3
    queries_seen = {r["query"] for r in diff["research_results"]}
    assert queries_seen == {"query A", "query B", "query C"}
    assert diff["research_rounds"] == 1
    assert diff["iteration_count"] == 1
    assert "next_agent" not in diff, "researcher no debe escribir next_agent (arista fija)"


def test_researcher_handles_partial_failure(state_with_queries):
    responses = {
        "query A": [{"url": "uA", "title": "tA", "content": "cA"}],
        "query B": RuntimeError("Tavily 500"),
        "query C": [{"url": "uC", "title": "tC", "content": "cC"}],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert len(diff["research_results"]) == 2
    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "researcher"


def test_researcher_handles_total_failure(state_with_queries):
    responses = {
        "query A": RuntimeError("Tavily 500"),
        "query B": RuntimeError("Tavily 500"),
        "query C": RuntimeError("Tavily 500"),
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert diff["research_results"] == []
    assert len(diff["errors"]) == 3
    # Aún sigue al analyst (arista fija) — no se setea next_agent
    assert "next_agent" not in diff


def test_researcher_normalizes_result_shape(state_with_queries):
    responses = {
        "query A": [
            {"url": "uA", "title": "tA", "content": "C" * 500, "snippet": "snippetA"},
        ],
        "query B": [],
        "query C": [],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    r = diff["research_results"][0]
    assert set(r.keys()) == {"url", "title", "content", "snippet", "query"}
    assert r["query"] == "query A"
