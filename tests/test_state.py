# tests/test_state.py
from src.state import (
    AgentState,  # noqa: F401 (re-export check)
    NodeError,
    ResearchResult,
    TraceEvent,
    make_initial_state,
)


def test_research_result_typeddict_shape():
    r: ResearchResult = {
        "url": "https://x",
        "title": "T",
        "content": "C",
        "snippet": "S",
        "query": "q",
    }
    assert r["url"] == "https://x"

def test_trace_event_typeddict_shape():
    e: TraceEvent = {
        "timestamp": "2026-05-24T10:00:00",
        "node": "supervisor",
        "level": "info",
        "text": "hello",
    }
    assert e["level"] == "info"

def test_node_error_typeddict_shape():
    e: NodeError = {
        "node": "researcher",
        "iteration": 2,
        "exception_type": "TimeoutError",
        "message": "timeout",
    }
    assert e["iteration"] == 2

def test_make_initial_state_has_expected_defaults():
    s = make_initial_state("¿Qué es X?")
    assert s["question"] == "¿Qué es X?"
    assert s["queries"] == []
    assert s["enough_info"] is None
    assert s["next_agent"] == ""
    assert s["research_results"] == []
    assert s["analysis"] == ""
    assert s["gaps"] == []
    assert s["report"] == ""
    assert s["trace"] == []
    assert s["iteration_count"] == 0
    assert s["research_rounds"] == 0
    assert s["errors"] == []
