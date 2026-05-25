from unittest.mock import MagicMock, patch

from src.agents.analyst import analyst_node
from src.schemas import AnalystOutput
from src.state import make_initial_state


def _fake_llm_returning(structured_response):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(return_value=structured_response)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def test_analyst_returns_analysis_and_gaps():
    state = make_initial_state("¿Q?")
    state["research_results"] = [
        {"url": "uA", "title": "tA", "content": "cA", "snippet": "sA", "query": "qA"},
        {"url": "uB", "title": "tB", "content": "cB", "snippet": "sB", "query": "qB"},
    ]

    fake = _fake_llm_returning(
        AnalystOutput(analysis="síntesis textual", gaps=["gap1"])
    )
    with patch("src.agents.analyst.get_llm", return_value=fake):
        diff = analyst_node(state)

    assert diff["analysis"] == "síntesis textual"
    assert diff["gaps"] == ["gap1"]
    assert diff["iteration_count"] == 1
    assert "next_agent" not in diff
    assert diff["trace"][0]["node"] == "analyst"


def test_analyst_handles_empty_research_results():
    """Con research_results=[], el agente debe devolver disclaimer + gap original."""
    state = make_initial_state("¿Cuál es X?")
    state["research_results"] = []

    fake = _fake_llm_returning(
        AnalystOutput(analysis="no hay info suficiente", gaps=["¿Cuál es X?"])
    )
    with patch("src.agents.analyst.get_llm", return_value=fake):
        diff = analyst_node(state)

    assert diff["analysis"] == "no hay info suficiente"
    assert diff["gaps"] == ["¿Cuál es X?"]


def test_analyst_does_not_send_full_content_to_llm():
    """El prompt debe contener title/snippet/url, pero NO el content completo."""
    state = make_initial_state("¿Q?")
    long_content = "X" * 2000
    state["research_results"] = [
        {"url": "uA", "title": "tA", "content": long_content, "snippet": "snippet corto", "query": "qA"},
    ]
    captured_prompt = {}

    def capture_invoke(prompt):
        captured_prompt["text"] = prompt
        return AnalystOutput(analysis="x", gaps=[])

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=capture_invoke)
    llm.with_structured_output = MagicMock(return_value=structured)
    with patch("src.agents.analyst.get_llm", return_value=llm):
        analyst_node(state)

    assert long_content not in captured_prompt["text"], (
        "el prompt no debe incluir content completo"
    )
    assert "snippet corto" in captured_prompt["text"]
    assert "tA" in captured_prompt["text"]
    assert "uA" in captured_prompt["text"]
