from unittest.mock import MagicMock, patch

from src.agents.supervisor import supervisor_node
from src.schemas import EvalOutput, PlanOutput
from src.state import make_initial_state


def _fake_llm_returning(structured_response):
    """Crea un mock que imita .with_structured_output(...).invoke(...)."""
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(return_value=structured_response)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def test_supervisor_plan_mode_generates_3_queries():
    state = make_initial_state("¿Cuál es el estado de la IA en Chile?")
    fake = _fake_llm_returning(PlanOutput(queries=["q1", "q2", "q3"]))
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["queries"] == ["q1", "q2", "q3"]
    assert diff["next_agent"] == "researcher"
    assert diff["iteration_count"] == 1
    assert len(diff["trace"]) == 1
    assert diff["trace"][0]["node"] == "supervisor"
    assert "PLAN" in diff["trace"][0]["text"]


def test_supervisor_eval_mode_enough_info_routes_to_writer():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "síntesis suficiente"
    state["gaps"] = []
    state["research_rounds"] = 1

    fake = _fake_llm_returning(EvalOutput(enough_info=True, queries=[], reasoning="ya basta"))
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert "queries" not in diff or diff.get("queries") == []
    assert "EVAL" in diff["trace"][0]["text"]


def test_supervisor_eval_mode_insufficient_routes_to_researcher_with_new_queries():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "síntesis incompleta"
    state["gaps"] = ["falta dato X", "falta dato Y"]
    state["research_rounds"] = 1

    fake = _fake_llm_returning(
        EvalOutput(enough_info=False, queries=["new1", "new2", "new3"], reasoning="faltan datos")
    )
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["enough_info"] is False
    assert diff["next_agent"] == "researcher"
    assert diff["queries"] == ["new1", "new2", "new3"]


def test_supervisor_eval_bypass_when_iteration_cap_reached():
    """Si iteration_count >= MAX_ITERATIONS - 1, fuerza writer SIN llamar LLM."""
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "algo"
    state["iteration_count"] = 7  # MAX_ITERATIONS - 1 = 7
    state["research_rounds"] = 1

    fake = _fake_llm_returning(EvalOutput(enough_info=False, queries=["a", "b", "c"], reasoning="x"))
    with patch("src.agents.supervisor.get_llm", return_value=fake) as mock_get_llm:
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert mock_get_llm.called is False, "no debería invocar LLM si caps alcanzados"
    assert "cap" in diff["trace"][0]["text"].lower()


def test_supervisor_eval_bypass_when_research_rounds_cap_reached():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "algo"
    state["iteration_count"] = 4
    state["research_rounds"] = 3  # MAX_RESEARCH_ROUNDS = 3

    fake = _fake_llm_returning(EvalOutput(enough_info=False, queries=["a", "b", "c"], reasoning="x"))
    with patch("src.agents.supervisor.get_llm", return_value=fake) as mock_get_llm:
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert mock_get_llm.called is False


def test_supervisor_catches_llm_exception_and_returns_safe_diff():
    state = make_initial_state("¿Q?")

    def raise_boom(*args, **kwargs):
        raise RuntimeError("LLM boom")

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=raise_boom)
    llm.with_structured_output = MagicMock(return_value=structured)

    with patch("src.agents.supervisor.get_llm", return_value=llm):
        diff = supervisor_node(state)

    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "supervisor"
    assert diff["errors"][0]["exception_type"] == "RuntimeError"
    assert diff["next_agent"] == "writer"
    assert diff["enough_info"] is True
    assert any(e["level"] == "error" for e in diff["trace"])
