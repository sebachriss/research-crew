from unittest.mock import MagicMock, patch

from src.agents.supervisor import supervisor_node
from src.schemas import PlanOutput
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
