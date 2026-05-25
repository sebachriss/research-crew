import asyncio
from unittest.mock import MagicMock, patch

from src.graph import build_graph
from src.schemas import AnalystOutput, EvalOutput, PlanOutput
from src.state import make_initial_state


def _fake_llm_sequence(*responses):
    """LLM mock cuyo .with_structured_output().invoke() devuelve responses en orden,
    y cuyo .invoke() (sin estructurar, usado por writer) devuelve una respuesta de texto fija."""
    invocations = {"count": 0}

    def structured_invoke(prompt):
        idx = invocations["count"]
        invocations["count"] += 1
        return responses[idx]

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=structured_invoke)
    llm.with_structured_output = MagicMock(return_value=structured)
    llm.invoke = MagicMock(
        return_value=MagicMock(content="## TL;DR\n- punto [1]\n\n## Hallazgos clave\n### Uno\nblah [1]")
    )
    return llm


def _fake_tavily_response():
    def fake_search(query, **kwargs):
        return {
            "results": [
                {"url": f"https://x/{query}", "title": f"t-{query}", "content": "c", "snippet": "s"}
            ]
        }
    return fake_search


def test_graph_single_round_ends_in_writer():
    """Flujo feliz: supervisor PLAN -> researcher -> analyst -> supervisor EVAL (enough_info=True) -> writer -> END"""
    state = make_initial_state("¿Q?")

    plan = PlanOutput(queries=["q1", "q2", "q3"])
    analyst = AnalystOutput(analysis="síntesis ok", gaps=[])
    eval_ok = EvalOutput(enough_info=True, queries=[], reasoning="ya basta")

    fake_llm = _fake_llm_sequence(plan, analyst, eval_ok)
    fake_search = _fake_tavily_response()

    with (
        patch("src.agents.supervisor.get_llm", return_value=fake_llm),
        patch("src.agents.analyst.get_llm", return_value=fake_llm),
        patch("src.agents.writer.get_llm", return_value=fake_llm),
        patch("src.agents.researcher.TavilyClient") as mock_tv,
    ):
        mock_tv.return_value.search.side_effect = fake_search
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke(state))

    assert final_state["report"] != ""
    assert "## TL;DR" in final_state["report"]
    assert final_state["iteration_count"] <= 8
    assert final_state["research_rounds"] == 1


def test_graph_re_research_then_close():
    """Flujo con re-research: supervisor PLAN -> researcher -> analyst -> supervisor EVAL (insufficient) ->
    researcher -> analyst -> supervisor EVAL (enough) -> writer -> END"""
    state = make_initial_state("¿Q?")

    plan = PlanOutput(queries=["q1", "q2", "q3"])
    analyst1 = AnalystOutput(analysis="parcial", gaps=["falta X"])
    eval_insufficient = EvalOutput(
        enough_info=False, queries=["new1", "new2", "new3"], reasoning="falta info"
    )
    analyst2 = AnalystOutput(analysis="ahora sí completo", gaps=[])
    eval_ok = EvalOutput(enough_info=True, queries=[], reasoning="ya")

    fake_llm = _fake_llm_sequence(plan, analyst1, eval_insufficient, analyst2, eval_ok)
    fake_search = _fake_tavily_response()

    with (
        patch("src.agents.supervisor.get_llm", return_value=fake_llm),
        patch("src.agents.analyst.get_llm", return_value=fake_llm),
        patch("src.agents.writer.get_llm", return_value=fake_llm),
        patch("src.agents.researcher.TavilyClient") as mock_tv,
    ):
        mock_tv.return_value.search.side_effect = fake_search
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke(state))

    assert final_state["report"] != ""
    assert final_state["research_rounds"] == 2
    assert final_state["enough_info"] is True
