import asyncio
from unittest.mock import MagicMock, patch

from src.graph import build_graph
from src.schemas import AnalystOutput, EvalOutput, PlanOutput
from src.state import make_initial_state


def _fake_llm_sequence(*responses):
    """LLM mock cuyo .with_structured_output().invoke() devuelve responses en orden,
    y cuyo .astream() (usado por writer) yieldea chunks de texto fijos."""
    invocations = {"count": 0}

    def structured_invoke(prompt):
        idx = invocations["count"]
        invocations["count"] += 1
        return responses[idx]

    async def fake_astream(prompt):
        yield MagicMock(content="## TL;DR\n- punto [1]\n\n## Hallazgos clave\n### Uno\nblah [1]")

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=structured_invoke)
    llm.with_structured_output = MagicMock(return_value=structured)
    llm.astream = fake_astream
    # Keep .invoke too in case some test path still uses it; remove if unused.
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


def test_graph_completes_when_all_tavily_queries_fail():
    """Flujo degradado end-to-end: las 3 queries de Tavily lanzan excepción.
    El researcher las captura (gather con return_exceptions=True), el analyst
    recibe research_results=[], el writer produce un informe, y el grafo
    termina sin crashear con errores registrados en el estado."""
    state = make_initial_state("¿Q?")

    plan = PlanOutput(queries=["q1", "q2", "q3"])
    analyst = AnalystOutput(analysis="información insuficiente", gaps=["todo"])
    eval_ok = EvalOutput(enough_info=True, queries=[], reasoning="sin data disponible")

    fake_llm = _fake_llm_sequence(plan, analyst, eval_ok)

    def fake_search_raises(query, **kwargs):
        raise RuntimeError(f"Tavily caído para '{query}'")

    with (
        patch("src.agents.supervisor.get_llm", return_value=fake_llm),
        patch("src.agents.analyst.get_llm", return_value=fake_llm),
        patch("src.agents.writer.get_llm", return_value=fake_llm),
        patch("src.agents.researcher.TavilyClient") as mock_tv,
    ):
        mock_tv.return_value.search.side_effect = fake_search_raises
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke(state))

    assert final_state["report"] != ""
    assert final_state["research_results"] == []
    researcher_errors = [e for e in final_state["errors"] if e["node"] == "researcher"]
    assert len(researcher_errors) == 3
    assert final_state["iteration_count"] <= 8
