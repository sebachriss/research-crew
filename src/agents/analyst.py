"""Analyst: sintetiza research_results acumulados y reporta gaps.

Solo recibe title + snippet + url + query en el prompt (no `content` completo).
"""
from datetime import UTC, datetime
from pathlib import Path

from src.llm import get_llm
from src.schemas import AnalystOutput
from src.state import AgentState, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _trace(level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node="analyst", level=level, text=text)  # type: ignore[arg-type]


def _format_results(results: list[dict]) -> str:
    """Convierte la lista de resultados a un bloque de texto compacto."""
    if not results:
        return "(no hay resultados)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['title']} — {r['url']}\n    snippet: {r['snippet']}\n    (de query: {r['query']})"
        )
    return "\n".join(lines)


def analyst_node(state: AgentState) -> dict:
    llm = get_llm()
    structured = llm.with_structured_output(AnalystOutput)
    prompt = _PROMPT.format(
        question=state["question"],
        research_rounds=state["research_rounds"],
        results=_format_results(state["research_results"]),
    )
    result: AnalystOutput = structured.invoke(prompt)
    return {
        "analysis": result.analysis,
        "gaps": result.gaps,
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace(
                "info",
                f"síntesis completa ({len(state['research_results'])} resultados, "
                f"{len(result.gaps)} gaps)",
            )
        ],
    }
