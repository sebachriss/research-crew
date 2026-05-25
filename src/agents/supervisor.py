"""Supervisor del grafo. Único agente con LLM.

Se invoca en dos momentos:
1. Al arranque (estado vacío) → modo PLAN: descompone la pregunta en 3 sub-queries.
2. Después del Analyst (analysis seteado) → modo EVAL: decide si basta o pedir más research.

El código elige el modo según el estado; ambos prompts viven en `prompts/supervisor.txt`.
"""
from datetime import UTC, datetime
from pathlib import Path

from src.config import MAX_ITERATIONS, MAX_RESEARCH_ROUNDS
from src.llm import get_llm
from src.schemas import EvalOutput, PlanOutput
from src.state import AgentState, NodeError, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.txt"


def _load_sections() -> tuple[str, str]:
    """Lee el archivo de prompt y separa por '# MODE: PLAN' y '# MODE: EVAL'."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # Split por '# MODE:' header; ignora vacío inicial
    parts = text.split("# MODE: ")
    sections = {}
    for part in parts[1:]:
        name, body = part.split("\n", 1)
        body = body.strip()
        # Remove trailing separator (---) if present (used as a visual break in the .txt file)
        if body.endswith("---"):
            body = body[: -len("---")].rstrip()
        sections[name.strip()] = body
    return sections["PLAN"], sections["EVAL"]


_PLAN_TEMPLATE, _EVAL_TEMPLATE = _load_sections()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _trace(node: str, level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node=node, level=level, text=text)  # type: ignore[arg-type]


def supervisor_node(state: AgentState) -> dict:
    """Nodo supervisor. Captura excepciones para no crashear el grafo."""
    try:
        # MODO PLAN: arranque (sin queries ni resultados)
        if not state["queries"] and not state["research_results"]:
            return _run_plan(state)

        # MODO EVAL: después del analyst. Cada vez que hay un análisis fresco a evaluar.
        if state["analysis"]:
            return _run_eval(state)

        # No debería llegar aquí; safety fallback
        return {
            "next_agent": "writer",
            "iteration_count": state["iteration_count"] + 1,
            "trace": [_trace("supervisor", "warn", "estado inesperado, forzando writer")],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "next_agent": "writer",
            "enough_info": True,
            "iteration_count": state["iteration_count"] + 1,
            "errors": [
                NodeError(
                    node="supervisor",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            ],
            "trace": [_trace("supervisor", "error", f"falló: {type(exc).__name__}: {exc}")],
        }


def _run_plan(state: AgentState) -> dict:
    """Genera 3 sub-queries con LLM."""
    llm = get_llm()
    structured = llm.with_structured_output(PlanOutput)
    prompt = _PLAN_TEMPLATE.format(question=state["question"])
    result: PlanOutput = structured.invoke(prompt)
    return {
        "queries": result.queries,
        "next_agent": "researcher",
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace("supervisor", "info", f"PLAN: {len(result.queries)} sub-queries generadas")
        ],
    }


def _run_eval(state: AgentState) -> dict:
    """Evalúa si el análisis basta. Si no, genera follow-up queries."""
    # Caps: si ya estamos al límite, fuerza writer sin llamar LLM
    if (
        state["iteration_count"] >= MAX_ITERATIONS - 1
        or state["research_rounds"] >= MAX_RESEARCH_ROUNDS
    ):
        return {
            "enough_info": True,
            "next_agent": "writer",
            "iteration_count": state["iteration_count"] + 1,
            "trace": [_trace("supervisor", "warn", "cap alcanzado, forzando cierre")],
        }

    llm = get_llm()
    structured = llm.with_structured_output(EvalOutput)
    prompt = _EVAL_TEMPLATE.format(
        question=state["question"],
        analysis=state["analysis"],
        gaps="\n- " + "\n- ".join(state["gaps"]) if state["gaps"] else "(ninguno)",
        research_rounds=state["research_rounds"],
        max_rounds=MAX_RESEARCH_ROUNDS,
    )
    result: EvalOutput = structured.invoke(prompt)

    diff: dict = {
        "enough_info": result.enough_info,
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace(
                "supervisor",
                "info",
                f"EVAL: enough_info={result.enough_info} — {result.reasoning}",
            )
        ],
    }
    if result.enough_info:
        diff["next_agent"] = "writer"
    else:
        diff["queries"] = result.queries
        diff["next_agent"] = "researcher"
    return diff
