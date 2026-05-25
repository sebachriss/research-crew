"""Writer: genera el informe final en Markdown.

Post-procesamiento: la sección 'Fuentes' la construye este módulo (no el LLM)
a partir de las citas `[n]` que efectivamente aparecen en el texto generado.
Esto evita alucinación de URLs.
"""
import re
from datetime import UTC, datetime
from pathlib import Path

from src.llm import get_llm
from src.state import AgentState, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer.txt"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _trace(level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node="writer", level=level, text=text)  # type: ignore[arg-type]


def _format_sources_for_prompt(results: list[dict]) -> str:
    if not results:
        return "(no hay fuentes)"
    return "\n".join(
        f"[{i}] {r['title']} - {r['url']}" for i, r in enumerate(results, 1)
    )


def extract_cited_indices(text: str) -> list[int]:
    """Devuelve los índices únicos de citas [n] en el orden en que aparecen."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def build_sources_section(cited_indices: list[int], results: list[dict]) -> str:
    """Construye la sección Markdown de Fuentes solo con los [n] realmente citados."""
    lines = ["## Fuentes"]
    for n in cited_indices:
        # n es 1-based; results es 0-based
        if 1 <= n <= len(results):
            r = results[n - 1]
            lines.append(f"[{n}] {r['title']} - {r['url']}")
    if len(lines) == 1:
        lines.append("_No se citaron fuentes en el informe._")
    return "\n".join(lines)


def writer_node(state: AgentState) -> dict:
    """Genera el informe y le concatena la sección Fuentes."""
    llm = get_llm()
    prompt = _PROMPT.format(
        question=state["question"],
        analysis=state["analysis"],
        sources=_format_sources_for_prompt(state["research_results"]),
    )
    response = llm.invoke(prompt)
    body = response.content if hasattr(response, "content") else str(response)

    cited = extract_cited_indices(body)
    sources_section = build_sources_section(cited, state["research_results"])

    full_report = f"{body.strip()}\n\n{sources_section}\n"

    return {
        "report": full_report,
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace("info", f"informe generado ({len(cited)} fuentes citadas)"),
        ],
    }
