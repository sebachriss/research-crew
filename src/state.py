"""Estado compartido del grafo de agentes.

Diseñado para LangGraph: campos acumulativos usan `Annotated[..., add]` como reducer.
Cada agente devuelve solo el diff; el reducer se encarga de fusionar.
"""
from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage  # noqa: F401 (reservado por compat)


class ResearchResult(TypedDict):
    url: str
    title: str
    content: str
    snippet: str
    query: str


class TraceEvent(TypedDict):
    timestamp: str
    node: str
    level: Literal["info", "warn", "error"]
    text: str


class NodeError(TypedDict):
    node: str
    iteration: int
    exception_type: str
    message: str


class AgentState(TypedDict):
    question: str

    # Producido por Supervisor
    queries: list[str]
    enough_info: bool | None
    next_agent: Literal["researcher", "writer", "END", ""]

    # Producido por Researcher (acumulativo)
    research_results: Annotated[list[ResearchResult], add]

    # Producido por Analyst (sobrescrito cada ronda)
    analysis: str
    gaps: list[str]

    # Producido por Writer
    report: str

    # Control y traza
    trace: Annotated[list[TraceEvent], add]
    iteration_count: int
    research_rounds: int
    errors: Annotated[list[NodeError], add]


def make_initial_state(question: str) -> AgentState:
    """Construye un estado inicial mínimo con la pregunta del usuario."""
    return AgentState(
        question=question,
        queries=[],
        enough_info=None,
        next_agent="",
        research_results=[],
        analysis="",
        gaps=[],
        report="",
        trace=[],
        iteration_count=0,
        research_rounds=0,
        errors=[],
    )
