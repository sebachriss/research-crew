"""Construcción del StateGraph y entry point CLI.

Topología:
  - Aristas fijas: START->supervisor, researcher->analyst, analyst->supervisor, writer->END
  - Aristas condicionales: solo desde supervisor sobre state["next_agent"]
"""
from __future__ import annotations

import asyncio
import sys

from langgraph.graph import END, START, StateGraph

from src.agents.analyst import analyst_node
from src.agents.researcher import researcher_node
from src.agents.supervisor import supervisor_node
from src.agents.writer import writer_node
from src.state import AgentState, make_initial_state


def _route_from_supervisor(state: AgentState) -> str:
    """Lee state['next_agent'] y devuelve el nombre del nodo destino o END."""
    nxt = state["next_agent"]
    if nxt == "researcher":
        return "researcher"
    if nxt == "writer":
        return "writer"
    return END


def build_graph():
    """Construye el StateGraph del Research Crew."""
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)

    # Aristas fijas
    g.add_edge(START, "supervisor")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "supervisor")
    g.add_edge("writer", END)

    # Aristas condicionales desde supervisor
    g.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {"researcher": "researcher", "writer": "writer", END: END},
    )

    return g.compile()


async def run_cli(question: str) -> str:
    """Ejecuta el grafo en CLI y devuelve el report final."""
    from src.config import check_api_keys
    check_api_keys()

    graph = build_graph()
    initial = make_initial_state(question)
    final = await graph.ainvoke(initial)
    return final["report"]


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.graph '<pregunta>'", file=sys.stderr)
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    report = asyncio.run(run_cli(question))
    print(report)


if __name__ == "__main__":
    main()
