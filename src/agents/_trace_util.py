"""Utilidades compartidas para emitir TraceEvent desde los nodos del grafo.

Centralizado acá para evitar duplicar `_now`/`_trace` en cada agente.
"""
from datetime import UTC, datetime
from typing import Literal

from src.state import TraceEvent

TraceLevel = Literal["info", "warn", "error"]


def now() -> str:
    """Timestamp ISO8601 en UTC, redondeado al segundo."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def trace(node: str, level: TraceLevel, text: str) -> TraceEvent:
    """Construye un TraceEvent tipado. `node` identifica al emisor."""
    return TraceEvent(timestamp=now(), node=node, level=level, text=text)
