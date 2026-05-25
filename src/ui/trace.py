"""Render de TraceEvent a Markdown coloreado para Streamlit."""
from src.state import TraceEvent

COLOR_BY_LEVEL = {
    "info": "gray",
    "warn": "orange",
    "error": "red",
}


def render_trace_event(event: TraceEvent) -> str:
    """Convierte un TraceEvent a una línea de Markdown con color por nivel."""
    # Extrae HH:MM:SS del ISO timestamp
    ts = event["timestamp"]
    time_part = ts.split("T")[1].split("+")[0] if "T" in ts else ts
    color = COLOR_BY_LEVEL.get(event["level"], "gray")
    node = event["node"].upper()
    return f":{color}[**[{time_part}] {node}**] {event['text']}"
