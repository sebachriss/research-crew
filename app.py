"""Entrada de Streamlit. Mantén este archivo delgado: delega a src/ui/."""
from __future__ import annotations

import streamlit as st

from src.config import MAX_ITERATIONS, ConfigError, check_api_keys
from src.ui.async_bridge import run_graph_streaming
from src.ui.trace import render_trace_event

st.set_page_config(page_title="Research Crew", layout="wide")

# Estado de sesión
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "trace_events" not in st.session_state:
    st.session_state.trace_events = []
if "iteration_count" not in st.session_state:
    st.session_state.iteration_count = 0
if "research_rounds" not in st.session_state:
    st.session_state.research_rounds = 0
if "errors" not in st.session_state:
    st.session_state.errors = []
if "report" not in st.session_state:
    st.session_state.report = ""


# Sidebar: usamos placeholders que se refrescan dentro del loop, así la traza
# aparece incrementalmente y no toda de golpe al final.
with st.sidebar:
    st.title("Research Crew")
    st.caption("Sistema multi-agente con LangGraph")
    st.divider()
    st.subheader("Traza")
    trace_placeholder = st.empty()
    st.divider()
    metrics_placeholder = st.empty()
    errors_placeholder = st.empty()


def render_sidebar() -> None:
    """Repinta los placeholders de la sidebar desde st.session_state."""
    with trace_placeholder.container(height=400):
        if not st.session_state.trace_events:
            st.caption("_Esperando consulta…_")
        else:
            for ev in st.session_state.trace_events:
                st.markdown(render_trace_event(ev))

    with metrics_placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric("Iteración", f"{st.session_state.iteration_count}/{MAX_ITERATIONS}")
        col2.metric("Rondas", st.session_state.research_rounds)

    if st.session_state.errors:
        with errors_placeholder.expander(f"Errores ({len(st.session_state.errors)})"):
            for e in st.session_state.errors:
                st.code(
                    f"{e['node']} [iter {e['iteration']}]: "
                    f"{e['exception_type']}: {e['message']}"
                )
    else:
        errors_placeholder.empty()


render_sidebar()


# Main
st.title("Research Crew")

# Validar API keys al cargar
try:
    check_api_keys()
except ConfigError as e:
    st.error(f"Configuración faltante: {e}")
    st.stop()


question = st.text_input(
    "Pregunta",
    placeholder="¿Sobre qué quieres investigar?",
    disabled=st.session_state.is_running,
    label_visibility="collapsed",
)

report_placeholder = st.empty()

if st.button("Investigar", disabled=st.session_state.is_running or not question):
    # Reset estado de sesión para la nueva consulta
    st.session_state.is_running = True
    st.session_state.trace_events = []
    st.session_state.iteration_count = 0
    st.session_state.research_rounds = 0
    st.session_state.errors = []
    st.session_state.report = ""

    render_sidebar()  # limpia la traza visible
    report_placeholder.empty()

    streaming_text = ""

    for event_type, payload in run_graph_streaming(question):
        if event_type == "state":
            for _node_name, diff in payload.items():
                if "trace" in diff:
                    st.session_state.trace_events.extend(diff["trace"])
                if "iteration_count" in diff:
                    st.session_state.iteration_count = diff["iteration_count"]
                if "research_rounds" in diff:
                    st.session_state.research_rounds = diff["research_rounds"]
                if "errors" in diff:
                    st.session_state.errors.extend(diff["errors"])
                if "report" in diff:
                    st.session_state.report = diff["report"]
            render_sidebar()
        elif event_type == "token":
            streaming_text += payload
            report_placeholder.markdown(streaming_text + "▌")

    st.session_state.is_running = False
    # No llamamos st.rerun(): el bloque final renderiza el report y el download
    # button desde session_state, manteniendo el botón persistente entre runs.

# Render persistente del informe + download button. Funciona tanto justo
# después de una corrida como al recargar la página con un report ya cacheado.
if st.session_state.report and not st.session_state.is_running:
    report_placeholder.markdown(st.session_state.report)
    st.download_button(
        "Descargar informe (.md)",
        data=st.session_state.report,
        file_name="research-crew-report.md",
        mime="text/markdown",
    )
