"""Entrada de Streamlit. Mantén este archivo delgado: delega a src/ui/."""
from __future__ import annotations

import streamlit as st

from src.config import ConfigError, check_api_keys
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


# Sidebar
with st.sidebar:
    st.title("Research Crew")
    st.caption("Sistema multi-agente con LangGraph")

    st.divider()
    st.subheader("Traza")
    trace_container = st.container(height=400)
    with trace_container:
        if not st.session_state.trace_events:
            st.caption("_Esperando consulta…_")
        else:
            for ev in st.session_state.trace_events:
                st.markdown(render_trace_event(ev))

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Iteración", f"{st.session_state.iteration_count}/8")
    col2.metric("Rondas", st.session_state.research_rounds)

    if st.session_state.errors:
        with st.expander(f"Errores ({len(st.session_state.errors)})"):
            for e in st.session_state.errors:
                st.code(
                    f"{e['node']} [iter {e['iteration']}]: "
                    f"{e['exception_type']}: {e['message']}"
                )


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

if st.button("Investigar", disabled=st.session_state.is_running or not question):
    # Reset estado de sesión para la nueva consulta
    st.session_state.is_running = True
    st.session_state.trace_events = []
    st.session_state.iteration_count = 0
    st.session_state.research_rounds = 0
    st.session_state.errors = []
    st.session_state.report = ""

    main_container = st.container()
    report_placeholder = main_container.empty()
    streaming_text = ""

    # Consume el generador del bridge
    for event_type, payload in run_graph_streaming(question):
        if event_type == "state":
            # payload = {"node_name": diff_dict}
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
        elif event_type == "token":
            streaming_text += payload
            report_placeholder.markdown(streaming_text + "▌")

    # Render final del reporte completo (con sección Fuentes)
    if st.session_state.report:
        report_placeholder.markdown(st.session_state.report)
        st.download_button(
            "Descargar informe (.md)",
            data=st.session_state.report,
            file_name="research-crew-report.md",
            mime="text/markdown",
        )

    st.session_state.is_running = False
    st.rerun()

elif st.session_state.report and not st.session_state.is_running:
    # Mostrar último report al recargar la página
    st.markdown(st.session_state.report)
