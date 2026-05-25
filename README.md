# Research Crew

Sistema multi-agente que recibe una pregunta de investigación y produce un informe ejecutivo con citas, coordinando supervisor + 3 workers (researcher, analyst, writer) vía LangGraph.

## Arquitectura

```
                    START
                      │
                      ▼
              ┌──────────────┐
              │  Supervisor  │◀─────────┐
              │  (con LLM)   │          │
              └───┬──────┬───┘          │
                  │      │              │
                  ▼      ▼              │
          ┌──────────┐  ┌──────────┐    │
          │Researcher│  │  Writer  │──▶ END
          └────┬─────┘  └──────────┘
               ▼
          ┌──────────┐
          │ Analyst  │──────────────────┘
          └──────────┘
```

- **Supervisor** usa LLM en dos modos: PLAN (descompone la pregunta en 3 sub-queries) y EVAL (decide si el análisis basta o pedir más research).
- **Researcher** ejecuta las 3 queries en paralelo con Tavily (`asyncio.gather`).
- **Analyst** sintetiza los resultados y reporta gaps.
- **Writer** redacta el informe en Markdown con citas `[n]`; la sección Fuentes se construye por código a partir de las citas realmente usadas.

## Stack

- Python 3.11+
- [LangGraph](https://langchain-ai.github.io/langgraph/) — orquestación del grafo
- [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs) vía `langchain-google-genai`
- [Tavily](https://docs.tavily.com/) — búsqueda web optimizada para agentes
- [Streamlit](https://streamlit.io/) — UI con streaming
- pydantic, pytest, ruff

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
cp .env.example .env              # editar con GOOGLE_API_KEY y TAVILY_API_KEY
```

### CLI

```bash
python -m src.graph "¿Cuál es el estado actual de la regulación de IA en Chile?"
```

### UI

```bash
streamlit run app.py
```

Abre [localhost:8501](http://localhost:8501).

## Tests

```bash
pytest tests/ -v
```

Todos los tests usan mocks (no consumen API ni dependen de red). Tiempo objetivo: <5s.

## Decisiones de diseño

Ver [`docs/superpowers/specs/2026-05-24-research-crew-design.md`](docs/superpowers/specs/2026-05-24-research-crew-design.md) para el spec completo.

Highlights:

- **Supervisor con LLM solo en EVAL** — el routing trivial es determinístico (código), el LLM solo se usa donde aporta valor.
- **Aristas fijas + condicionales** — `researcher→analyst`, `analyst→supervisor`, `writer→END` son aristas fijas; las condicionales solo bajan del supervisor sobre `state["next_agent"]`.
- **Citas blindadas** — la sección Fuentes la construye Python, no el LLM, a partir de los `[n]` realmente citados. Garantiza cero alucinación de URLs.
- **Streaming token-por-token solo en Writer** — los otros nodos producen JSON estructurado pequeño; solo el reporte final justifica el efecto visual.
- **Manejo de errores graceful** — cada agente captura excepciones, las registra en el estado, y devuelve un diff seguro. El grafo siempre termina con un `report`, aunque sea degradado.

## Estructura del proyecto

```
src/
├── config.py              # constantes y env vars
├── state.py               # AgentState
├── schemas.py             # pydantic outputs
├── llm.py                 # factory del LLM
├── graph.py               # build_graph + CLI
├── agents/                # supervisor, researcher, analyst, writer
├── prompts/               # .txt separados por agente
└── ui/                    # helpers de Streamlit
app.py                     # entry Streamlit
tests/                     # tests unitarios + e2e con mocks
```

## Licencia

MIT
