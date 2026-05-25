# Research Crew — Diseño técnico

**Fecha:** 2026-05-24
**Estado:** Aprobado, listo para plan de implementación
**Contexto base:** `CLAUDE.md` en la raíz del repo

---

## 1. Resumen

Sistema multi-agente que recibe una pregunta de investigación y devuelve un informe ejecutivo Markdown con citas. Coordina supervisor + 3 workers (researcher, analyst, writer) con LangGraph. Demo visual en Streamlit con traza de eventos por agente y streaming token-por-token del informe final.

**Stack confirmado (de CLAUDE.md):** Python 3.11+, LangGraph, Gemini 2.5 Flash (vía `langchain-google-genai`), Tavily, Streamlit, `python-dotenv`, `pydantic`, `ruff`.

**Decisiones del brainstorm:**
- Supervisor con LLM solo en un punto (evaluación post-Analyst); todo lo demás es routing determinístico.
- Supervisor genera 3 sub-queries iniciales (modo PLAN) y queries de seguimiento (modo EVAL si dice que falta info).
- Researcher ejecuta queries en paralelo con `asyncio.gather`, sin LLM.
- Informe final: Markdown ~600–1000 palabras, idioma = pregunta, citas inline `[n]`, sección Fuentes generada por código.
- UI: sidebar con timeline de eventos del grafo (`stream_mode="updates"`) + streaming token-por-token del Writer en el main panel (`stream_mode="messages"`).

---

## 2. Arquitectura

### Grafo

```
                    START
                      │
                      ▼
              ┌──────────────┐
              │  Supervisor  │◀─────────┐
              │  (con LLM)   │          │
              └───┬──────┬───┘          │
       plan_ready │      │ enough_info  │
                  ▼      ▼              │
          ┌──────────┐  ┌──────────┐    │
          │Researcher│  │  Writer  │──▶ END
          └────┬─────┘  └──────────┘
               ▼
          ┌──────────┐
          │ Analyst  │──────────────────┘
          └──────────┘
```

### Topología del grafo

**Aristas fijas** (`add_edge`):
- `START → supervisor`
- `researcher → analyst`
- `analyst → supervisor`
- `writer → END`

**Aristas condicionales** (`add_conditional_edges` solo desde supervisor) — evalúan `state["next_agent"]`:
- `"researcher"` → researcher
- `"writer"` → writer
- `"END"` → END (no usado en práctica porque writer→END es arista fija; reservado por completitud)

### Lógica interna del supervisor

| Condición al entrar al supervisor | Modo | Resultado |
|---|---|---|
| `queries == []` y `research_results == []` | **PLAN (LLM)** | LLM genera 3 sub-queries → `next_agent="researcher"` |
| `analysis != ""` y `enough_info is None` y caps OK | **EVAL (LLM)** | LLM decide `enough_info`; si `True` → `next_agent="writer"`; si `False` → genera nuevas `queries` y `next_agent="researcher"` |
| `iteration_count >= MAX_ITERATIONS - 1` o `research_rounds >= MAX_RESEARCH_ROUNDS` | **EVAL bypass** | Sin LLM: `enough_info=True`, `next_agent="writer"`, trace warn |

**Único uso de LLM en el supervisor:**
- Modo PLAN: descomponer `question` en 3 sub-queries (al inicio).
- Modo EVAL: leer `question + analysis + gaps` y devolver `enough_info: bool` + nuevas `queries` (si False).

**Caps:**
- `MAX_ITERATIONS = 8` (global del grafo)
- `MAX_RESEARCH_ROUNDS = 3`
- `QUERIES_PER_ROUND = 3`
- `TAVILY_MAX_RESULTS = 5` por query

Si se llega a cualquiera de los caps superiores, el supervisor fuerza Writer con la info que haya. El grafo nunca termina sin `report`.

---

## 3. Estado compartido

`src/state.py`:

```python
from typing import Literal, TypedDict
from operator import add
from typing_extensions import Annotated

class ResearchResult(TypedDict):
    url: str
    title: str
    content: str       # texto limpio devuelto por Tavily
    snippet: str       # resumen corto para citas
    query: str         # sub-query que produjo este resultado

class TraceEvent(TypedDict):
    timestamp: str                                # ISO-8601
    node: str                                     # "supervisor" | "researcher" | "analyst" | "writer"
    level: Literal["info", "warn", "error"]
    text: str

class NodeError(TypedDict):
    node: str
    iteration: int
    exception_type: str
    message: str

class AgentState(TypedDict):
    # Input
    question: str

    # Producido por Supervisor
    queries: list[str]                            # sobrescrito cada ronda
    enough_info: bool | None
    next_agent: Literal["researcher", "writer", "END", ""]   # "" es el valor inicial; el supervisor lo setea en cada paso

    # Producido por Researcher
    research_results: Annotated[list[ResearchResult], add]  # acumulativo

    # Producido por Analyst
    analysis: str                                 # sobrescrito cada ronda (último incluye todo)
    gaps: list[str]

    # Producido por Writer
    report: str

    # Control y traza
    trace: Annotated[list[TraceEvent], add]       # acumulativo
    iteration_count: int                          # incrementa en cada return de nodo
    research_rounds: int                          # incrementa en cada return del Researcher
    errors: Annotated[list[NodeError], add]       # acumulativo
```

**Invariantes:**
- **Solo el supervisor escribe `next_agent`.** Los workers (researcher, analyst, writer) no lo tocan.
- El grafo combina dos tipos de aristas:
  - **Fijas:** `researcher → analyst`, `analyst → supervisor`, `writer → END`. Definidas con `add_edge`.
  - **Condicionales (solo desde supervisor):** evalúan `state["next_agent"]` ∈ `{"researcher", "writer", "END"}`. Definidas con `add_conditional_edges`.
- `research_results`, `trace`, `errors` son acumulativos (reducer = `operator.add`).
- Todos los nodos devuelven solo el diff, no el estado completo.

---

## 4. Agentes

### 4.1 Supervisor — `src/agents/supervisor.py`

Único agente con LLM en su lógica. Se invoca en exactamente dos momentos:
1. **Al arranque del grafo** (estado vacío) → ejecuta modo PLAN.
2. **Después del Analyst** (vía arista fija `analyst → supervisor`) → ejecuta modo EVAL.

| Modo | Condición de entrada | Acción LLM | Output diff |
|---|---|---|---|
| **PLAN** | `queries == []` y `research_results == []` | LLM descompone `question` en 3 sub-queries → `PlanOutput` | `queries`, `next_agent="researcher"`, `trace += [PLAN_event]`, `iteration_count += 1` |
| **EVAL** | `analysis != ""` y `enough_info is None` | LLM lee `question + analysis + gaps` → `EvalOutput { enough_info, queries, reasoning }` | `enough_info`, `queries` (si `enough_info=False`), `next_agent` (`"writer"` o `"researcher"`), `trace += [EVAL_event]`, `iteration_count += 1` |

**Caps aplicados antes de invocar el LLM en EVAL:**
- Si `iteration_count >= MAX_ITERATIONS - 1` o `research_rounds >= MAX_RESEARCH_ROUNDS`: saltar LLM, forzar `enough_info=True`, `next_agent="writer"`, agregar `TraceEvent(level="warn", text="cap alcanzado, forzando cierre")`.
- El LLM en modo EVAL nunca se invoca si los caps ya están al límite.

**Prompt único `prompts/supervisor.txt`** con dos secciones `# MODE: PLAN` y `# MODE: EVAL`; el código elige cuál enviar según el estado. Salida estructurada vía pydantic + `.with_structured_output(PlanOutput)` / `(EvalOutput)`.

**Schemas (`src/schemas.py`):**
```python
class PlanOutput(BaseModel):
    queries: list[str] = Field(..., min_length=3, max_length=3)

class EvalOutput(BaseModel):
    enough_info: bool
    queries: list[str] = Field(default_factory=list)  # vacío si enough_info=True
    reasoning: str
```

### 4.2 Researcher — `src/agents/researcher.py`

Sin LLM. Ejecuta búsquedas Tavily en paralelo.

**Lógica:**
1. Lee `state["queries"]` (siempre tendrá 3 entradas si llegó aquí).
2. Construye coroutines: `[tavily_search(q) for q in queries]`.
3. `results = await asyncio.gather(*coros, return_exceptions=True)`.
4. Para cada resultado:
   - Si es excepción → append a `errors`, agrega trace warn.
   - Si es ok → normaliza a `ResearchResult` (campos: url, title, content, snippet, query).
5. Devuelve `{"research_results": new_results, "research_rounds": state["research_rounds"] + 1, "trace": [...], "iteration_count": +1}`.

No toca `next_agent` — la arista `researcher → analyst` es fija.

**Caso: 3 queries fallan (0 resultados nuevos):** Researcher agrega 3 entradas a `errors` y un `TraceEvent(level="warn")`, pero igual sigue al Analyst por la arista fija. El Analyst, al ver `research_results` sin contenido nuevo y total potencialmente vacío, devuelve un `analysis` con disclaimer y `gaps` poblados. El Supervisor en EVAL decide otra ronda o fuerza cierre según los caps.

### 4.3 Analyst — `src/agents/analyst.py`

LLM. Sintetiza todo `research_results` acumulado (todas las rondas).

**Input al prompt (`prompts/analyst.txt`):**
- `question`
- Lista compacta de resultados: solo `title + snippet + url + query` (NO el `content` completo, para acotar tokens).
- `research_rounds` actual (para saber si es síntesis fresca o iterativa).

**Output estructurado:**
```python
class AnalystOutput(BaseModel):
    analysis: str                # 200-400 palabras
    gaps: list[str]              # puede ser []
```

**Diff:** `{"analysis": ..., "gaps": ..., "trace": [...], "iteration_count": +1}`.

No toca `next_agent` — la arista `analyst → supervisor` es fija.

Sobrescribe `analysis` cada ronda. Si `research_results` está vacío, el prompt instruye devolver un disclaimer en `analysis` y poblar `gaps` con la pregunta original.

### 4.4 Writer — `src/agents/writer.py`

LLM con streaming.

**Input al prompt (`prompts/writer.txt`):**
- `question`
- `analysis` (síntesis final del Analyst)
- Lista de fuentes numerada: `[1] {title} - {url}`, `[2] {title} - {url}`, …
- Instrucciones de formato:
  - ~600–1000 palabras en idioma de la pregunta.
  - Estructura: TL;DR (3 bullets) → Hallazgos clave (3-5 subsecciones) → Contradicciones/Gaps → Conclusión.
  - Citas inline `[n]` solo de la lista provista; NO inventar fuentes ni URLs.
  - NO incluir sección "Fuentes" en el output del LLM (se construye por código).

**Streaming:** se invoca con `llm.astream()`. Los tokens fluyen vía `graph.astream(stream_mode=["updates", "messages"])` con filtro `metadata["langgraph_node"] == "writer"`.

**Post-procesamiento (código, no LLM):**
1. Parsear el texto final del Writer y extraer los `[n]` realmente citados (regex).
2. Construir sección `## Fuentes` listando solo esas, en el orden de aparición, con `[n] título - URL`.
3. Concatenar al final del report.

**Diff:** `{"report": full_markdown, "trace": [...], "iteration_count": +1}`.

No toca `next_agent` — la arista `writer → END` es fija.

### Reglas comunes a todos los agentes

- Función pura `(state) -> dict`; retorna solo el diff.
- Captura todas las excepciones internas; las registra en `errors` con `TraceEvent(level="error")`; no propaga.
- Incrementa `iteration_count` en su return.
- Solo el supervisor escribe `next_agent`. Los workers (researcher, analyst, writer) nunca lo tocan: su próximo destino lo define una arista fija.

---

## 5. UI Streamlit

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Research Crew                                    [⚙ Modelo] │
├──────────────────────┬──────────────────────────────────────┤
│  Sidebar             │  Main                                │
│  📋 Traza            │  [input box: pregunta]               │
│  [HH:MM:SS] PLAN     │  [🔍 Investigar]                     │
│  → 3 sub-queries     │  ──────────────                      │
│  [HH:MM:SS] SEARCH   │  ## TL;DR                            │
│  → "..."             │  - ...                               │
│  ...                 │                                      │
│  ──────────────      │  ## Hallazgos                        │
│  Iteración: N/8      │  ... texto streaming ▮               │
│  Rondas: M           │                                      │
│  ⚠ Errores [expand]  │  ## Fuentes (post-proceso)           │
└──────────────────────┴──────────────────────────────────────┘
```

### Componentes

**Sidebar (`st.sidebar`):**
- Sección **Traza**: loop sobre `state["trace"]`, renderizado con `src/ui/trace.py` → markdown coloreado por `level`. Auto-scroll al fondo.
- **Contadores**: `Iteración N/8`, `Rondas research M`.
- **Errores expandibles** si `errors != []`.

**Main panel** — tres estados:
- **Idle**: `st.text_input` + botón `🔍 Investigar`.
- **Running**: input deshabilitado; cuerpo recibe streaming token-por-token con `st.write_stream`.
- **Done**: `st.markdown(state["report"])`, botón de descarga `.md`, botón de copiar.

**Config (sidebar top):**
- Selector de modelo (`gemini-2.5-flash` / `gemini-2.5-pro`).
- Slider `Max iteraciones` (default 8).
- NO se expone API key.

### Wiring async

`src/ui/async_bridge.py` envuelve `graph.astream(stream_mode=["updates", "messages"])` en un generador sync usando `asyncio.run` + `queue.Queue` (patrón estándar async-to-sync para Streamlit).

```python
async def run_graph_async(question: str):
    graph = build_graph()
    initial_state = AgentState(
        question=question, queries=[], enough_info=None,
        next_agent="",   # se setea al primer paso del supervisor
        research_results=[], analysis="", gaps=[], report="",
        trace=[], iteration_count=0, research_rounds=0, errors=[],
    )
    async for event_type, payload in graph.astream(
        initial_state, stream_mode=["updates", "messages"]
    ):
        if event_type == "updates":
            yield ("state", payload)
        elif event_type == "messages":
            token, metadata = payload
            if metadata.get("langgraph_node") == "writer":
                yield ("token", token.content)
```

`st.session_state`: `current_state` (último diff acumulado), `is_running` (bool).

---

## 6. Estructura de archivos

```
research-crew/
├── README.md
├── CLAUDE.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── state.py               # AgentState, ResearchResult, TraceEvent, NodeError
│   ├── graph.py               # build_graph() + entry CLI
│   ├── llm.py                 # factory: get_llm(model_name)
│   ├── config.py              # MAX_ITERATIONS, MAX_RESEARCH_ROUNDS, QUERIES_PER_ROUND, TAVILY_MAX_RESULTS, MODEL_NAME_DEFAULT
│   ├── schemas.py             # PlanOutput, EvalOutput, AnalystOutput
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── researcher.py
│   │   ├── analyst.py
│   │   └── writer.py
│   │
│   ├── prompts/
│   │   ├── supervisor.txt     # secciones # MODE: PLAN y # MODE: EVAL
│   │   ├── analyst.txt
│   │   └── writer.txt
│   │
│   └── ui/
│       ├── __init__.py
│       ├── trace.py           # render de TraceEvent
│       └── async_bridge.py    # wrapper async→sync
│
├── app.py                     # entry Streamlit (delegado a src/ui/)
│
└── tests/
    ├── __init__.py
    ├── conftest.py            # fixtures
    ├── test_state.py
    ├── test_supervisor.py
    ├── test_researcher.py
    ├── test_analyst.py
    ├── test_writer.py
    └── test_graph.py
```

### Diferencias vs CLAUDE.md (todas justificadas)

- `+ src/config.py` — centraliza constantes (cumple regla de no-hardcoding de CLAUDE.md).
- `+ src/schemas.py` — pydantic models compartidos entre supervisor y analyst (evita imports circulares).
- `+ src/ui/` — separa render de `app.py` para mantener `app.py` delgado y testeable.
- `- src/prompts/researcher.txt` — el Researcher no usa LLM, no necesita prompt.

### `requirements.txt`

```
langgraph>=0.2.0
langchain-google-genai>=2.0.0
langchain-core>=0.3.0
tavily-python>=0.5.0
streamlit>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0

pytest>=8.0.0
pytest-asyncio>=0.24.0
ruff>=0.6.0
```

### `pyproject.toml`

```toml
[project]
name = "research-crew"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### `.env.example`

```
GOOGLE_API_KEY=
TAVILY_API_KEY=
MODEL_NAME=gemini-2.5-flash
MAX_ITERATIONS=8
```

---

## 7. Testing

**Principio:** todos los tests corren en <5s sin consumir API. Cero llamadas reales a Gemini ni a Tavily.

| Archivo | Cubre |
|---|---|
| `test_state.py` | Reducers (`add` para acumulativos), tipado básico |
| `test_supervisor.py` | Todas las celdas de la tabla de routing + modos PLAN/EVAL con LLM mockeado + caps forzados |
| `test_researcher.py` | Paralelismo (3 calls concurrentes), normalización, fallo parcial (1/3), fallo total (3/3) |
| `test_analyst.py` | Síntesis JSON válida; con `research_results=[]` devuelve disclaimer + gaps |
| `test_writer.py` | Genera report con citas `[n]`; sección Fuentes construida por código solo con `[n]` realmente citados |
| `test_graph.py` | E2E: (a) ronda única con `enough_info=True`; (b) re-research con segunda ronda + cierre |

**Fixtures (`conftest.py`):**
- `mock_llm_supervisor`: devuelve `PlanOutput` o `EvalOutput` según el prompt.
- `mock_tavily`: monkeypatch sobre `TavilyClient.search`, respuestas configurables por query.
- `sample_state`: estado inicial mínimo válido.

**Lo que NO se testea:**
- Calidad del output del LLM (no aplica con mocks).
- UI Streamlit (validación manual).

**Comando único:** `pytest tests/ -v`. Tiempo objetivo: <5s.

---

## 8. Manejo de errores

Principio (regla de CLAUDE.md): cada agente captura, registra en `errors`, no propaga. El grafo siempre termina con un `report`, aunque sea degradado.

| Origen | Detección | Acción | Efecto en flujo |
|---|---|---|---|
| Tavily error en 1 de 3 queries | `return_exceptions=True` en `gather` | Append a `errors`; las exitosas siguen | Sin cambio: Analyst recibe los que llegaron |
| Tavily error en 3 de 3 | Iteración detecta todos `Exception` | Append errors fatales; trace warn; sigue al Analyst (arista fija) | Analyst recibe `research_results` sin contenido nuevo → devuelve disclaimer y `gaps` poblados → supervisor en EVAL decide otra ronda o fuerza cierre según caps |
| LLM devuelve JSON no parseable | `pydantic.ValidationError` | 1 reintento con prompt "tu respuesta no fue JSON válido"; si falla, default seguro (PLAN: queries dummy desde la question; EVAL: `enough_info=True` forzado) | No bloquea el flujo |
| LLM rate-limited (429) | Excepción del SDK | Backoff exponencial 3 reintentos (1s/2s/4s) | Si todos fallan: error fatal, supervisor fuerza Writer |
| `iteration_count >= 8` | Check en supervisor | Routing forzado a Writer | Reporte se genera con la info disponible |
| `research_rounds >= 3` | Check en supervisor EVAL | `enough_info=True` forzado | Pasa a Writer |
| Writer falla completamente | Catch en el nodo | `report` = mensaje genérico de fallback | Grafo termina, UI no crashea |
| `.env` sin keys | Validación al inicio (no por nodo) | Lanza `ConfigError` antes de `astream` | UI muestra mensaje rojo de configuración |

**Lo que NO hacemos:**
- No reintentos propios de Tavily (1 intento por query).
- No circuit breakers ni rate limiters propios (confiamos en backoff del SDK).
- No persistencia entre sesiones.

---

## 9. Decisiones de diseño (anexo)

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Supervisor con LLM solo en EVAL | LLM en todo routing | Routing trivial es determinístico — usar LLM ahí gasta tokens y latencia sin ganancia |
| Supervisor genera queries (no el Researcher) | Researcher autónomo | Mantiene patrón supervisor-workers limpio; Researcher queda testeable como nodo sin LLM |
| 3 queries en paralelo (`asyncio.gather`) | Secuencial 1 a la vez | Latencia más baja; 9 búsquedas máximo en 3 rondas cabe en free tier Tavily |
| Sobrescribir `queries` por ronda | `plan` + `followup_queries` separados | Un solo campo, menos lógica condicional; la ronda se infiere de `research_rounds` |
| `TraceEvent` propio en vez de `BaseMessage` | Reutilizar `BaseMessage` de langchain | Propósito (UI trace) es distinto de conversación LLM; tipo simple es más fácil de renderizar |
| Sección Fuentes construida por código | LLM la genera | Evita alucinación de URLs; garantiza que solo aparezcan fuentes realmente citadas |
| Streaming token-por-token solo en Writer | Streaming en todos los nodos | Solo el reporte final justifica el efecto visual; los otros nodos producen JSON estructurado pequeño |
| Tests con mocks 100% | Tests con API real | Velocidad (<5s), reproducibilidad, no consume free tier |
