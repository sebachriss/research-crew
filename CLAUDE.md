# CLAUDE.md — Research Crew

> Documento de contexto para Claude Code. Léelo al inicio de cada sesión antes de hacer cambios al proyecto.

---

## 🎯 Qué es este proyecto

**Research Crew** es un sistema multi-agente que recibe una pregunta de investigación y produce un informe ejecutivo con citas, coordinando varios agentes especializados mediante **LangGraph**.

Es un proyecto de portafolio para demostrar arquitectura de agentes con LLMs en producción. El énfasis está en:

1. **Arquitectura clara** de un grafo multi-agente (supervisor + workers).
2. **Código legible y bien comentado** — alguien debe entender el flujo leyendo un solo archivo.
3. **Demo visual** con Streamlit que muestre la traza de cada agente en tiempo real.
4. **README impecable** — el repo se evalúa en 30 segundos.

**No es** un producto comercial ni un sistema de investigación nivel PhD. Es una **demostración técnica**.

---

## 🏗 Arquitectura

```
                    ┌─────────────┐
                    │  Supervisor │  ← decide qué worker llamar y cuándo terminar
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐       ┌─────────┐
   │Researcher│      │ Analyst  │       │ Writer  │
   │  (web)   │      │(síntesis)│       │(informe)│
   └──────────┘      └──────────┘       └─────────┘
```

### Flujo de ejecución

1. Usuario ingresa pregunta en la UI Streamlit.
2. **Supervisor** evalúa el estado y decide el siguiente nodo:
   - Si falta información → llama a `Researcher`.
   - Si hay info suficiente sin sintetizar → llama a `Analyst`.
   - Si hay análisis sin redactar → llama a `Writer`.
   - Si el `Writer` ya entregó → termina (`END`).
3. **Researcher** ejecuta búsquedas web con la tool de Tavily y agrega resultados al estado.
4. **Analyst** sintetiza hallazgos, detecta contradicciones, identifica gaps.
5. **Writer** redacta el informe final en Markdown con citas numeradas.
6. La UI muestra el informe + la traza completa (qué hizo cada agente, en qué orden).

### Estado compartido (`AgentState`)

```python
class AgentState(TypedDict):
    question: str                    # pregunta original del usuario
    plan: str                        # plan inicial del supervisor
    research_results: list[dict]     # [{url, title, content, snippet}]
    analysis: str                    # síntesis del Analyst
    report: str                      # informe final en Markdown
    messages: list[BaseMessage]      # historial de mensajes entre agentes
    next_agent: str                  # decisión del supervisor: researcher | analyst | writer | END
    iteration_count: int             # contador para evitar loops infinitos
```

---

## 🛠 Stack técnico

| Componente | Elección | Notas |
|---|---|---|
| Lenguaje | Python 3.11+ | Tipado estricto donde se pueda |
| Orquestación | **LangGraph** | `StateGraph` con ciclos condicionales |
| LLM | **Gemini 2.5 Flash** | Vía `langchain-google-genai` |
| Búsqueda web | **Tavily** | Free tier, diseñada para agentes |
| UI | **Streamlit** | Streaming de progreso en tiempo real |
| Gestión de envs | `python-dotenv` | API keys en `.env`, nunca commiteado |
| Linter/Format | `ruff` | Configurado en `pyproject.toml` |

### Dependencias mínimas

```
langgraph>=0.2.0
langchain-google-genai>=2.0.0
langchain-core>=0.3.0
tavily-python>=0.5.0
streamlit>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## 📁 Estructura del repo

```
research-crew/
├── README.md                  # Documentación pública (el "vendedor" del repo)
├── CLAUDE.md                  # Este archivo
├── .env.example               # Variables de entorno (sin valores reales)
├── .gitignore
├── pyproject.toml             # Config de ruff y proyecto
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── state.py               # AgentState (TypedDict)
│   ├── graph.py               # Construcción del StateGraph
│   ├── llm.py                 # Factory del LLM (Gemini)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py      # Lógica de routing
│   │   ├── researcher.py      # Búsquedas web con Tavily
│   │   ├── analyst.py         # Síntesis
│   │   └── writer.py          # Redacción del informe
│   │
│   └── prompts/
│       ├── supervisor.txt
│       ├── researcher.txt
│       ├── analyst.txt
│       └── writer.txt
│
├── app.py                     # Entrada de Streamlit
│
└── tests/
    ├── test_graph.py          # Tests del flujo end-to-end
    └── test_agents.py         # Tests unitarios por agente (mocked LLM)
```

---

## 📋 Convenciones de código

### Estilo

- **Tipado**: usar type hints en todas las funciones públicas.
- **Docstrings**: estilo Google, breves pero presentes en cada función no trivial.
- **Imports**: agrupados con isort (stdlib → terceros → locales).
- **Nombres**: `snake_case` en Python, `PascalCase` para clases.
- **Longitud de línea**: 100 caracteres.

### Patrones

- **Prompts en archivos `.txt` separados**, no inline en código. Facilita iterar sin tocar lógica.
- **Cada agente es una función pura**: `def agent_node(state: AgentState) -> dict`. Retorna solo el diff del estado, no el estado completo.
- **El supervisor no usa LLM para routing trivial**. Si la regla es determinística (ej. "si no hay research_results, ir a researcher"), se usa código. El LLM solo decide en casos ambiguos.
- **Manejo de errores**: cada agente captura excepciones y las registra en el estado, no las propaga. El supervisor decide qué hacer.
- **Iteración máxima**: 8 iteraciones totales del grafo. Si se excede, el supervisor fuerza `END`.

### Anti-patrones a evitar

- ❌ Llamadas síncronas bloqueantes en la UI sin feedback.
- ❌ Prompts gigantes con todo el contexto inline — usar el estado.
- ❌ Tools que retornen estructuras inconsistentes — siempre tipadas con Pydantic.
- ❌ Hardcoding de modelos, temperaturas, o keys — todo via config/env.

---

## 🚦 Cómo trabajar en este proyecto

### Setup inicial

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con GOOGLE_API_KEY y TAVILY_API_KEY
```

### Correr localmente

```bash
# CLI (para debug rápido)
python -m src.graph "¿Cuál es el estado actual de la regulación de IA en Chile?"

# UI Streamlit
streamlit run app.py
```

### Tests

```bash
pytest tests/ -v
```

---

## 📍 Estado actual del proyecto

> Esta sección la mantenemos actualizada manualmente conforme avanza el proyecto.

**Fase actual:** 🟢 Implementación completa (41/41 tests pasando, ruff clean)

### Hecho

**Día 1 — Sábado AM**
- [x] Estructura de carpetas (`src/`, `src/agents/`, `src/ui/`, `src/prompts/`, `tests/`).
- [x] Setup de `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`.
- [x] `src/config.py` con env vars + `check_api_keys`.
- [x] `src/state.py` con `AgentState`, `ResearchResult`, `TraceEvent`, `NodeError` y `make_initial_state`.
- [x] `src/schemas.py` con `PlanOutput`, `EvalOutput`, `AnalystOutput`.
- [x] `src/llm.py` con factory de Gemini.
- [x] `src/agents/supervisor.py` con modos PLAN y EVAL + caps bypass.

**Día 1 — Sábado PM**
- [x] `src/agents/researcher.py` con Tavily async parallel (`asyncio.gather`).
- [x] `src/agents/analyst.py` con síntesis estructurada.
- [x] `src/agents/writer.py` con post-proceso de citas (sección Fuentes construida por código).
- [x] Prompts en `src/prompts/supervisor.txt`, `analyst.txt`, `writer.txt`.
- [x] `src/graph.py` con `build_graph` + CLI entry.
- [x] Tests e2e end-to-end con LLM y Tavily mockeados (ronda única + re-research).

**Día 2 — Domingo AM**
- [x] `app.py` con Streamlit, sidebar de traza, streaming token-por-token del informe.
- [x] `src/ui/async_bridge.py` (wrapper async→sync).
- [x] `src/ui/trace.py` (render de `TraceEvent` coloreado por nivel).

**Día 2 — Domingo PM**
- [x] README con diagrama, quickstart, decisiones de diseño, estructura.
- [x] Manejo de errores robusto (try/except genérico en supervisor/analyst/writer).
- [ ] Pulido visual de la UI (validación manual pendiente con `.env` configurado).
- [ ] Push final a GitHub público.

### Cobertura

- 41 tests automatizados, todos con mocks (no consumen API). Tiempo: <3s.
- Ruff clean en todo `src/` y `tests/`.
- 24 commits frecuentes (uno por task del plan + fixes intermedios).
- Spec: `docs/superpowers/specs/2026-05-24-research-crew-design.md`.
- Plan: `docs/superpowers/plans/2026-05-24-research-crew.md`.

---

## 🎓 Decisiones de diseño (y por qué)

**¿Por qué LangGraph y no CrewAI/AutoGen?**
LangGraph da control explícito del flujo (ciclos, condiciones, estado). CrewAI es más "mágico" pero menos transparente — para un proyecto de portafolio quiero que el grafo sea visible y entendible.

**¿Por qué Gemini 2.5 Flash y no Claude o GPT?**
Free tier real, latencia baja, contexto largo. Para un demo público es la mejor opción costo/beneficio. El código debe quedar agnóstico al provider (factory en `llm.py`) para poder cambiar después.

**¿Por qué Tavily y no SerpAPI/Brave?**
Diseñada específicamente para agentes — devuelve resultados ya optimizados para LLMs (contenido limpio, sin HTML). Free tier de 1000 búsquedas/mes alcanza para demo.

**¿Por qué Streamlit y no Gradio?**
Más flexible para layouts custom (sidebar con traza + main con informe). Gradio es más rápido para apps simples, pero acá necesitamos UI con varias secciones.

**¿Por qué un supervisor explícito y no agentes que se llaman entre sí?**
El patrón supervisor-workers es más limpio y predecible. Los flujos peer-to-peer (un worker llama a otro) son más difíciles de debuggear y razonar.

---

## 🚨 Reglas importantes para Claude Code

1. **Nunca commitear `.env` ni keys reales**. Verificar `.gitignore` antes de cada commit.
2. **Antes de instalar una dependencia nueva**, preguntar si vale la pena. Mantener `requirements.txt` mínimo.
3. **Si un cambio toca la arquitectura del grafo**, actualizar el diagrama de este archivo.
4. **Al terminar una fase**, actualizar la sección "Estado actual del proyecto" marcando los checkboxes.
5. **Si encuentras un trade-off de diseño no documentado**, agregarlo a "Decisiones de diseño".
6. **Mantener los prompts cortos y específicos**. Un prompt de >50 líneas suele indicar que el agente está haciendo demasiado.
7. **No optimizar prematuramente**. Funcionalidad correcta primero, optimización después.

---

## 📚 Referencias útiles

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Tavily API: https://docs.tavily.com/
- Gemini API: https://ai.google.dev/gemini-api/docs
- Patrón supervisor en LangGraph: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
