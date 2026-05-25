# Research Crew Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un sistema multi-agente que recibe una pregunta y produce un informe ejecutivo en Markdown con citas, coordinando supervisor + 3 workers (researcher, analyst, writer) vía LangGraph, con UI Streamlit que muestra la traza y hace streaming del informe.

**Architecture:** Grafo LangGraph con un supervisor que usa LLM solo en dos modos (PLAN al inicio, EVAL después del Analyst). Aristas fijas para `researcher→analyst`, `analyst→supervisor`, `writer→END`. Aristas condicionales solo desde el supervisor sobre `state["next_agent"]`. Researcher ejecuta búsquedas Tavily en paralelo con `asyncio.gather`. Writer hace streaming token-por-token a la UI.

**Tech Stack:** Python 3.11+, LangGraph, Gemini 2.5 Flash (langchain-google-genai), Tavily, Streamlit, pydantic, pytest, ruff.

**Spec base:** `docs/superpowers/specs/2026-05-24-research-crew-design.md`

---

## File Structure

```
src/
├── config.py              # Constantes y carga de env vars (MAX_ITERATIONS, etc.)
├── state.py               # AgentState, ResearchResult, TraceEvent, NodeError
├── schemas.py             # PlanOutput, EvalOutput, AnalystOutput (pydantic)
├── llm.py                 # get_llm(model_name) factory
├── graph.py               # build_graph() + CLI entry
├── agents/
│   ├── supervisor.py      # nodo supervisor (modos PLAN/EVAL)
│   ├── researcher.py      # Tavily async parallel
│   ├── analyst.py         # síntesis estructurada
│   └── writer.py          # streaming + post-proceso de citas
├── prompts/
│   ├── supervisor.txt     # con secciones # MODE: PLAN y # MODE: EVAL
│   ├── analyst.txt
│   └── writer.txt
└── ui/
    ├── trace.py           # render de TraceEvent
    └── async_bridge.py    # wrapper async→sync para Streamlit

app.py                     # entry de Streamlit (delgado)
tests/                     # tests unitarios + e2e con mocks
```

**Decisión clave de decomposición:** cada agente vive en su propio archivo con un test correspondiente. Los `prompts/*.txt` están separados del código para iterar sin tocar lógica. `app.py` queda mínimo: delega a `src/ui/`.

---

## Task 1: Setup del proyecto

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/agents/__init__.py`
- Create: `src/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py` (esqueleto)

- [ ] **Step 1: Crear `pyproject.toml`**

```toml
[project]
name = "research-crew"
version = "0.1.0"
requires-python = ">=3.11"
description = "Multi-agent research system with LangGraph"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["."]
```

- [ ] **Step 2: Crear `requirements.txt`**

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

- [ ] **Step 3: Crear `.env.example`**

```
GOOGLE_API_KEY=
TAVILY_API_KEY=
MODEL_NAME=gemini-2.5-flash
MAX_ITERATIONS=8
```

- [ ] **Step 4: Crear archivos `__init__.py` vacíos**

Crear archivos vacíos:
- `src/__init__.py`
- `src/agents/__init__.py`
- `src/ui/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Crear `tests/conftest.py` mínimo**

```python
"""Fixtures compartidas para los tests. Se irán agregando conforme avancen los tasks."""
import pytest

# Marcamos todos los tests async automáticamente con asyncio_mode=auto en pyproject.
```

- [ ] **Step 6: Crear venv e instalar dependencias**

Run:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Expected: instala todo sin errores.

- [ ] **Step 7: Verificar setup**

Run: `pytest --collect-only` (no debería haber tests aún, sale OK)
Expected: `collected 0 items`

Run: `ruff check .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt .env.example src/ tests/
git commit -m "chore: setup inicial del proyecto (pyproject, deps, estructura)"
```

---

## Task 2: Config (`src/config.py`)

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_config.py
import os
from unittest.mock import patch

def test_config_loads_defaults():
    from src import config
    assert config.MAX_ITERATIONS == 8
    assert config.MAX_RESEARCH_ROUNDS == 3
    assert config.QUERIES_PER_ROUND == 3
    assert config.TAVILY_MAX_RESULTS == 5
    assert config.MODEL_NAME_DEFAULT == "gemini-2.5-flash"

def test_config_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_ITERATIONS", "12")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-pro")
    # Re-import to pick up env vars
    import importlib
    from src import config
    importlib.reload(config)
    assert config.MAX_ITERATIONS == 12
    assert config.MODEL_NAME_DEFAULT == "gemini-2.5-pro"

def test_config_requires_api_keys_via_check():
    from src.config import check_api_keys, ConfigError
    with patch.dict(os.environ, {}, clear=True):
        try:
            check_api_keys()
            assert False, "should have raised"
        except ConfigError as e:
            assert "GOOGLE_API_KEY" in str(e) or "TAVILY_API_KEY" in str(e)
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implementar `src/config.py`**

```python
"""Configuración centralizada del proyecto.

Lee defaults desde aquí y permite override vía variables de entorno (.env).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Caps del grafo
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "8"))
MAX_RESEARCH_ROUNDS: int = int(os.getenv("MAX_RESEARCH_ROUNDS", "3"))
QUERIES_PER_ROUND: int = int(os.getenv("QUERIES_PER_ROUND", "3"))

# Tavily
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# LLM
MODEL_NAME_DEFAULT: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")


class ConfigError(RuntimeError):
    """Falta configuración requerida (API keys, etc.)."""


def check_api_keys() -> None:
    """Valida que las API keys estén presentes. Lanza ConfigError si faltan."""
    missing = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ConfigError(
            f"Faltan variables de entorno: {', '.join(missing)}. "
            f"Cópialas en .env desde .env.example."
        )
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/test_config.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: módulo config con env vars y validación de API keys"
```

---

## Task 3: State (`src/state.py`)

**Files:**
- Create: `src/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_state.py
from src.state import (
    AgentState, ResearchResult, TraceEvent, NodeError,
    make_initial_state,
)

def test_research_result_typeddict_shape():
    r: ResearchResult = {
        "url": "https://x",
        "title": "T",
        "content": "C",
        "snippet": "S",
        "query": "q",
    }
    assert r["url"] == "https://x"

def test_trace_event_typeddict_shape():
    e: TraceEvent = {
        "timestamp": "2026-05-24T10:00:00",
        "node": "supervisor",
        "level": "info",
        "text": "hello",
    }
    assert e["level"] == "info"

def test_node_error_typeddict_shape():
    e: NodeError = {
        "node": "researcher",
        "iteration": 2,
        "exception_type": "TimeoutError",
        "message": "timeout",
    }
    assert e["iteration"] == 2

def test_make_initial_state_has_expected_defaults():
    s = make_initial_state("¿Qué es X?")
    assert s["question"] == "¿Qué es X?"
    assert s["queries"] == []
    assert s["enough_info"] is None
    assert s["next_agent"] == ""
    assert s["research_results"] == []
    assert s["analysis"] == ""
    assert s["gaps"] == []
    assert s["report"] == ""
    assert s["trace"] == []
    assert s["iteration_count"] == 0
    assert s["research_rounds"] == 0
    assert s["errors"] == []
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_state.py -v`
Expected: `ImportError` o `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/state.py`**

```python
"""Estado compartido del grafo de agentes.

Diseñado para LangGraph: campos acumulativos usan `Annotated[..., add]` como reducer.
Cada agente devuelve solo el diff; el reducer se encarga de fusionar.
"""
from operator import add
from typing import Literal, TypedDict
from typing_extensions import Annotated

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
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/test_state.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/state.py tests/test_state.py
git commit -m "feat: AgentState con TypedDicts y reducers para campos acumulativos"
```

---

## Task 4: Schemas (`src/schemas.py`)

**Files:**
- Create: `src/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from src.schemas import PlanOutput, EvalOutput, AnalystOutput

def test_plan_output_requires_exactly_3_queries():
    p = PlanOutput(queries=["a", "b", "c"])
    assert len(p.queries) == 3

    with pytest.raises(ValidationError):
        PlanOutput(queries=["only one"])
    with pytest.raises(ValidationError):
        PlanOutput(queries=["1", "2", "3", "4"])

def test_eval_output_enough_info_true_allows_empty_queries():
    e = EvalOutput(enough_info=True, queries=[], reasoning="ok")
    assert e.enough_info is True
    assert e.queries == []

def test_eval_output_default_queries_is_empty_list():
    e = EvalOutput(enough_info=True, reasoning="ok")
    assert e.queries == []

def test_analyst_output_basic_shape():
    a = AnalystOutput(analysis="texto…", gaps=["q1", "q2"])
    assert a.analysis == "texto…"
    assert a.gaps == ["q1", "q2"]

def test_analyst_output_gaps_can_be_empty():
    a = AnalystOutput(analysis="…", gaps=[])
    assert a.gaps == []
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_schemas.py -v`
Expected: `ModuleNotFoundError: No module named 'src.schemas'`.

- [ ] **Step 3: Implementar `src/schemas.py`**

```python
"""Schemas pydantic para outputs estructurados de los LLMs.

Se usan con `.with_structured_output(Schema)` de langchain_google_genai.
Compartidos entre supervisor (PlanOutput, EvalOutput) y analyst (AnalystOutput).
"""
from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    """Output del supervisor en modo PLAN: descomposición en 3 sub-queries."""

    queries: list[str] = Field(..., min_length=3, max_length=3)


class EvalOutput(BaseModel):
    """Output del supervisor en modo EVAL: decisión de seguir investigando o cerrar."""

    enough_info: bool
    queries: list[str] = Field(default_factory=list)
    reasoning: str


class AnalystOutput(BaseModel):
    """Output del analyst: síntesis textual + gaps no respondidos."""

    analysis: str
    gaps: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/test_schemas.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/schemas.py tests/test_schemas.py
git commit -m "feat: schemas pydantic para outputs estructurados (PlanOutput, EvalOutput, AnalystOutput)"
```

---

## Task 5: LLM factory (`src/llm.py`)

**Files:**
- Create: `src/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_llm.py
from unittest.mock import patch
from src.llm import get_llm

def test_get_llm_uses_default_model_when_no_arg(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm()
        assert mock_class.called
        kwargs = mock_class.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"

def test_get_llm_uses_provided_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm("gemini-2.5-pro")
        kwargs = mock_class.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-pro"

def test_get_llm_sets_temperature_zero(monkeypatch):
    """Para decisiones del supervisor y síntesis del analyst queremos determinismo."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm()
        kwargs = mock_class.call_args.kwargs
        assert kwargs["temperature"] == 0.0
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_llm.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/llm.py`**

```python
"""Factory de LLM. Centraliza la creación para que el resto del código sea agnóstico al provider.

Si en el futuro cambiamos a Claude/GPT, solo modificamos este archivo.
"""
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import MODEL_NAME_DEFAULT


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    """Construye una instancia de ChatGoogleGenerativeAI.

    Args:
        model_name: nombre del modelo. Si es None usa MODEL_NAME_DEFAULT.
        temperature: 0.0 por default (queremos determinismo en routing y síntesis).

    Returns:
        Una instancia configurada lista para usar.
    """
    model = model_name or MODEL_NAME_DEFAULT
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
    )
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/test_llm.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: factory de LLM (Gemini) con default desde config"
```

---

## Task 6: Prompts del Supervisor

**Files:**
- Create: `src/prompts/__init__.py` (vacío)
- Create: `src/prompts/supervisor.txt`

- [ ] **Step 1: Crear `src/prompts/__init__.py` vacío**

- [ ] **Step 2: Crear `src/prompts/supervisor.txt`**

```
# MODE: PLAN

Eres un supervisor de investigación. Recibes una pregunta y debes descomponerla en 3 sub-queries de búsqueda web que, ejecutadas en paralelo, cubran el espacio de la pregunta sin solaparse.

PREGUNTA: {question}

Reglas:
- Exactamente 3 queries, no más, no menos.
- Cada query debe ser específica y autosuficiente (no asumir contexto de las otras).
- Si la pregunta es en español, las queries deben ser en español.
- Si la pregunta es en inglés, en inglés.
- Evita queries genéricas como "qué es X" si la pregunta ya lo da por entendido.

Devuelve un JSON con campo `queries: list[str]` con exactamente 3 elementos.

---

# MODE: EVAL

Eres un supervisor evaluando si el análisis hecho hasta ahora basta para responder la pregunta original, o si vale la pena otra ronda de investigación.

PREGUNTA: {question}

ANÁLISIS ACTUAL:
{analysis}

GAPS REPORTADOS POR EL ANALYST:
{gaps}

RONDA DE RESEARCH ACTUAL: {research_rounds} (cap = {max_rounds})

Decide:
- `enough_info: true` si el análisis ya cubre la pregunta con razonable profundidad y los gaps no son críticos.
- `enough_info: false` si hay gaps críticos Y aún no llegamos al cap de rondas. En ese caso, genera 3 nuevas queries enfocadas en cerrar esos gaps.

Reglas:
- Si `enough_info: true`, devuelve `queries: []`.
- Si `enough_info: false`, devuelve exactamente 3 queries nuevas (no repetir las anteriores).
- Las queries deben atacar específicamente los gaps, no la pregunta original.

Devuelve JSON con: `enough_info: bool`, `queries: list[str]` (vacía o 3 elementos), `reasoning: str` (1-2 frases).
```

- [ ] **Step 3: Verificar que existe y se puede leer**

Run: `python -c "from pathlib import Path; print(Path('src/prompts/supervisor.txt').read_text(encoding='utf-8')[:50])"`
Expected: imprime `# MODE: PLAN\n\nEres un supervisor de investigación. R...`

- [ ] **Step 4: Commit**

```bash
git add src/prompts/__init__.py src/prompts/supervisor.txt
git commit -m "feat: prompt del supervisor con modos PLAN y EVAL"
```

---

## Task 7: Supervisor agent — modo PLAN

**Files:**
- Create: `src/agents/supervisor.py`
- Test: `tests/test_supervisor.py`

- [ ] **Step 1: Escribir test failing del modo PLAN**

```python
# tests/test_supervisor.py
from unittest.mock import MagicMock, patch
from src.state import make_initial_state
from src.schemas import PlanOutput
from src.agents.supervisor import supervisor_node


def _fake_llm_returning(structured_response):
    """Crea un mock que imita .with_structured_output(...).invoke(...)."""
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(return_value=structured_response)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def test_supervisor_plan_mode_generates_3_queries():
    state = make_initial_state("¿Cuál es el estado de la IA en Chile?")
    fake = _fake_llm_returning(PlanOutput(queries=["q1", "q2", "q3"]))
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["queries"] == ["q1", "q2", "q3"]
    assert diff["next_agent"] == "researcher"
    assert diff["iteration_count"] == 1
    assert len(diff["trace"]) == 1
    assert diff["trace"][0]["node"] == "supervisor"
    assert "PLAN" in diff["trace"][0]["text"]
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_supervisor.py -v`
Expected: `ModuleNotFoundError: No module named 'src.agents.supervisor'`.

- [ ] **Step 3: Implementar `src/agents/supervisor.py` con modo PLAN**

```python
"""Supervisor del grafo. Único agente con LLM.

Se invoca en dos momentos:
1. Al arranque (estado vacío) → modo PLAN: descompone la pregunta en 3 sub-queries.
2. Después del Analyst (analysis seteado) → modo EVAL: decide si basta o pedir más research.

El código elige el modo según el estado; ambos prompts viven en `prompts/supervisor.txt`.
"""
from datetime import datetime, timezone
from pathlib import Path

from src.config import MAX_ITERATIONS, MAX_RESEARCH_ROUNDS
from src.llm import get_llm
from src.schemas import EvalOutput, PlanOutput
from src.state import AgentState, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.txt"


def _load_sections() -> tuple[str, str]:
    """Lee el archivo de prompt y separa por '# MODE: PLAN' y '# MODE: EVAL'."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # Split por '# MODE:' header; ignora vacío inicial
    parts = text.split("# MODE: ")
    sections = {}
    for part in parts[1:]:
        name, body = part.split("\n", 1)
        sections[name.strip()] = body.strip()
    return sections["PLAN"], sections["EVAL"]


_PLAN_TEMPLATE, _EVAL_TEMPLATE = _load_sections()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trace(node: str, level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node=node, level=level, text=text)  # type: ignore[arg-type]


def supervisor_node(state: AgentState) -> dict:
    """Nodo supervisor. Detecta el modo y ejecuta."""
    # MODO PLAN: arranque (sin queries ni resultados)
    if not state["queries"] and not state["research_results"]:
        return _run_plan(state)

    # MODO EVAL: después del analyst
    if state["analysis"] and state["enough_info"] is None:
        return _run_eval(state)

    # No debería llegar aquí; safety fallback
    return {
        "next_agent": "writer",
        "iteration_count": state["iteration_count"] + 1,
        "trace": [_trace("supervisor", "warn", "estado inesperado, forzando writer")],
    }


def _run_plan(state: AgentState) -> dict:
    """Genera 3 sub-queries con LLM."""
    llm = get_llm()
    structured = llm.with_structured_output(PlanOutput)
    prompt = _PLAN_TEMPLATE.format(question=state["question"])
    result: PlanOutput = structured.invoke(prompt)
    return {
        "queries": result.queries,
        "next_agent": "researcher",
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace("supervisor", "info", f"PLAN: {len(result.queries)} sub-queries generadas")
        ],
    }


def _run_eval(state: AgentState) -> dict:
    """Evalúa si el análisis basta. Si no, genera follow-up queries."""
    # Caps: si ya estamos al límite, fuerza writer sin llamar LLM
    if (
        state["iteration_count"] >= MAX_ITERATIONS - 1
        or state["research_rounds"] >= MAX_RESEARCH_ROUNDS
    ):
        return {
            "enough_info": True,
            "next_agent": "writer",
            "iteration_count": state["iteration_count"] + 1,
            "trace": [_trace("supervisor", "warn", "cap alcanzado, forzando cierre")],
        }

    llm = get_llm()
    structured = llm.with_structured_output(EvalOutput)
    prompt = _EVAL_TEMPLATE.format(
        question=state["question"],
        analysis=state["analysis"],
        gaps="\n- " + "\n- ".join(state["gaps"]) if state["gaps"] else "(ninguno)",
        research_rounds=state["research_rounds"],
        max_rounds=MAX_RESEARCH_ROUNDS,
    )
    result: EvalOutput = structured.invoke(prompt)

    diff: dict = {
        "enough_info": result.enough_info,
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace(
                "supervisor",
                "info",
                f"EVAL: enough_info={result.enough_info} — {result.reasoning}",
            )
        ],
    }
    if result.enough_info:
        diff["next_agent"] = "writer"
    else:
        diff["queries"] = result.queries
        diff["next_agent"] = "researcher"
    return diff
```

- [ ] **Step 4: Verificar que pasa el test de PLAN**

Run: `pytest tests/test_supervisor.py::test_supervisor_plan_mode_generates_3_queries -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: supervisor en modo PLAN (genera 3 sub-queries con LLM)"
```

---

## Task 8: Supervisor agent — modo EVAL

**Files:**
- Modify: `tests/test_supervisor.py` (agregar tests)

- [ ] **Step 1: Agregar tests EVAL al archivo existente**

```python
# Append a tests/test_supervisor.py
from src.schemas import EvalOutput


def test_supervisor_eval_mode_enough_info_routes_to_writer():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "síntesis suficiente"
    state["gaps"] = []
    state["research_rounds"] = 1

    fake = _fake_llm_returning(EvalOutput(enough_info=True, queries=[], reasoning="ya basta"))
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert "queries" not in diff or diff.get("queries") == []
    assert "EVAL" in diff["trace"][0]["text"]


def test_supervisor_eval_mode_insufficient_routes_to_researcher_with_new_queries():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "síntesis incompleta"
    state["gaps"] = ["falta dato X", "falta dato Y"]
    state["research_rounds"] = 1

    fake = _fake_llm_returning(
        EvalOutput(enough_info=False, queries=["new1", "new2", "new3"], reasoning="faltan datos")
    )
    with patch("src.agents.supervisor.get_llm", return_value=fake):
        diff = supervisor_node(state)
    assert diff["enough_info"] is False
    assert diff["next_agent"] == "researcher"
    assert diff["queries"] == ["new1", "new2", "new3"]


def test_supervisor_eval_bypass_when_iteration_cap_reached():
    """Si iteration_count >= MAX_ITERATIONS - 1, fuerza writer SIN llamar LLM."""
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "algo"
    state["iteration_count"] = 7  # MAX_ITERATIONS - 1 = 7
    state["research_rounds"] = 1

    fake = _fake_llm_returning(EvalOutput(enough_info=False, queries=["a", "b", "c"], reasoning="x"))
    with patch("src.agents.supervisor.get_llm", return_value=fake) as mock_get_llm:
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert mock_get_llm.called is False, "no debería invocar LLM si caps alcanzados"
    assert "cap" in diff["trace"][0]["text"].lower()


def test_supervisor_eval_bypass_when_research_rounds_cap_reached():
    state = make_initial_state("Q?")
    state["queries"] = ["q1", "q2", "q3"]
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q1"}
    ]
    state["analysis"] = "algo"
    state["iteration_count"] = 4
    state["research_rounds"] = 3  # MAX_RESEARCH_ROUNDS = 3

    fake = _fake_llm_returning(EvalOutput(enough_info=False, queries=["a", "b", "c"], reasoning="x"))
    with patch("src.agents.supervisor.get_llm", return_value=fake) as mock_get_llm:
        diff = supervisor_node(state)
    assert diff["enough_info"] is True
    assert diff["next_agent"] == "writer"
    assert mock_get_llm.called is False
```

- [ ] **Step 2: Verificar que pasan**

Run: `pytest tests/test_supervisor.py -v`
Expected: 5 tests pass (1 de PLAN + 4 de EVAL).

- [ ] **Step 3: Commit**

```bash
git add tests/test_supervisor.py
git commit -m "test: cobertura del supervisor en modo EVAL incluyendo caps bypass"
```

---

## Task 9: Researcher agent (Tavily async parallel)

**Files:**
- Create: `src/agents/researcher.py`
- Test: `tests/test_researcher.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_researcher.py
import asyncio
from unittest.mock import patch
import pytest

from src.state import make_initial_state
from src.agents.researcher import researcher_node


@pytest.fixture
def state_with_queries():
    s = make_initial_state("¿Q?")
    s["queries"] = ["query A", "query B", "query C"]
    return s


def _make_tavily_response(query_to_results: dict):
    """Helper: devuelve función mock que simula TavilyClient.search."""
    def fake_search(query, **kwargs):
        if isinstance(query_to_results.get(query), Exception):
            raise query_to_results[query]
        return {
            "results": query_to_results.get(query, []),
        }
    return fake_search


def test_researcher_executes_all_queries_in_parallel(state_with_queries):
    responses = {
        "query A": [{"url": "uA", "title": "tA", "content": "cA"}],
        "query B": [{"url": "uB", "title": "tB", "content": "cB"}],
        "query C": [{"url": "uC", "title": "tC", "content": "cC"}],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert len(diff["research_results"]) == 3
    queries_seen = {r["query"] for r in diff["research_results"]}
    assert queries_seen == {"query A", "query B", "query C"}
    assert diff["research_rounds"] == 1
    assert diff["iteration_count"] == 1
    assert "next_agent" not in diff, "researcher no debe escribir next_agent (arista fija)"


def test_researcher_handles_partial_failure(state_with_queries):
    responses = {
        "query A": [{"url": "uA", "title": "tA", "content": "cA"}],
        "query B": RuntimeError("Tavily 500"),
        "query C": [{"url": "uC", "title": "tC", "content": "cC"}],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert len(diff["research_results"]) == 2
    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "researcher"


def test_researcher_handles_total_failure(state_with_queries):
    responses = {
        "query A": RuntimeError("Tavily 500"),
        "query B": RuntimeError("Tavily 500"),
        "query C": RuntimeError("Tavily 500"),
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    assert diff["research_results"] == []
    assert len(diff["errors"]) == 3
    # Aún sigue al analyst (arista fija) — no se setea next_agent
    assert "next_agent" not in diff


def test_researcher_normalizes_result_shape(state_with_queries):
    responses = {
        "query A": [
            {"url": "uA", "title": "tA", "content": "C" * 500, "snippet": "snippetA"},
        ],
        "query B": [],
        "query C": [],
    }
    fake = _make_tavily_response(responses)
    with patch("src.agents.researcher.TavilyClient") as mock_class:
        mock_class.return_value.search.side_effect = fake
        diff = asyncio.run(researcher_node(state_with_queries))

    r = diff["research_results"][0]
    assert set(r.keys()) == {"url", "title", "content", "snippet", "query"}
    assert r["query"] == "query A"
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_researcher.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/agents/researcher.py`**

```python
"""Researcher: ejecuta búsquedas Tavily en paralelo. Sin LLM.

`TavilyClient.search` es sincrónico; lo envolvemos con `asyncio.to_thread`
para usarlo con `asyncio.gather`.
"""
import asyncio
import os
from datetime import datetime, timezone

from tavily import TavilyClient

from src.config import TAVILY_MAX_RESULTS
from src.state import AgentState, NodeError, ResearchResult, TraceEvent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trace(level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node="researcher", level=level, text=text)  # type: ignore[arg-type]


def _normalize(raw_result: dict, query: str) -> ResearchResult:
    """Pasa de un dict crudo de Tavily a ResearchResult."""
    return ResearchResult(
        url=raw_result.get("url", ""),
        title=raw_result.get("title", ""),
        content=raw_result.get("content", ""),
        snippet=raw_result.get("snippet") or (raw_result.get("content", "")[:200]),
        query=query,
    )


async def _search_one(client: TavilyClient, query: str) -> list[dict]:
    """Wrapper async sobre TavilyClient.search."""
    return await asyncio.to_thread(
        client.search,
        query,
        max_results=TAVILY_MAX_RESULTS,
        search_depth="basic",
    )


async def researcher_node(state: AgentState) -> dict:
    """Ejecuta las queries del state en paralelo con Tavily."""
    queries = state["queries"]
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    coros = [_search_one(client, q) for q in queries]
    results = await asyncio.gather(*coros, return_exceptions=True)

    new_results: list[ResearchResult] = []
    new_errors: list[NodeError] = []
    trace: list[TraceEvent] = []

    for query, result in zip(queries, results, strict=False):
        if isinstance(result, Exception):
            new_errors.append(
                NodeError(
                    node="researcher",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(result).__name__,
                    message=str(result),
                )
            )
            trace.append(_trace("warn", f"falló query '{query}': {type(result).__name__}"))
            continue

        raw_items = result.get("results", []) if isinstance(result, dict) else []
        for item in raw_items:
            new_results.append(_normalize(item, query))
        trace.append(_trace("info", f"query '{query}' devolvió {len(raw_items)} resultados"))

    diff: dict = {
        "research_results": new_results,
        "research_rounds": state["research_rounds"] + 1,
        "iteration_count": state["iteration_count"] + 1,
        "trace": trace,
    }
    if new_errors:
        diff["errors"] = new_errors
    return diff
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_researcher.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/researcher.py tests/test_researcher.py
git commit -m "feat: researcher con Tavily async parallel y manejo de fallos parciales/totales"
```

---

## Task 10: Analyst prompt

**Files:**
- Create: `src/prompts/analyst.txt`

- [ ] **Step 1: Crear `src/prompts/analyst.txt`**

```
Eres un analista de investigación. Sintetiza los resultados de búsqueda para responder la pregunta original.

PREGUNTA: {question}

RONDA DE INVESTIGACIÓN: {research_rounds}

RESULTADOS DISPONIBLES:
{results}

Tu tarea:
1. Identifica los hallazgos clave que se desprenden de los resultados.
2. Detecta contradicciones entre fuentes si las hay.
3. Identifica gaps: preguntas relevantes que los resultados no responden.
4. Resume en `analysis` (200-400 palabras): hallazgos + contradicciones + contexto necesario.
5. Lista los gaps en `gaps` (puede ser []). Cada gap es una pregunta concreta no resuelta.

Reglas:
- Apóyate solo en lo que aparece en los resultados; no inventes datos.
- Si los resultados están vacíos o son irrelevantes, di explícitamente que no hay información suficiente y pon la pregunta original como único gap.
- Idioma del análisis = idioma de la pregunta.

Devuelve JSON con: `analysis: str`, `gaps: list[str]`.
```

- [ ] **Step 2: Commit**

```bash
git add src/prompts/analyst.txt
git commit -m "feat: prompt del analyst"
```

---

## Task 11: Analyst agent

**Files:**
- Create: `src/agents/analyst.py`
- Test: `tests/test_analyst.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_analyst.py
from unittest.mock import MagicMock, patch

from src.state import make_initial_state
from src.schemas import AnalystOutput
from src.agents.analyst import analyst_node


def _fake_llm_returning(structured_response):
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(return_value=structured_response)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def test_analyst_returns_analysis_and_gaps():
    state = make_initial_state("¿Q?")
    state["research_results"] = [
        {"url": "uA", "title": "tA", "content": "cA", "snippet": "sA", "query": "qA"},
        {"url": "uB", "title": "tB", "content": "cB", "snippet": "sB", "query": "qB"},
    ]

    fake = _fake_llm_returning(
        AnalystOutput(analysis="síntesis textual", gaps=["gap1"])
    )
    with patch("src.agents.analyst.get_llm", return_value=fake):
        diff = analyst_node(state)

    assert diff["analysis"] == "síntesis textual"
    assert diff["gaps"] == ["gap1"]
    assert diff["iteration_count"] == 1
    assert "next_agent" not in diff
    assert diff["trace"][0]["node"] == "analyst"


def test_analyst_handles_empty_research_results():
    """Con research_results=[], el agente debe devolver disclaimer + gap original."""
    state = make_initial_state("¿Cuál es X?")
    state["research_results"] = []

    fake = _fake_llm_returning(
        AnalystOutput(analysis="no hay info suficiente", gaps=["¿Cuál es X?"])
    )
    with patch("src.agents.analyst.get_llm", return_value=fake):
        diff = analyst_node(state)

    assert diff["analysis"] == "no hay info suficiente"
    assert diff["gaps"] == ["¿Cuál es X?"]


def test_analyst_does_not_send_full_content_to_llm():
    """El prompt debe contener title/snippet/url, pero NO el content completo."""
    state = make_initial_state("¿Q?")
    long_content = "X" * 2000
    state["research_results"] = [
        {"url": "uA", "title": "tA", "content": long_content, "snippet": "snippet corto", "query": "qA"},
    ]
    captured_prompt = {}

    def capture_invoke(prompt):
        captured_prompt["text"] = prompt
        return AnalystOutput(analysis="x", gaps=[])

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=capture_invoke)
    llm.with_structured_output = MagicMock(return_value=structured)
    with patch("src.agents.analyst.get_llm", return_value=llm):
        analyst_node(state)

    assert long_content not in captured_prompt["text"], (
        "el prompt no debe incluir content completo"
    )
    assert "snippet corto" in captured_prompt["text"]
    assert "tA" in captured_prompt["text"]
    assert "uA" in captured_prompt["text"]
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_analyst.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/agents/analyst.py`**

```python
"""Analyst: sintetiza research_results acumulados y reporta gaps.

Solo recibe title + snippet + url + query en el prompt (no `content` completo).
"""
from datetime import datetime, timezone
from pathlib import Path

from src.llm import get_llm
from src.schemas import AnalystOutput
from src.state import AgentState, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trace(level: str, text: str) -> TraceEvent:
    return TraceEvent(timestamp=_now(), node="analyst", level=level, text=text)  # type: ignore[arg-type]


def _format_results(results: list[dict]) -> str:
    """Convierte la lista de resultados a un bloque de texto compacto."""
    if not results:
        return "(no hay resultados)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['title']} — {r['url']}\n    snippet: {r['snippet']}\n    (de query: {r['query']})"
        )
    return "\n".join(lines)


def analyst_node(state: AgentState) -> dict:
    llm = get_llm()
    structured = llm.with_structured_output(AnalystOutput)
    prompt = _PROMPT.format(
        question=state["question"],
        research_rounds=state["research_rounds"],
        results=_format_results(state["research_results"]),
    )
    result: AnalystOutput = structured.invoke(prompt)
    return {
        "analysis": result.analysis,
        "gaps": result.gaps,
        "iteration_count": state["iteration_count"] + 1,
        "trace": [
            _trace(
                "info",
                f"síntesis completa ({len(state['research_results'])} resultados, "
                f"{len(result.gaps)} gaps)",
            )
        ],
    }
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_analyst.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/analyst.py tests/test_analyst.py
git commit -m "feat: analyst con síntesis estructurada y prompt sin content completo"
```

---

## Task 12: Writer prompt

**Files:**
- Create: `src/prompts/writer.txt`

- [ ] **Step 1: Crear `src/prompts/writer.txt`**

```
Eres un redactor de informes ejecutivos. Recibes una síntesis y una lista de fuentes; produces un informe en Markdown.

PREGUNTA ORIGINAL: {question}

SÍNTESIS DEL ANALYST:
{analysis}

FUENTES DISPONIBLES:
{sources}

Reglas de formato:
1. Idioma del informe = idioma de la pregunta.
2. Largo: 600-1000 palabras.
3. Estructura obligatoria (con estos headers exactos en Markdown):
   ## TL;DR
   (3 bullets con los puntos centrales)

   ## Hallazgos clave
   (3-5 subsecciones con header `### nombre del hallazgo`)

   ## Contradicciones y gaps
   (si las hay; si no, escribe "Sin contradicciones identificadas.")

   ## Conclusión
   (1-2 párrafos cerrando la respuesta)

4. Citas: cada vez que afirmes algo basado en una fuente, agrega `[n]` inline donde `n` es el número de la fuente en la lista provista. Ejemplo: "La ley fue presentada en 2023 [1] y modificada en 2024 [3]."
5. NO inventes fuentes ni URLs. Solo usa los números `[n]` de la lista provista.
6. NO incluyas una sección "Fuentes" al final. Se construye por código a partir de los `[n]` que cites.

Responde solo con el Markdown del informe.
```

- [ ] **Step 2: Commit**

```bash
git add src/prompts/writer.txt
git commit -m "feat: prompt del writer con estructura y reglas de citado"
```

---

## Task 13: Writer agent (con post-proceso de citas)

**Files:**
- Create: `src/agents/writer.py`
- Test: `tests/test_writer.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_writer.py
from unittest.mock import MagicMock, patch

from src.state import make_initial_state
from src.agents.writer import writer_node, build_sources_section, extract_cited_indices


def test_extract_cited_indices_finds_all_unique_in_order():
    text = "blah [1] blah [3] more [1] then [2] end"
    assert extract_cited_indices(text) == [1, 3, 2]


def test_extract_cited_indices_empty_when_no_citations():
    assert extract_cited_indices("texto sin citas") == []


def test_build_sources_section_uses_only_cited():
    results = [
        {"url": "uA", "title": "tA", "content": "", "snippet": "", "query": "qA"},
        {"url": "uB", "title": "tB", "content": "", "snippet": "", "query": "qB"},
        {"url": "uC", "title": "tC", "content": "", "snippet": "", "query": "qC"},
    ]
    cited = [1, 3]
    section = build_sources_section(cited, results)
    assert "[1] tA" in section
    assert "uA" in section
    assert "[3] tC" in section
    assert "uC" in section
    assert "tB" not in section
    assert "uB" not in section


def test_build_sources_section_ignores_out_of_range_indices():
    results = [{"url": "uA", "title": "tA", "content": "", "snippet": "", "query": "qA"}]
    cited = [1, 2, 99]
    section = build_sources_section(cited, results)
    assert "[1] tA" in section
    # [2] y [99] no deben aparecer
    assert "[2]" not in section
    assert "[99]" not in section


def test_writer_node_produces_report_with_sources_appended():
    state = make_initial_state("¿Q?")
    state["analysis"] = "síntesis"
    state["research_results"] = [
        {"url": "uA", "title": "tA", "content": "", "snippet": "", "query": "qA"},
        {"url": "uB", "title": "tB", "content": "", "snippet": "", "query": "qB"},
    ]

    fake_text = "## TL;DR\n- punto importante [1]\n\n## Hallazgos clave\n### Uno\nblah [2]"
    fake_llm = MagicMock()
    fake_llm.invoke = MagicMock(return_value=MagicMock(content=fake_text))

    with patch("src.agents.writer.get_llm", return_value=fake_llm):
        diff = writer_node(state)

    assert "## TL;DR" in diff["report"]
    assert "## Fuentes" in diff["report"]
    assert "[1] tA" in diff["report"]
    assert "[2] tB" in diff["report"]
    assert "next_agent" not in diff
    assert diff["iteration_count"] == 1
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_writer.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implementar `src/agents/writer.py`**

```python
"""Writer: genera el informe final en Markdown.

Post-procesamiento: la sección 'Fuentes' la construye este módulo (no el LLM)
a partir de las citas `[n]` que efectivamente aparecen en el texto generado.
Esto evita alucinación de URLs.
"""
import re
from datetime import datetime, timezone
from pathlib import Path

from src.llm import get_llm
from src.state import AgentState, TraceEvent

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer.txt"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_writer.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/writer.py tests/test_writer.py
git commit -m "feat: writer con post-proceso de citas (sección Fuentes construida por código)"
```

---

## Task 14: Graph builder (`src/graph.py`)

**Files:**
- Create: `src/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Escribir test e2e con mocks**

```python
# tests/test_graph.py
import asyncio
from unittest.mock import MagicMock, patch

from src.state import make_initial_state
from src.schemas import PlanOutput, EvalOutput, AnalystOutput
from src.graph import build_graph


def _fake_llm_sequence(*responses):
    """LLM mock cuyo .with_structured_output().invoke() devuelve responses en orden,
    y cuyo .invoke() (sin estructurar, usado por writer) devuelve una respuesta de texto fija."""
    invocations = {"count": 0}

    def structured_invoke(prompt):
        idx = invocations["count"]
        invocations["count"] += 1
        return responses[idx]

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=structured_invoke)
    llm.with_structured_output = MagicMock(return_value=structured)
    llm.invoke = MagicMock(return_value=MagicMock(content="## TL;DR\n- punto [1]\n\n## Hallazgos clave\n### Uno\nblah [1]"))
    return llm


def _fake_tavily_response():
    def fake_search(query, **kwargs):
        return {
            "results": [
                {"url": f"https://x/{query}", "title": f"t-{query}", "content": "c", "snippet": "s"}
            ]
        }
    return fake_search


def test_graph_single_round_ends_in_writer():
    """Flujo feliz: supervisor PLAN → researcher → analyst → supervisor EVAL (enough_info=True) → writer → END"""
    state = make_initial_state("¿Q?")

    plan = PlanOutput(queries=["q1", "q2", "q3"])
    analyst = AnalystOutput(analysis="síntesis ok", gaps=[])
    eval_ok = EvalOutput(enough_info=True, queries=[], reasoning="ya basta")

    fake_llm = _fake_llm_sequence(plan, analyst, eval_ok)
    fake_search = _fake_tavily_response()

    with (
        patch("src.agents.supervisor.get_llm", return_value=fake_llm),
        patch("src.agents.analyst.get_llm", return_value=fake_llm),
        patch("src.agents.writer.get_llm", return_value=fake_llm),
        patch("src.agents.researcher.TavilyClient") as mock_tv,
    ):
        mock_tv.return_value.search.side_effect = fake_search
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke(state))

    assert final_state["report"] != ""
    assert "## TL;DR" in final_state["report"]
    assert final_state["iteration_count"] <= 8
    assert final_state["research_rounds"] == 1


def test_graph_re_research_then_close():
    """Flujo con re-research: supervisor PLAN → researcher → analyst → supervisor EVAL (insufficient) →
    researcher → analyst → supervisor EVAL (enough) → writer → END"""
    state = make_initial_state("¿Q?")

    plan = PlanOutput(queries=["q1", "q2", "q3"])
    analyst1 = AnalystOutput(analysis="parcial", gaps=["falta X"])
    eval_insufficient = EvalOutput(
        enough_info=False, queries=["new1", "new2", "new3"], reasoning="falta info"
    )
    analyst2 = AnalystOutput(analysis="ahora sí completo", gaps=[])
    eval_ok = EvalOutput(enough_info=True, queries=[], reasoning="ya")

    fake_llm = _fake_llm_sequence(plan, analyst1, eval_insufficient, analyst2, eval_ok)
    fake_search = _fake_tavily_response()

    with (
        patch("src.agents.supervisor.get_llm", return_value=fake_llm),
        patch("src.agents.analyst.get_llm", return_value=fake_llm),
        patch("src.agents.writer.get_llm", return_value=fake_llm),
        patch("src.agents.researcher.TavilyClient") as mock_tv,
    ):
        mock_tv.return_value.search.side_effect = fake_search
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke(state))

    assert final_state["report"] != ""
    assert final_state["research_rounds"] == 2
    assert final_state["enough_info"] is True
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_graph.py -v`
Expected: `ModuleNotFoundError: No module named 'src.graph'`.

- [ ] **Step 3: Implementar `src/graph.py`**

```python
"""Construcción del StateGraph y entry point CLI.

Topología:
  - Aristas fijas: START→supervisor, researcher→analyst, analyst→supervisor, writer→END
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
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_graph.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Correr la suite completa para asegurar que nada rompió**

Run: `pytest tests/ -v`
Expected: todos los tests pasan (~22 tests aprox).

- [ ] **Step 6: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: build_graph con aristas fijas + condicionales + CLI entry"
```

---

## Task 15: UI helpers — async bridge

**Files:**
- Create: `src/ui/async_bridge.py`
- Test: `tests/test_async_bridge.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_async_bridge.py
from src.ui.async_bridge import run_graph_streaming


def test_run_graph_streaming_yields_state_and_token_events(monkeypatch):
    """Verifica que el bridge convierte graph.astream en un sync generator que yieldea tuples."""
    # Mock del grafo: simula 2 eventos updates + 1 token + 1 update final
    async def fake_astream(initial_state, stream_mode):
        yield ("updates", {"supervisor": {"queries": ["q1"]}})
        yield ("updates", {"researcher": {"research_results": []}})
        yield ("messages", (type("Tok", (), {"content": "Hola"})(), {"langgraph_node": "writer"}))
        yield ("messages", (type("Tok", (), {"content": "Mundo"})(), {"langgraph_node": "writer"}))
        yield ("updates", {"writer": {"report": "final"}})

    class FakeGraph:
        astream = staticmethod(fake_astream)

    monkeypatch.setattr("src.ui.async_bridge.build_graph", lambda: FakeGraph())

    events = list(run_graph_streaming("pregunta"))
    state_events = [e for e in events if e[0] == "state"]
    token_events = [e for e in events if e[0] == "token"]

    assert len(state_events) == 3
    assert len(token_events) == 2
    assert token_events[0][1] == "Hola"
    assert token_events[1][1] == "Mundo"


def test_run_graph_streaming_filters_non_writer_messages(monkeypatch):
    """Tokens de otros nodos (no writer) deben ser ignorados."""
    async def fake_astream(initial_state, stream_mode):
        yield ("messages", (type("Tok", (), {"content": "X"})(), {"langgraph_node": "supervisor"}))
        yield ("messages", (type("Tok", (), {"content": "Y"})(), {"langgraph_node": "writer"}))

    class FakeGraph:
        astream = staticmethod(fake_astream)

    monkeypatch.setattr("src.ui.async_bridge.build_graph", lambda: FakeGraph())

    tokens = [e[1] for e in run_graph_streaming("q") if e[0] == "token"]
    assert tokens == ["Y"]
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_async_bridge.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/ui/async_bridge.py`**

```python
"""Bridge async→sync para Streamlit.

Streamlit no soporta `async for` directamente en el main thread. Este módulo expone
un generador sync que internamente corre `graph.astream` en un loop async y va
yieldando eventos. Estrategia: bombearlos a una asyncio.Queue, y consumirla con
asyncio.run mediante una corutina recolectora — simple y suficiente para demo.
"""
import asyncio
from typing import Generator

from src.graph import build_graph
from src.state import make_initial_state


def run_graph_streaming(
    question: str,
) -> Generator[tuple[str, object], None, None]:
    """Generador sync que yieldea eventos del grafo.

    Yields:
        ("state", payload_dict)  — un update con el diff de algún nodo.
        ("token", str)            — un token de streaming del writer.
    """
    graph = build_graph()
    initial = make_initial_state(question)

    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    async def producer():
        try:
            async for event_type, payload in graph.astream(
                initial, stream_mode=["updates", "messages"]
            ):
                if event_type == "updates":
                    await queue.put(("state", payload))
                elif event_type == "messages":
                    token, metadata = payload
                    if metadata.get("langgraph_node") == "writer":
                        await queue.put(("token", token.content))
        finally:
            await queue.put(DONE)

    async def consume_and_yield():
        task = asyncio.create_task(producer())
        items: list[tuple[str, object]] = []
        while True:
            item = await queue.get()
            if item is DONE:
                break
            items.append(item)
        await task
        return items

    items = asyncio.run(consume_and_yield())
    for item in items:
        yield item
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_async_bridge.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ui/async_bridge.py tests/test_async_bridge.py
git commit -m "feat: async→sync bridge para Streamlit (yields state y token events)"
```

---

## Task 16: UI trace renderer

**Files:**
- Create: `src/ui/trace.py`
- Test: `tests/test_trace.py`

- [ ] **Step 1: Escribir test failing**

```python
# tests/test_trace.py
from src.ui.trace import render_trace_event, COLOR_BY_LEVEL


def test_render_trace_event_info_has_timestamp_and_node_and_text():
    event = {
        "timestamp": "2026-05-24T10:00:00",
        "node": "supervisor",
        "level": "info",
        "text": "PLAN: 3 queries",
    }
    md = render_trace_event(event)
    assert "10:00:00" in md
    assert "SUPERVISOR" in md
    assert "PLAN: 3 queries" in md


def test_render_trace_event_uses_color_by_level():
    event_info = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "info", "text": "t"}
    event_warn = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "warn", "text": "t"}
    event_error = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "error", "text": "t"}

    md_info = render_trace_event(event_info)
    md_warn = render_trace_event(event_warn)
    md_error = render_trace_event(event_error)

    assert COLOR_BY_LEVEL["info"] in md_info
    assert COLOR_BY_LEVEL["warn"] in md_warn
    assert COLOR_BY_LEVEL["error"] in md_error
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_trace.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `src/ui/trace.py`**

```python
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
```

- [ ] **Step 4: Verificar que pasan los tests**

Run: `pytest tests/test_trace.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ui/trace.py tests/test_trace.py
git commit -m "feat: render de TraceEvent a Markdown coloreado"
```

---

## Task 17: Streamlit app (`app.py`)

**Files:**
- Create: `app.py`

> Nota: la UI de Streamlit no se cubre con tests automáticos. Se valida manualmente al final.

- [ ] **Step 1: Implementar `app.py`**

```python
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


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Research Crew")
    st.caption("Sistema multi-agente con LangGraph")

    st.divider()
    st.subheader("📋 Traza")
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
        with st.expander(f"⚠ Errores ({len(st.session_state.errors)})"):
            for e in st.session_state.errors:
                st.code(f"{e['node']} [iter {e['iteration']}]: {e['exception_type']}: {e['message']}")


# ─── Main ──────────────────────────────────────────────────────────────────
st.title("Research Crew")

# Validar API keys al cargar
try:
    check_api_keys()
except ConfigError as e:
    st.error(f"⚙ Configuración faltante: {e}")
    st.stop()


question = st.text_input(
    "Pregunta",
    placeholder="¿Sobre qué quieres investigar?",
    disabled=st.session_state.is_running,
    label_visibility="collapsed",
)

if st.button("🔍 Investigar", disabled=st.session_state.is_running or not question):
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
            for node_name, diff in payload.items():
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
            "💾 Descargar informe (.md)",
            data=st.session_state.report,
            file_name="research-crew-report.md",
            mime="text/markdown",
        )

    st.session_state.is_running = False
    st.rerun()

elif st.session_state.report and not st.session_state.is_running:
    # Mostrar último report al recargar la página
    st.markdown(st.session_state.report)
```

- [ ] **Step 2: Verificar que Streamlit arranca sin errores de sintaxis**

Run: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Smoke test: levantar la app (con `.env` configurado)**

Esto requiere `.env` con `GOOGLE_API_KEY` y `TAVILY_API_KEY` reales. Si no las tienes, salta este paso y verifica con una pregunta dummy luego.

Run: `streamlit run app.py`
Expected: abre browser en localhost:8501, se ve la UI sin errores. Ingresar pregunta → ver traza en sidebar + streaming en main.

Si todo OK, detener con Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: UI Streamlit con sidebar de traza y streaming del informe"
```

---

## Task 18: Manejo de errores robusto (wrappers de los agentes)

**Files:**
- Modify: `src/agents/supervisor.py`
- Modify: `src/agents/analyst.py`
- Modify: `src/agents/writer.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_analyst.py`
- Modify: `tests/test_writer.py`

**Objetivo:** cubrir la regla común del spec §8: *"cada agente captura excepciones, las registra en `errors`, no propaga"*. Researcher ya cumple (vía `return_exceptions=True`). Falta envolver supervisor, analyst y writer.

> **Decisión de scope:** este task implementa el wrapper genérico que evita crashes. Los retries específicos del spec (re-prompt para `pydantic.ValidationError`, backoff exponencial para 429) quedan **fuera del MVP** — ver "Fuera de scope" al final del plan.

- [ ] **Step 1: Agregar test failing al supervisor**

Append a `tests/test_supervisor.py`:

```python
def test_supervisor_catches_llm_exception_and_returns_safe_diff():
    state = make_initial_state("¿Q?")

    def raise_boom(*args, **kwargs):
        raise RuntimeError("LLM boom")

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=raise_boom)
    llm.with_structured_output = MagicMock(return_value=structured)

    with patch("src.agents.supervisor.get_llm", return_value=llm):
        diff = supervisor_node(state)

    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "supervisor"
    assert diff["errors"][0]["exception_type"] == "RuntimeError"
    # Fallback seguro: si falla PLAN, fuerza writer con report degradado luego
    assert diff["next_agent"] == "writer"
    assert diff["enough_info"] is True
    assert any(e["level"] == "error" for e in diff["trace"])
```

- [ ] **Step 2: Verificar que falla**

Run: `pytest tests/test_supervisor.py::test_supervisor_catches_llm_exception_and_returns_safe_diff -v`
Expected: FAIL con `RuntimeError: LLM boom` (la excepción se propaga).

- [ ] **Step 3: Envolver supervisor en try/except**

Editar `src/agents/supervisor.py` — reemplazar la función `supervisor_node` por:

```python
def supervisor_node(state: AgentState) -> dict:
    """Nodo supervisor. Captura excepciones para no crashear el grafo."""
    try:
        if not state["queries"] and not state["research_results"]:
            return _run_plan(state)
        if state["analysis"] and state["enough_info"] is None:
            return _run_eval(state)
        return {
            "next_agent": "writer",
            "iteration_count": state["iteration_count"] + 1,
            "trace": [_trace("supervisor", "warn", "estado inesperado, forzando writer")],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "next_agent": "writer",
            "enough_info": True,
            "iteration_count": state["iteration_count"] + 1,
            "errors": [
                NodeError(
                    node="supervisor",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            ],
            "trace": [_trace("supervisor", "error", f"falló: {type(exc).__name__}: {exc}")],
        }
```

Y agregar import al inicio del archivo:
```python
from src.state import AgentState, NodeError, TraceEvent
```
(la línea `from src.state import AgentState, TraceEvent` se reemplaza por la nueva.)

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/test_supervisor.py -v`
Expected: 6 tests pass (5 anteriores + 1 nuevo).

- [ ] **Step 5: Agregar test failing al analyst**

Append a `tests/test_analyst.py`:

```python
def test_analyst_catches_llm_exception_and_returns_safe_diff():
    state = make_initial_state("¿Q?")
    state["research_results"] = [
        {"url": "u", "title": "t", "content": "c", "snippet": "s", "query": "q"}
    ]

    def raise_boom(prompt):
        raise RuntimeError("analyst boom")

    llm = MagicMock()
    structured = MagicMock()
    structured.invoke = MagicMock(side_effect=raise_boom)
    llm.with_structured_output = MagicMock(return_value=structured)

    with patch("src.agents.analyst.get_llm", return_value=llm):
        diff = analyst_node(state)

    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "analyst"
    # Fallback: analysis con disclaimer + gaps con la pregunta original para que supervisor decida
    assert "no fue posible" in diff["analysis"].lower() or "error" in diff["analysis"].lower()
    assert diff["gaps"] == [state["question"]]
```

- [ ] **Step 6: Verificar que falla**

Run: `pytest tests/test_analyst.py::test_analyst_catches_llm_exception_and_returns_safe_diff -v`
Expected: FAIL.

- [ ] **Step 7: Envolver analyst en try/except**

Editar `src/agents/analyst.py` — reemplazar `analyst_node` por:

```python
def analyst_node(state: AgentState) -> dict:
    try:
        llm = get_llm()
        structured = llm.with_structured_output(AnalystOutput)
        prompt = _PROMPT.format(
            question=state["question"],
            research_rounds=state["research_rounds"],
            results=_format_results(state["research_results"]),
        )
        result: AnalystOutput = structured.invoke(prompt)
        return {
            "analysis": result.analysis,
            "gaps": result.gaps,
            "iteration_count": state["iteration_count"] + 1,
            "trace": [
                _trace(
                    "info",
                    f"síntesis completa ({len(state['research_results'])} resultados, "
                    f"{len(result.gaps)} gaps)",
                )
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "analysis": "No fue posible completar la síntesis por un error interno.",
            "gaps": [state["question"]],
            "iteration_count": state["iteration_count"] + 1,
            "errors": [
                NodeError(
                    node="analyst",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            ],
            "trace": [_trace("error", f"falló: {type(exc).__name__}: {exc}")],
        }
```

Actualizar el import al inicio para incluir `NodeError`:
```python
from src.state import AgentState, NodeError, TraceEvent
```

- [ ] **Step 8: Verificar que pasa**

Run: `pytest tests/test_analyst.py -v`
Expected: 4 tests pass.

- [ ] **Step 9: Agregar test failing al writer**

Append a `tests/test_writer.py`:

```python
def test_writer_catches_llm_exception_and_returns_fallback_report():
    state = make_initial_state("¿Q?")
    state["analysis"] = "síntesis"
    state["research_results"] = []

    fake_llm = MagicMock()
    fake_llm.invoke = MagicMock(side_effect=RuntimeError("writer boom"))

    with patch("src.agents.writer.get_llm", return_value=fake_llm):
        diff = writer_node(state)

    assert diff["report"] != ""
    assert "no fue posible" in diff["report"].lower() or "error" in diff["report"].lower()
    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "writer"
```

- [ ] **Step 10: Verificar que falla**

Run: `pytest tests/test_writer.py::test_writer_catches_llm_exception_and_returns_fallback_report -v`
Expected: FAIL.

- [ ] **Step 11: Envolver writer en try/except**

Editar `src/agents/writer.py` — reemplazar `writer_node` por:

```python
_FALLBACK_REPORT = (
    "## Error\n\n"
    "No fue posible generar el informe completo por un error interno en el writer. "
    "Revisa la sección de errores para más detalles.\n"
)


def writer_node(state: AgentState) -> dict:
    """Genera el informe y le concatena la sección Fuentes."""
    try:
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
    except Exception as exc:  # noqa: BLE001
        return {
            "report": _FALLBACK_REPORT,
            "iteration_count": state["iteration_count"] + 1,
            "errors": [
                NodeError(
                    node="writer",
                    iteration=state["iteration_count"] + 1,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            ],
            "trace": [_trace("error", f"falló: {type(exc).__name__}: {exc}")],
        }
```

Actualizar import al inicio:
```python
from src.state import AgentState, NodeError, TraceEvent
```

- [ ] **Step 12: Verificar que pasa**

Run: `pytest tests/test_writer.py -v`
Expected: 6 tests pass.

- [ ] **Step 13: Correr la suite completa**

Run: `pytest tests/ -v`
Expected: todos los tests pasan.

- [ ] **Step 14: Commit**

```bash
git add src/agents/supervisor.py src/agents/analyst.py src/agents/writer.py \
        tests/test_supervisor.py tests/test_analyst.py tests/test_writer.py
git commit -m "feat: try/except genérico en supervisor/analyst/writer (no crashea, fallback degradado)"
```

---

## Task 19: README

**Files:**
- Modify: `README.md` (puede no existir; créalo)

- [ ] **Step 1: Crear/actualizar `README.md`**

```markdown
# Research Crew

Sistema multi-agente que recibe una pregunta de investigación y produce un informe ejecutivo con citas, coordinando supervisor + 3 workers (researcher, analyst, writer) vía LangGraph.

## Arquitectura

\`\`\`
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
\`\`\`

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

\`\`\`bash
python -m venv .venv
.venv\\Scripts\\activate           # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
cp .env.example .env              # editar con GOOGLE_API_KEY y TAVILY_API_KEY
\`\`\`

### CLI

\`\`\`bash
python -m src.graph "¿Cuál es el estado actual de la regulación de IA en Chile?"
\`\`\`

### UI

\`\`\`bash
streamlit run app.py
\`\`\`

Abre [localhost:8501](http://localhost:8501).

## Tests

\`\`\`bash
pytest tests/ -v
\`\`\`

Todos los tests usan mocks (no consumen API ni dependen de red). Tiempo objetivo: <5s.

## Decisiones de diseño

Ver [`docs/superpowers/specs/2026-05-24-research-crew-design.md`](docs/superpowers/specs/2026-05-24-research-crew-design.md) para el spec completo.

Highlights:
- **Supervisor con LLM solo en EVAL** — el routing trivial es determinístico (código), el LLM solo se usa donde aporta valor.
- **Aristas fijas + condicionales** — `researcher→analyst`, `analyst→supervisor`, `writer→END` son aristas fijas; las condicionales solo bajan del supervisor sobre `state["next_agent"]`.
- **Citas blindadas** — la sección Fuentes la construye Python, no el LLM, a partir de los `[n]` realmente citados. Garantiza cero alucinación de URLs.
- **Streaming token-por-token solo en Writer** — los otros nodos producen JSON estructurado pequeño; solo el reporte final justifica el efecto visual.

## Estructura del proyecto

\`\`\`
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
\`\`\`

## Licencia

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README con arquitectura, quickstart y decisiones de diseño"
```

---

## Task 20: Validación final

- [ ] **Step 1: Correr la suite completa de tests**

Run: `pytest tests/ -v`
Expected: ~24 tests, todos pasan, tiempo total <5s.

- [ ] **Step 2: Ruff check**

Run: `ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Verificar que el CLI arranca**

Si tienes `.env` con keys reales:
Run: `python -m src.graph "¿Qué es LangGraph?"`
Expected: se ven los prints del flujo y al final un report en Markdown.

Si no, omitir y dejar para validación manual posterior.

- [ ] **Step 4: Verificar que Streamlit arranca**

Si tienes `.env` con keys reales:
Run: `streamlit run app.py`
Expected: la app abre en browser, la pregunta se procesa, se ve la traza y el informe.

- [ ] **Step 5: Actualizar sección "Estado actual" en CLAUDE.md**

Marcar todos los checkboxes de las fases del Día 1 y Día 2 en `CLAUDE.md` como completados. Cambiar "Fase actual" a `🟢 Implementación completa`.

- [ ] **Step 6: Commit final**

```bash
git add CLAUDE.md
git commit -m "docs: marcar implementación completa en CLAUDE.md"
```

---

## Resumen de tareas

| # | Task | Archivos clave |
|---|---|---|
| 1 | Setup proyecto | pyproject, requirements, .env.example, __init__.py |
| 2 | Config | src/config.py |
| 3 | State | src/state.py |
| 4 | Schemas | src/schemas.py |
| 5 | LLM factory | src/llm.py |
| 6 | Prompt supervisor | src/prompts/supervisor.txt |
| 7 | Supervisor PLAN | src/agents/supervisor.py |
| 8 | Supervisor EVAL | (extiende test) |
| 9 | Researcher | src/agents/researcher.py |
| 10 | Prompt analyst | src/prompts/analyst.txt |
| 11 | Analyst | src/agents/analyst.py |
| 12 | Prompt writer | src/prompts/writer.txt |
| 13 | Writer | src/agents/writer.py |
| 14 | Graph builder + CLI | src/graph.py |
| 15 | Async bridge | src/ui/async_bridge.py |
| 16 | Trace renderer | src/ui/trace.py |
| 17 | Streamlit app | app.py |
| 18 | Manejo de errores robusto | supervisor.py, analyst.py, writer.py |
| 19 | README | README.md |
| 20 | Validación final | — |

**Total estimado:** ~27 tests automáticos, todos con mocks (corren en <5s sin consumir API). El sistema soporta flujo de ronda única, re-research hasta cap de 3 rondas, y degradación graceful ante fallos de LLM o Tavily.

---

## Fuera de scope (post-MVP)

Estas features están en el spec §8 pero NO se implementan en este plan. Decisión consciente para no inflar el MVP del demo. Si se necesitan después, son extensiones aisladas:

- **Retry de `pydantic.ValidationError`** — re-prompt al LLM cuando devuelve JSON no parseable. Hoy: el try/except genérico de Task 18 captura la excepción y devuelve un diff seguro (fallback). Mejora futura: 1 reintento explícito con un prompt "tu respuesta no fue JSON válido".
- **Backoff exponencial para 429** — 3 reintentos con 1s/2s/4s ante rate limit de Gemini. Hoy: el try/except genérico captura y degrada. Mejora futura: usar `tenacity` o lógica manual en `src/llm.py`.
- **Persistencia entre sesiones** — guardar historial de consultas previas. Streamlit pierde estado al recargar; podría usar SQLite o `st.session_state` persistente.
- **Selector dinámico de modelo en runtime** — `app.py` lo declara como dropdown pero hoy lee de env var. Cambiar a `st.selectbox` que propague a `get_llm(selected_model)`.
- **Detección automática de idioma** — hoy se confía en que el LLM detecte por el idioma de la pregunta. Mejora: `langdetect` + pass explícito al prompt.
