# Seguimiento — Research Crew

**Última sesión:** 2026-05-25
**Próxima sesión:** smoke tests con `.env` + push a GitHub

---

## ✅ Estado actual

Implementación **completa + minor issues cerrados** en sesión 2026-05-25.

- 42 tests automatizados pasando, todos con mocks (corren en <2s) — se agregó el e2e degradado
- Ruff clean en todo `src/` y `tests/`
- Arquitectura fiel al spec: supervisor + 3 workers vía LangGraph, prompts en `.txt`, error handling graceful
- Streaming real funcionando: bridge async→sync incremental + writer con `astream`
- `_now/_trace` centralizados en `src/agents/_trace_util.py` (Literal tipado, sin `type: ignore`)
- `LICENSE` MIT presente; `.env.example` completo; `app.py` usa `MAX_ITERATIONS` del config

Para verificar que todo sigue OK al retomar:

```powershell
cd C:\Users\Seba\Documents\github
.venv\Scripts\Activate.ps1
pytest tests/ -v       # esperado: 41 passed
ruff check .           # esperado: All checks passed!
git log --oneline | head -10
```

---

## 🔧 Pendiente para mañana

### 1. Configurar `.env` con API keys reales

Editar `.env` (todavía no existe; partir de `.env.example`):

```powershell
cp .env.example .env
notepad .env
```

Llenar:
- `GOOGLE_API_KEY=` → tu key de [Google AI Studio](https://aistudio.google.com/apikey)
- `TAVILY_API_KEY=` → tu key de [Tavily](https://app.tavily.com/home)

### 2. Smoke test del CLI

```powershell
.venv\Scripts\python.exe -m src.graph "¿Cuál es el estado actual de la regulación de IA en Chile?"
```

**Qué validar:**
- Termina sin crash
- Imprime un informe Markdown con `## TL;DR`, `## Hallazgos clave`, `## Fuentes`
- Las URLs de la sección Fuentes son reales (no inventadas por el LLM)
- `iteration_count` final ≤ 8

### 3. Smoke test de la UI Streamlit

```powershell
streamlit run app.py
```

Abre [localhost:8501](http://localhost:8501) y prueba:

- [ ] Sidebar muestra la traza de eventos **en tiempo real** mientras el grafo corre (no todo al final — el fix del bridge debería hacer esto). Si la traza aparece de golpe al final, hay un bug latente.
- [ ] Main panel muestra tokens del informe haciéndose stream token-por-token. Si aparece todo de golpe, los callbacks de Gemini no están emitiendo tokens vía `stream_mode="messages"` y hay que ajustar.
- [ ] Botón "Descargar informe (.md)" funciona.
- [ ] Si hay errores en algún nodo, se ven en el expander del sidebar.

### 4. Push a GitHub público (si los smoke tests pasan)

```powershell
gh repo create research-crew --public --source=. --remote=origin --description "Sistema multi-agente con LangGraph"
git push -u origin main
```

(`gh` está configurado con la cuenta personal `sebaceronu@gmail.com`.)

---

## ⚠ Issues conocidos no resueltos (Minor)

Los 6 issues del code review final están **resueltos** (commits `86c57a3`, `ae6a7aa`, `a06c659` del 2026-05-25):

1. ~~`_now()` / `_trace()` duplicados~~ → centralizados en `src/agents/_trace_util.py`.
2. ~~`.env.example` incompleto~~ → agregados `MAX_RESEARCH_ROUNDS`, `QUERIES_PER_ROUND`, `TAVILY_MAX_RESULTS`.
3. ~~`app.py:44` hardcodea `/8`~~ → ahora usa `MAX_ITERATIONS` del config.
4. ~~No hay `LICENSE`~~ → `LICENSE` MIT presente en raíz.
5. ~~`type: ignore[arg-type]` en `TraceEvent`~~ → `trace(level: TraceLevel)` con `Literal` tipado.
6. ~~Falta test e2e degradado~~ → `test_graph_completes_when_all_tavily_queries_fail` en `tests/test_graph.py`.

---

## 📁 Archivos importantes

| Archivo | Para qué |
|---|---|
| `CLAUDE.md` | Contexto del proyecto, leído al inicio de cada sesión |
| `docs/superpowers/specs/2026-05-24-research-crew-design.md` | Spec técnico aprobado |
| `docs/superpowers/plans/2026-05-24-research-crew.md` | Plan de implementación (20 tasks) |
| `docs/seguimiento.md` | **Este archivo** |

## 🔑 Decisiones de diseño clave (recordatorio)

- **Supervisor con LLM solo en EVAL** — el routing trivial es código determinístico.
- **Aristas fijas + condicionales** — `researcher→analyst`, `analyst→supervisor`, `writer→END` son fijas; solo el supervisor emite `next_agent` para las condicionales.
- **Citas blindadas** — la sección Fuentes la construye Python (no el LLM) a partir de los `[n]` que efectivamente aparecen en el texto.
- **`if state["analysis"]:`** en el supervisor (NO `and state["enough_info"] is None`) — fix crítico de Task 14 para permitir re-research.

---

## 🚀 Cuando retomes

Una vez completados los pasos 1-4 arriba, el proyecto está listo como portfolio piece. Próximas mejoras opcionales:

- Detección automática de idioma de la pregunta
- Persistencia de queries previas (historial en `st.session_state`)
- Selector dinámico de modelo (`gemini-2.5-flash` vs `gemini-2.5-pro`) en la sidebar
- Retry de `pydantic.ValidationError` con re-prompt
- Backoff exponencial para 429 de Gemini

Todas listadas en la sección "Fuera de scope (post-MVP)" del plan.
