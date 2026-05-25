# Seguimiento — Research Crew

**Última sesión:** 2026-05-25 (segunda parte: smoke tests CLI + UI, fixes derivados)
**Próxima sesión:** push a GitHub público

---

## ✅ Estado actual

Implementación **completa + minor issues cerrados + smoke tests validados** en sesión 2026-05-25.

- 44 tests automatizados pasando, todos con mocks (corren en <2s)
- Ruff clean en todo `src/` y `tests/`
- Arquitectura fiel al spec: supervisor + 3 workers vía LangGraph, prompts en `.txt`, error handling graceful
- Streaming real validado en navegador con Gemini + Tavily reales: traza streamea, tokens fluyen, botón Descargar persistente
- `_now/_trace` centralizados en `src/agents/_trace_util.py` (Literal tipado, sin `type: ignore`)
- `LICENSE` MIT presente; `.env.example` completo; `app.py` usa `MAX_ITERATIONS` del config
- Regex de citas acepta agrupadas `[1, 2, 3]` (fix derivado del smoke test CLI)
- UI Streamlit con sidebar live via placeholders + render persistente del report

Para verificar que todo sigue OK al retomar:

```powershell
cd C:\Users\Seba\Documents\github
.venv\Scripts\Activate.ps1
pytest tests/ -v       # esperado: 41 passed
ruff check .           # esperado: All checks passed!
git log --oneline | head -10
```

---

## ✅ Pendientes cerrados (2026-05-25)

### 1-3. Configurar .env + smoke tests CLI/UI

CLI: validado con "¿Cuál es el estado actual de la regulación de IA en Chile?" — produjo informe completo con TL;DR, Hallazgos, Fuentes reales (uhc.cl, iapp.org). Bug detectado y fixeado: citas agrupadas (commit `a314388`).

UI: validada en navegador con la misma pregunta — sidebar streamea evento por evento, tokens del report fluyen, botón Descargar persistente. Bugs detectados y fixeados (commit `27012c0`): sidebar via placeholders, eliminado `st.rerun()`, render persistente del informe.

### 4. Push a GitHub público

Repo público: **https://github.com/sebalda/research-crew** (branch `main`, 32 commits).

Nota: `gh` requirió `auth login` para agregar la cuenta `sebalda` (la activa antes era `sebakobai1` de trabajo) y `gh auth setup-git` para que git resolviera el credential helper correcto en HTTPS — el primer push devolvió 403 porque Git Credential Manager tenía cacheada la otra cuenta.

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
