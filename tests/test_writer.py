import asyncio
from unittest.mock import MagicMock, patch

from src.agents.writer import build_sources_section, extract_cited_indices, writer_node
from src.state import make_initial_state


def test_extract_cited_indices_finds_all_unique_in_order():
    text = "blah [1] blah [3] more [1] then [2] end"
    assert extract_cited_indices(text) == [1, 3, 2]


def test_extract_cited_indices_empty_when_no_citations():
    assert extract_cited_indices("texto sin citas") == []


def test_extract_cited_indices_expands_grouped_citations():
    """Gemini suele consolidar múltiples citas en un solo bracket:
    [9, 21, 26, 32]. Cada índice debe aparecer en el resultado."""
    text = "primero [1, 2, 3] luego [4,5] y solo [6]"
    assert extract_cited_indices(text) == [1, 2, 3, 4, 5, 6]


def test_extract_cited_indices_dedupes_across_groups():
    """Índices repetidos entre grupos y citas sueltas se cuentan una vez."""
    text = "blah [1, 2] y [2, 3] y luego [1]"
    assert extract_cited_indices(text) == [1, 2, 3]


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

    async def fake_astream(prompt):
        # Emit in 2 chunks to verify accumulation works
        yield MagicMock(content=fake_text[:30])
        yield MagicMock(content=fake_text[30:])

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream

    with patch("src.agents.writer.get_llm", return_value=fake_llm):
        diff = asyncio.run(writer_node(state))

    assert "## TL;DR" in diff["report"]
    assert "## Fuentes" in diff["report"]
    assert "[1] tA" in diff["report"]
    assert "[2] tB" in diff["report"]
    assert "next_agent" not in diff
    assert diff["iteration_count"] == 1


def test_writer_catches_llm_exception_and_returns_fallback_report():
    state = make_initial_state("¿Q?")
    state["analysis"] = "síntesis"
    state["research_results"] = []

    fake_llm = MagicMock()
    fake_llm.astream = MagicMock(side_effect=RuntimeError("writer boom"))

    with patch("src.agents.writer.get_llm", return_value=fake_llm):
        diff = asyncio.run(writer_node(state))

    assert diff["report"] != ""
    assert "no fue posible" in diff["report"].lower() or "error" in diff["report"].lower()
    assert len(diff["errors"]) == 1
    assert diff["errors"][0]["node"] == "writer"
