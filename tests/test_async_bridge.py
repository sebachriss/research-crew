from src.ui.async_bridge import run_graph_streaming


def test_run_graph_streaming_yields_state_and_token_events(monkeypatch):
    """Verifica que el bridge convierte graph.astream en un sync generator que yieldea tuples."""
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
