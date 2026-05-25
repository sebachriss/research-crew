"""Bridge async→sync para Streamlit.

Streamlit no soporta `async for` directamente en el main thread. Este módulo
expone un generador sync que internamente corre `graph.astream` en un thread
de background y bombea eventos a una `queue.Queue` sync. El generator drena
esa queue desde el main thread, yieldando eventos a medida que llegan — esto
es lo que da la sensación de streaming en tiempo real en la UI.
"""
import asyncio
import queue
import threading
from collections.abc import Generator

from src.graph import build_graph
from src.state import make_initial_state


def run_graph_streaming(
    question: str,
) -> Generator[tuple[str, object], None, None]:
    """Generador sync que yieldea eventos del grafo en tiempo real.

    Yields:
        ("state", payload_dict)  — un update con el diff de algún nodo.
        ("token", str)            — un token de streaming del writer.
    """
    graph = build_graph()
    initial = make_initial_state(question)

    q: queue.Queue = queue.Queue()
    DONE = object()

    def thread_target():
        async def producer():
            try:
                async for event_type, payload in graph.astream(
                    initial, stream_mode=["updates", "messages"]
                ):
                    if event_type == "updates":
                        q.put(("state", payload))
                    elif event_type == "messages":
                        token, metadata = payload
                        if metadata.get("langgraph_node") == "writer":
                            q.put(("token", token.content))
            finally:
                q.put(DONE)

        asyncio.run(producer())

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is DONE:
            break
        yield item

    thread.join()
