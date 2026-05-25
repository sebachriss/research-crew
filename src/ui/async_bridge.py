"""Bridge async→sync para Streamlit.

Streamlit no soporta `async for` directamente en el main thread. Este módulo expone
un generador sync que internamente corre `graph.astream` en un loop async y va
yieldando eventos. Estrategia: bombearlos a una asyncio.Queue, y consumirla con
asyncio.run mediante una corutina recolectora — simple y suficiente para demo.
"""
import asyncio
from collections.abc import Generator

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
    yield from items
