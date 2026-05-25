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
