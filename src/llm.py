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
