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
