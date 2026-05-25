import pytest
from pydantic import ValidationError

from src.schemas import AnalystOutput, EvalOutput, PlanOutput


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
