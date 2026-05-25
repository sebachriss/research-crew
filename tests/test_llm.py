from unittest.mock import patch

from src.llm import get_llm


def test_get_llm_uses_default_model_when_no_arg(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm()
        assert mock_class.called
        kwargs = mock_class.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash"


def test_get_llm_uses_provided_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm("gemini-2.5-pro")
        kwargs = mock_class.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-pro"


def test_get_llm_sets_temperature_zero(monkeypatch):
    """Para decisiones del supervisor y síntesis del analyst queremos determinismo."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.ChatGoogleGenerativeAI") as mock_class:
        get_llm()
        kwargs = mock_class.call_args.kwargs
        assert kwargs["temperature"] == 0.0
