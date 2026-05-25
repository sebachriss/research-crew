# tests/test_config.py
import importlib
import os
from unittest.mock import patch


def test_config_loads_defaults():
    from src import config
    assert config.MAX_ITERATIONS == 8
    assert config.MAX_RESEARCH_ROUNDS == 3
    assert config.QUERIES_PER_ROUND == 3
    assert config.TAVILY_MAX_RESULTS == 5
    assert config.MODEL_NAME_DEFAULT == "gemini-2.5-flash"


def test_config_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_ITERATIONS", "12")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-pro")
    # Re-import to pick up env vars
    from src import config
    importlib.reload(config)
    assert config.MAX_ITERATIONS == 12
    assert config.MODEL_NAME_DEFAULT == "gemini-2.5-pro"


def test_config_requires_api_keys_via_check():
    from src.config import ConfigError, check_api_keys
    with patch.dict(os.environ, {}, clear=True):
        try:
            check_api_keys()
            raise AssertionError("should have raised")
        except ConfigError as e:
            assert "GOOGLE_API_KEY" in str(e) or "TAVILY_API_KEY" in str(e)
