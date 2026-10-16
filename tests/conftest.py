"""Shared test setup: build a complete pipeline in a temporary folder.

The tests never read or write the real data/ and outputs/ folders, and they
never call the Groq API.
"""

import pytest

from src import config, data_processing, generate_data, mock_sap_api, risk_engine


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "RAW_DIR", root / "raw")
        patch.setattr(config, "PROCESSED_DIR", root / "processed")
        patch.setattr(config, "OUTPUT_DIR", root / "outputs")
        patch.setattr(config, "AUDIT_LOG_PATH", root / "outputs" / "audit_log.jsonl")

        generate_data.generate_all(raw_dir=config.RAW_DIR)
        outputs, summary = data_processing.run(config.RAW_DIR, config.PROCESSED_DIR)
        mock_sap_api.reload_data()
        scored = risk_engine.score_open_orders(outputs["open_orders"])

        yield {"root": root, "outputs": outputs, "summary": summary, "scored": scored}
    mock_sap_api.reload_data()


@pytest.fixture
def no_api_key(monkeypatch):
    """Make sure no test can reach the real LLM."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.load_env_file", lambda path=None: None)
