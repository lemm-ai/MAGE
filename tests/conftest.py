"""Pytest configuration and shared fixtures."""

import pytest
import logging
from pathlib import Path


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path_factory.mktemp("test_outputs")
    return output_dir


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@pytest.fixture
def sample_config_dict():
    """Provide a sample configuration dictionary."""
    return {
        "audio": {
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 2,
            "max_duration": 300.0,
            "default_duration": 30.0
        },
        "model": {
            "model_type": "default",
            "device": "cpu",
            "precision": "float32"
        },
        "generation": {
            "default_style": "ambient",
            "default_tempo": 120,
            "complexity": 0.5
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs"
        }
    }
