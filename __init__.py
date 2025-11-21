"""MAGE - Mixed Audio Generation Engine.

A comprehensive AI-based music generation system with robust error handling,
logging, and modular architecture.
"""

__version__ = "0.1.0"
__author__ = "MAGE Development Team"

from mage.core.engine import MAGE
from mage.config.config import Config
from mage.exceptions.exceptions import (
    MAGEException,
    AudioGenerationError,
    ModelLoadError,
    ConfigurationError,
)

__all__ = [
    "MAGE",
    "Config",
    "MAGEException",
    "AudioGenerationError",
    "ModelLoadError",
    "ConfigurationError",
]
