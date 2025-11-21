"""Stem separation module for MAGE.

This module provides audio stem separation using Demucs, allowing extraction
of vocals, bass, drums, and other instruments from mixed audio.
"""

from mage.stems.separator import DemucsSeparator, StemManager
from mage.stems.types import StemType, SeparatedStems

__all__ = ["DemucsSeparator", "StemManager", "StemType", "SeparatedStems"]
