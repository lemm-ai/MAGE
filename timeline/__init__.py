"""
Timeline module for multi-track audio composition and arrangement.

This module provides classes for:
- Track: Individual audio tracks with timing and effects
- Timeline: Multi-track container with mixing and rendering
- ArrangementEngine: Automatic song structure creation
"""

from .types import (
    TrackType,
    FadeType,
    Track,
    ArrangementSection,
    TimelineMarker
)
from .timeline import Timeline
from .arrangement import ArrangementEngine

__all__ = [
    "TrackType",
    "FadeType",
    "Track",
    "ArrangementSection",
    "TimelineMarker",
    "Timeline",
    "ArrangementEngine"
]
