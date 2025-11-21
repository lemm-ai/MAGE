"""
Timeline type definitions for MAGE.

This module provides type definitions and data structures for multi-track audio
composition, including tracks, timelines, and arrangement configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np


class TrackType(Enum):
    """Types of audio tracks in a timeline."""
    VOCALS = "vocals"
    INSTRUMENTAL = "instrumental"
    BASS = "bass"
    DRUMS = "drums"
    OTHER = "other"
    MASTER = "master"


class FadeType(Enum):
    """Types of fade curves."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    S_CURVE = "s_curve"


@dataclass
class Track:
    """
    Represents a single audio track in a timeline.
    
    Attributes:
        name: Human-readable name for the track
        track_type: Type of audio content (vocals, bass, drums, etc.)
        audio_data: Audio samples as numpy array (channels, samples)
        sample_rate: Sample rate of the audio in Hz
        start_time: Start position in seconds from timeline beginning
        duration: Duration of the track in seconds (None = use full audio)
        volume: Volume multiplier (0.0 to 1.0)
        pan: Stereo pan (-1.0 = left, 0.0 = center, 1.0 = right)
        fade_in: Fade-in duration in seconds
        fade_out: Fade-out duration in seconds
        fade_type: Type of fade curve to apply
        mute: Whether the track is muted
        solo: Whether the track is soloed (only solo tracks play)
        metadata: Additional metadata for the track
    """
    name: str
    track_type: TrackType
    audio_data: np.ndarray
    sample_rate: int
    start_time: float = 0.0
    duration: Optional[float] = None
    volume: float = 1.0
    pan: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    fade_type: FadeType = FadeType.LINEAR
    mute: bool = False
    solo: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate track parameters."""
        if self.volume < 0.0 or self.volume > 1.0:
            raise ValueError(f"Volume must be between 0.0 and 1.0, got {self.volume}")
        
        if self.pan < -1.0 or self.pan > 1.0:
            raise ValueError(f"Pan must be between -1.0 and 1.0, got {self.pan}")
        
        if self.start_time < 0.0:
            raise ValueError(f"Start time must be non-negative, got {self.start_time}")
        
        if self.fade_in < 0.0:
            raise ValueError(f"Fade-in duration must be non-negative, got {self.fade_in}")
        
        if self.fade_out < 0.0:
            raise ValueError(f"Fade-out duration must be non-negative, got {self.fade_out}")
        
        if self.audio_data.ndim not in [1, 2]:
            raise ValueError(f"Audio data must be 1D (mono) or 2D (stereo), got {self.audio_data.ndim}D")
    
    @property
    def end_time(self) -> float:
        """Calculate the end time of the track."""
        track_duration = self.duration if self.duration is not None else self.get_audio_duration()
        return self.start_time + track_duration
    
    def get_audio_duration(self) -> float:
        """Get the duration of the audio data in seconds."""
        num_samples = self.audio_data.shape[-1]  # Last dimension is always samples
        return num_samples / self.sample_rate
    
    @property
    def is_stereo(self) -> bool:
        """Check if the track is stereo."""
        return self.audio_data.ndim == 2 and self.audio_data.shape[0] == 2
    
    @property
    def is_mono(self) -> bool:
        """Check if the track is mono."""
        return self.audio_data.ndim == 1 or (self.audio_data.ndim == 2 and self.audio_data.shape[0] == 1)


@dataclass
class ArrangementSection:
    """
    Represents a section in a song arrangement.
    
    Attributes:
        name: Name of the section (e.g., "intro", "verse", "chorus")
        start_time: Start time in seconds
        duration: Duration in seconds
        track_config: Configuration for which tracks to play and at what volume
        tempo_multiplier: Tempo adjustment for this section (1.0 = normal)
        metadata: Additional section metadata
    """
    name: str
    start_time: float
    duration: float
    track_config: Dict[TrackType, float] = field(default_factory=dict)  # track_type -> volume
    tempo_multiplier: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate section parameters."""
        if self.start_time < 0.0:
            raise ValueError(f"Start time must be non-negative, got {self.start_time}")
        
        if self.duration <= 0.0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        
        if self.tempo_multiplier <= 0.0:
            raise ValueError(f"Tempo multiplier must be positive, got {self.tempo_multiplier}")
        
        for volume in self.track_config.values():
            if volume < 0.0 or volume > 1.0:
                raise ValueError(f"Track volume must be between 0.0 and 1.0, got {volume}")
    
    @property
    def end_time(self) -> float:
        """Calculate the end time of the section."""
        return self.start_time + self.duration


@dataclass
class TimelineMarker:
    """
    Represents a marker/cue point in the timeline.
    
    Attributes:
        name: Marker name
        time: Position in seconds
        color: Optional color for UI display
        metadata: Additional marker metadata
    """
    name: str
    time: float
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate marker parameters."""
        if self.time < 0.0:
            raise ValueError(f"Marker time must be non-negative, got {self.time}")
