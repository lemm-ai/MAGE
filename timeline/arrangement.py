"""
Arrangement engine for automatic song composition.

This module provides the ArrangementEngine class for automatically
arranging tracks into a coherent song structure based on musical conventions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

from ..utils.logger import MAGELogger
from ..exceptions.exceptions import (
    MAGEException,
    InvalidParameterError
)
from .types import (
    Track,
    TrackType,
    ArrangementSection
)
from .timeline import Timeline


logger = MAGELogger.get_logger(__name__)


class ArrangementEngine:
    """
    Automatically arranges tracks into a song structure.
    
    The ArrangementEngine takes multiple tracks and creates an arrangement
    with standard song sections (intro, verse, chorus, bridge, outro) while
    managing track volumes and transitions.
    """
    
    # Default section durations (in seconds)
    DEFAULT_DURATIONS = {
        "intro": 8.0,
        "verse": 16.0,
        "prechorus": 8.0,
        "chorus": 16.0,
        "bridge": 12.0,
        "outro": 8.0,
        "breakdown": 8.0,
        "solo": 16.0
    }
    
    # Default track volume configurations for each section type
    DEFAULT_TRACK_VOLUMES = {
        "intro": {
            TrackType.VOCALS: 0.0,
            TrackType.INSTRUMENTAL: 0.7,
            TrackType.BASS: 0.5,
            TrackType.DRUMS: 0.6
        },
        "verse": {
            TrackType.VOCALS: 0.9,
            TrackType.INSTRUMENTAL: 0.6,
            TrackType.BASS: 0.7,
            TrackType.DRUMS: 0.7
        },
        "prechorus": {
            TrackType.VOCALS: 0.95,
            TrackType.INSTRUMENTAL: 0.7,
            TrackType.BASS: 0.8,
            TrackType.DRUMS: 0.8
        },
        "chorus": {
            TrackType.VOCALS: 1.0,
            TrackType.INSTRUMENTAL: 0.8,
            TrackType.BASS: 0.9,
            TrackType.DRUMS: 0.9
        },
        "bridge": {
            TrackType.VOCALS: 0.85,
            TrackType.INSTRUMENTAL: 0.75,
            TrackType.BASS: 0.6,
            TrackType.DRUMS: 0.7
        },
        "breakdown": {
            TrackType.VOCALS: 0.5,
            TrackType.INSTRUMENTAL: 0.4,
            TrackType.BASS: 0.3,
            TrackType.DRUMS: 0.2
        },
        "solo": {
            TrackType.VOCALS: 0.0,
            TrackType.INSTRUMENTAL: 1.0,
            TrackType.BASS: 0.8,
            TrackType.DRUMS: 0.85
        },
        "outro": {
            TrackType.VOCALS: 0.7,
            TrackType.INSTRUMENTAL: 0.6,
            TrackType.BASS: 0.5,
            TrackType.DRUMS: 0.4
        }
    }
    
    def __init__(
        self,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4)
    ):
        """
        Initialize the arrangement engine.
        
        Args:
            tempo: Tempo in beats per minute
            time_signature: Time signature as (beats_per_bar, note_value)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if tempo <= 0:
            raise InvalidParameterError(f"Tempo must be positive, got {tempo}")
        
        if time_signature[0] <= 0 or time_signature[1] <= 0:
            raise InvalidParameterError(f"Invalid time signature: {time_signature}")
        
        self.tempo = tempo
        self.time_signature = time_signature
        
        logger.info(
            f"Created ArrangementEngine at {tempo} BPM, "
            f"{time_signature[0]}/{time_signature[1]} time"
        )
    
    def get_bar_duration(self) -> float:
        """
        Calculate duration of one bar in seconds.
        
        Returns:
            Bar duration in seconds
        """
        beats_per_bar = self.time_signature[0]
        seconds_per_beat = 60.0 / self.tempo
        return beats_per_bar * seconds_per_beat
    
    def bars_to_seconds(self, num_bars: int) -> float:
        """
        Convert number of bars to seconds.
        
        Args:
            num_bars: Number of bars
            
        Returns:
            Duration in seconds
        """
        return num_bars * self.get_bar_duration()
    
    def seconds_to_bars(self, duration: float) -> int:
        """
        Convert seconds to number of bars (rounded).
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Number of bars (rounded)
        """
        return round(duration / self.get_bar_duration())
    
    def create_simple_structure(
        self,
        include_intro: bool = True,
        num_verses: int = 2,
        num_choruses: int = 3,
        include_bridge: bool = True,
        include_outro: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Create a simple song structure.
        
        Args:
            include_intro: Whether to include intro
            num_verses: Number of verses
            num_choruses: Number of choruses
            include_bridge: Whether to include bridge
            include_outro: Whether to include outro
            
        Returns:
            List of (section_name, duration) tuples
        """
        structure = []
        
        if include_intro:
            structure.append(("intro", self.DEFAULT_DURATIONS["intro"]))
        
        # Verse-chorus pattern
        for i in range(max(num_verses, num_choruses)):
            if i < num_verses:
                structure.append((f"verse{i+1}", self.DEFAULT_DURATIONS["verse"]))
            if i < num_choruses - 1:  # Save last chorus for end
                structure.append((f"chorus{i+1}", self.DEFAULT_DURATIONS["chorus"]))
        
        # Bridge before final chorus
        if include_bridge:
            structure.append(("bridge", self.DEFAULT_DURATIONS["bridge"]))
        
        # Final chorus
        structure.append((f"chorus{num_choruses}", self.DEFAULT_DURATIONS["chorus"]))
        
        if include_outro:
            structure.append(("outro", self.DEFAULT_DURATIONS["outro"]))
        
        logger.debug(f"Created simple structure with {len(structure)} sections")
        return structure
    
    def create_sections_from_structure(
        self,
        structure: List[Tuple[str, float]],
        custom_volumes: Optional[Dict[str, Dict[TrackType, float]]] = None
    ) -> List[ArrangementSection]:
        """
        Create arrangement sections from a song structure.
        
        Args:
            structure: List of (section_name, duration) tuples
            custom_volumes: Optional custom volume configurations
            
        Returns:
            List of ArrangementSection objects
        """
        sections = []
        current_time = 0.0
        
        for section_name, duration in structure:
            # Determine base section type (remove numbers)
            base_type = ''.join(c for c in section_name if not c.isdigit())
            
            # Get volume configuration
            if custom_volumes and section_name in custom_volumes:
                volumes = custom_volumes[section_name]
            elif base_type in self.DEFAULT_TRACK_VOLUMES:
                volumes = self.DEFAULT_TRACK_VOLUMES[base_type]
            else:
                # Default: all tracks at 0.8
                volumes = {
                    TrackType.VOCALS: 0.8,
                    TrackType.INSTRUMENTAL: 0.8,
                    TrackType.BASS: 0.8,
                    TrackType.DRUMS: 0.8
                }
            
            section = ArrangementSection(
                name=section_name,
                start_time=current_time,
                duration=duration,
                track_config=volumes.copy()
            )
            sections.append(section)
            current_time += duration
            
            logger.debug(f"Created section '{section_name}' at {section.start_time:.2f}s")
        
        return sections
    
    def arrange_tracks(
        self,
        tracks: List[Track],
        structure: Optional[List[Tuple[str, float]]] = None,
        timeline_name: str = "Arranged Song"
    ) -> Timeline:
        """
        Automatically arrange tracks into a timeline with song structure.
        
        Args:
            tracks: List of tracks to arrange
            structure: Optional custom structure (if None, uses default)
            timeline_name: Name for the created timeline
            
        Returns:
            Timeline with arranged tracks
            
        Raises:
            ValidationError: If tracks are invalid
        """
        try:
            logger.info(f"Arranging {len(tracks)} tracks...")
            
            if not tracks:
                raise InvalidParameterError("No tracks provided for arrangement")
            
            # Validate all tracks have the same sample rate
            sample_rates = set(t.sample_rate for t in tracks)
            if len(sample_rates) > 1:
                logger.warning(f"Multiple sample rates detected: {sample_rates}")
                # Use the most common sample rate
                sample_rate = max(sample_rates, key=lambda sr: sum(1 for t in tracks if t.sample_rate == sr))
            else:
                sample_rate = tracks[0].sample_rate
            
            # Create timeline
            timeline = Timeline(sample_rate=sample_rate, name=timeline_name)
            
            # Create structure if not provided
            if structure is None:
                structure = self.create_simple_structure()
            
            # Create sections
            sections = self.create_sections_from_structure(structure)
            for section in sections:
                timeline.add_section(section)
            
            # Add all tracks to timeline
            for track in tracks:
                timeline.add_track(track, validate=False)
            
            logger.info(
                f"Created arrangement '{timeline_name}' with "
                f"{len(sections)} sections, {len(tracks)} tracks, "
                f"duration {timeline.get_duration():.2f}s"
            )
            
            return timeline
            
        except Exception as e:
            logger.error(f"Arrangement failed: {e}")
            raise InvalidParameterError(f"Failed to arrange tracks: {e}")
    
    def create_crossfade_transition(
        self,
        track1: Track,
        track2: Track,
        crossfade_duration: float = 2.0
    ) -> List[Track]:
        """
        Create a crossfade transition between two tracks.
        
        Args:
            track1: First track
            track2: Second track
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            List of modified tracks with crossfade applied
        """
        try:
            logger.debug(
                f"Creating {crossfade_duration}s crossfade between "
                f"'{track1.name}' and '{track2.name}'"
            )
            
            # Create copies to avoid modifying originals
            t1 = Track(
                name=track1.name,
                track_type=track1.track_type,
                audio_data=track1.audio_data.copy(),
                sample_rate=track1.sample_rate,
                start_time=track1.start_time,
                duration=track1.duration,
                volume=track1.volume,
                pan=track1.pan,
                mute=track1.mute,
                solo=track1.solo
            )
            
            t2 = Track(
                name=track2.name,
                track_type=track2.track_type,
                audio_data=track2.audio_data.copy(),
                sample_rate=track2.sample_rate,
                start_time=track1.end_time - crossfade_duration,  # Overlap
                duration=track2.duration,
                volume=track2.volume,
                pan=track2.pan,
                mute=track2.mute,
                solo=track2.solo
            )
            
            # Apply fades
            t1.fade_out = crossfade_duration
            t2.fade_in = crossfade_duration
            
            return [t1, t2]
            
        except Exception as e:
            logger.error(f"Crossfade creation failed: {e}")
            raise InvalidParameterError(f"Failed to create crossfade: {e}")
    
    def quantize_to_grid(
        self,
        time: float,
        grid_division: int = 4
    ) -> float:
        """
        Quantize a time value to the nearest beat grid.
        
        Args:
            time: Time in seconds
            grid_division: Grid division (4 = quarter notes, 8 = eighth notes, etc.)
            
        Returns:
            Quantized time in seconds
        """
        beat_duration = 60.0 / self.tempo
        grid_duration = beat_duration / (grid_division / 4)
        
        num_grids = round(time / grid_duration)
        quantized = num_grids * grid_duration
        
        logger.debug(f"Quantized {time:.3f}s to {quantized:.3f}s (grid={grid_division})")
        return quantized
    
    def get_info(self) -> Dict:
        """
        Get arrangement engine information.
        
        Returns:
            Dictionary with engine info
        """
        return {
            "tempo": self.tempo,
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
            "bar_duration": self.get_bar_duration(),
            "beat_duration": 60.0 / self.tempo
        }
