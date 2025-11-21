"""
Timeline management for multi-track audio composition.

This module provides the Timeline class for managing multiple audio tracks,
applying effects, and rendering the final mixed audio output.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import soundfile as sf

from ..utils.logger import MAGELogger
from ..exceptions.exceptions import (
    MAGEException,
    AudioProcessingError,
    InvalidParameterError
)
from .types import (
    Track,
    TrackType,
    FadeType,
    ArrangementSection,
    TimelineMarker
)


logger = MAGELogger.get_logger(__name__)


class Timeline:
    """
    Multi-track audio timeline for composition and arrangement.
    
    The Timeline manages multiple audio tracks, handles synchronization,
    applies effects, and renders the final mixed output.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        name: str = "Untitled Timeline"
    ):
        """
        Initialize a new timeline.
        
        Args:
            sample_rate: Sample rate for the timeline in Hz
            name: Human-readable name for the timeline
            
        Raises:
            ValidationError: If sample rate is invalid
        """
        if sample_rate <= 0:
            raise InvalidParameterError(f"Sample rate must be positive, got {sample_rate}")
        
        self.sample_rate = sample_rate
        self.name = name
        self.tracks: List[Track] = []
        self.sections: List[ArrangementSection] = []
        self.markers: List[TimelineMarker] = []
        
        logger.info(f"Created timeline '{name}' at {sample_rate} Hz")
    
    def add_track(
        self,
        track: Track,
        validate: bool = True
    ) -> None:
        """
        Add a track to the timeline.
        
        Args:
            track: Track to add
            validate: Whether to validate track compatibility
            
        Raises:
            ValidationError: If track validation fails
        """
        try:
            if validate:
                if track.sample_rate != self.sample_rate:
                    logger.warning(
                        f"Track '{track.name}' sample rate ({track.sample_rate} Hz) "
                        f"differs from timeline ({self.sample_rate} Hz). "
                        f"Resampling may be needed."
                    )
            
            self.tracks.append(track)
            logger.info(
                f"Added track '{track.name}' ({track.track_type.value}) "
                f"at {track.start_time:.2f}s, duration {track.get_audio_duration():.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to add track '{track.name}': {e}")
            raise InvalidParameterError(f"Track validation failed: {e}")
    
    def remove_track(self, track_name: str) -> bool:
        """
        Remove a track from the timeline by name.
        
        Args:
            track_name: Name of the track to remove
            
        Returns:
            True if track was removed, False if not found
        """
        for i, track in enumerate(self.tracks):
            if track.name == track_name:
                removed = self.tracks.pop(i)
                logger.info(f"Removed track '{removed.name}'")
                return True
        
        logger.warning(f"Track '{track_name}' not found")
        return False
    
    def get_track(self, track_name: str) -> Optional[Track]:
        """
        Get a track by name.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Track if found, None otherwise
        """
        for track in self.tracks:
            if track.name == track_name:
                return track
        return None
    
    def get_tracks_by_type(self, track_type: TrackType) -> List[Track]:
        """
        Get all tracks of a specific type.
        
        Args:
            track_type: Type of tracks to retrieve
            
        Returns:
            List of matching tracks
        """
        return [t for t in self.tracks if t.track_type == track_type]
    
    def add_section(self, section: ArrangementSection) -> None:
        """
        Add an arrangement section to the timeline.
        
        Args:
            section: Section to add
        """
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.start_time)
        logger.info(
            f"Added section '{section.name}' at {section.start_time:.2f}s, "
            f"duration {section.duration:.2f}s"
        )
    
    def add_marker(self, marker: TimelineMarker) -> None:
        """
        Add a marker to the timeline.
        
        Args:
            marker: Marker to add
        """
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.time)
        logger.info(f"Added marker '{marker.name}' at {marker.time:.2f}s")
    
    def get_duration(self) -> float:
        """
        Get the total duration of the timeline.
        
        Returns:
            Duration in seconds
        """
        if not self.tracks:
            return 0.0
        
        return max(track.end_time for track in self.tracks)
    
    def _apply_fade(
        self,
        audio: np.ndarray,
        fade_duration: float,
        fade_type: FadeType,
        is_fade_in: bool
    ) -> np.ndarray:
        """
        Apply fade to audio data.
        
        Args:
            audio: Audio array (channels, samples) or (samples,)
            fade_duration: Fade duration in seconds
            fade_type: Type of fade curve
            is_fade_in: True for fade-in, False for fade-out
            
        Returns:
            Audio with fade applied
        """
        if fade_duration <= 0:
            return audio
        
        fade_samples = int(fade_duration * self.sample_rate)
        if fade_samples == 0:
            return audio
        
        # Get audio shape
        is_stereo = audio.ndim == 2
        num_samples = audio.shape[-1]
        fade_samples = min(fade_samples, num_samples)
        
        # Generate fade curve
        x = np.linspace(0, 1, fade_samples)
        
        if fade_type == FadeType.LINEAR:
            curve = x
        elif fade_type == FadeType.EXPONENTIAL:
            curve = x ** 2
        elif fade_type == FadeType.LOGARITHMIC:
            curve = np.sqrt(x)
        elif fade_type == FadeType.S_CURVE:
            curve = (np.sin((x - 0.5) * np.pi) + 1) / 2
        else:
            curve = x  # Default to linear
        
        # Invert for fade-out
        if not is_fade_in:
            curve = curve[::-1]
        
        # Apply fade
        result = audio.copy()
        if is_stereo:
            if is_fade_in:
                result[:, :fade_samples] *= curve
            else:
                result[:, -fade_samples:] *= curve
        else:
            if is_fade_in:
                result[:fade_samples] *= curve
            else:
                result[-fade_samples:] *= curve
        
        return result
    
    def _apply_pan(self, audio: np.ndarray, pan: float) -> np.ndarray:
        """
        Apply stereo panning to audio.
        
        Args:
            audio: Audio array (mono or stereo)
            pan: Pan value (-1.0 to 1.0)
            
        Returns:
            Stereo audio with panning applied
        """
        if pan == 0.0:
            # No panning needed
            if audio.ndim == 1:
                return np.stack([audio, audio])
            return audio
        
        # Convert to stereo if mono
        if audio.ndim == 1:
            stereo = np.stack([audio, audio])
        else:
            stereo = audio.copy()
        
        # Apply constant power panning
        left_gain = np.cos((pan + 1) * np.pi / 4)
        right_gain = np.sin((pan + 1) * np.pi / 4)
        
        stereo[0] *= left_gain
        stereo[1] *= right_gain
        
        return stereo
    
    def _process_track_audio(self, track: Track) -> np.ndarray:
        """
        Process a track's audio with all effects applied.
        
        Args:
            track: Track to process
            
        Returns:
            Processed stereo audio (2, samples)
        """
        audio = track.audio_data.copy()
        
        # Apply volume
        if track.volume != 1.0:
            audio = audio * track.volume
        
        # Apply fades
        if track.fade_in > 0:
            audio = self._apply_fade(audio, track.fade_in, track.fade_type, True)
        
        if track.fade_out > 0:
            audio = self._apply_fade(audio, track.fade_out, track.fade_type, False)
        
        # Apply panning (converts to stereo)
        audio = self._apply_pan(audio, track.pan)
        
        return audio
    
    def render(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        apply_sections: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Render the timeline to mixed audio.
        
        Args:
            start_time: Start rendering from this time (seconds)
            end_time: End rendering at this time (None = timeline end)
            apply_sections: Whether to apply section configurations
            
        Returns:
            Tuple of (stereo_audio, sample_rate)
            
        Raises:
            AudioProcessingError: If rendering fails
        """
        try:
            logger.info(f"Rendering timeline '{self.name}'...")
            
            # Determine render duration
            timeline_duration = self.get_duration()
            if end_time is None:
                end_time = timeline_duration
            
            render_duration = end_time - start_time
            if render_duration <= 0:
                raise InvalidParameterError("Render duration must be positive")
            
            num_samples = int(render_duration * self.sample_rate)
            mixed_audio = np.zeros((2, num_samples), dtype=np.float32)
            
            # Check for solo tracks
            has_solo = any(t.solo and not t.mute for t in self.tracks)
            
            # Process each track
            for track in self.tracks:
                # Skip muted tracks
                if track.mute:
                    logger.debug(f"Skipping muted track '{track.name}'")
                    continue
                
                # Skip non-solo tracks if any track is soloed
                if has_solo and not track.solo:
                    logger.debug(f"Skipping non-solo track '{track.name}' (solo mode)")
                    continue
                
                # Check if track overlaps render window
                if track.end_time <= start_time or track.start_time >= end_time:
                    logger.debug(f"Track '{track.name}' outside render window")
                    continue
                
                # Process track audio
                processed_audio = self._process_track_audio(track)
                
                # Calculate track position in render buffer
                track_start = max(0, track.start_time - start_time)
                track_end = min(render_duration, track.end_time - start_time)
                
                # Calculate sample positions
                start_sample = int(track_start * self.sample_rate)
                end_sample = int(track_end * self.sample_rate)
                
                # Calculate source audio range
                source_start = max(0, int((start_time - track.start_time) * self.sample_rate))
                source_end = source_start + (end_sample - start_sample)
                
                # Mix into output buffer
                if source_end <= processed_audio.shape[1]:
                    mixed_audio[:, start_sample:end_sample] += processed_audio[:, source_start:source_end]
                else:
                    # Handle case where we need less samples than available
                    available = processed_audio.shape[1] - source_start
                    mixed_audio[:, start_sample:start_sample + available] += processed_audio[:, source_start:]
                
                logger.debug(
                    f"Mixed track '{track.name}' from {track_start:.2f}s to {track_end:.2f}s"
                )
            
            # Apply section configurations if requested
            if apply_sections and self.sections:
                logger.debug("Applying section configurations...")
                for section in self.sections:
                    # Check if section overlaps render window
                    if section.end_time <= start_time or section.start_time >= end_time:
                        continue
                    
                    section_start = max(0, section.start_time - start_time)
                    section_end = min(render_duration, section.end_time - start_time)
                    
                    start_sample = int(section_start * self.sample_rate)
                    end_sample = int(section_end * self.sample_rate)
                    
                    # Apply track volumes from section config
                    # Note: This is a simplified implementation
                    # In a full version, you'd need to track which audio came from which track
                    logger.debug(
                        f"Section '{section.name}' at {section_start:.2f}s - {section_end:.2f}s"
                    )
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                logger.warning(f"Audio clipping detected (max={max_val:.2f}), normalizing...")
                mixed_audio = mixed_audio / max_val
            
            logger.info(
                f"Rendered {render_duration:.2f}s of audio "
                f"({len(self.tracks)} tracks, {num_samples} samples)"
            )
            
            return mixed_audio, self.sample_rate
            
        except Exception as e:
            logger.error(f"Timeline rendering failed: {e}")
            raise AudioProcessingError(f"Failed to render timeline: {e}")
    
    def export(
        self,
        output_path: Path,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        apply_sections: bool = True,
        format: str = "wav"
    ) -> None:
        """
        Render and export timeline to audio file.
        
        Args:
            output_path: Path to save exported audio
            start_time: Start export from this time (seconds)
            end_time: End export at this time (None = timeline end)
            apply_sections: Whether to apply section configurations
            format: Audio format (wav, flac, ogg, mp3)
            
        Raises:
            AudioProcessingError: If export fails
        """
        try:
            logger.info(f"Exporting timeline to {output_path}...")
            
            # Render audio
            audio, sample_rate = self.render(start_time, end_time, apply_sections)
            
            # Transpose for soundfile (samples, channels)
            audio = audio.T
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export
            sf.write(str(output_path), audio, sample_rate, format=format)
            
            logger.info(f"Successfully exported timeline to {output_path}")
            
        except Exception as e:
            logger.error(f"Timeline export failed: {e}")
            raise AudioProcessingError(f"Failed to export timeline: {e}")
    
    def get_info(self) -> Dict:
        """
        Get timeline information.
        
        Returns:
            Dictionary with timeline info
        """
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "duration": self.get_duration(),
            "num_tracks": len(self.tracks),
            "num_sections": len(self.sections),
            "num_markers": len(self.markers),
            "tracks": [
                {
                    "name": t.name,
                    "type": t.track_type.value,
                    "start": t.start_time,
                    "end": t.end_time,
                    "duration": t.get_audio_duration(),
                    "mute": t.mute,
                    "solo": t.solo
                }
                for t in self.tracks
            ]
        }
