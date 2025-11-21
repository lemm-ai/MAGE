"""Types and enums for stem separation."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import numpy as np


class StemType(Enum):
    """Types of audio stems that can be separated."""
    
    VOCALS = "vocals"
    BASS = "bass"
    DRUMS = "drums"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"
    
    def __str__(self):
        return self.value


@dataclass
class SeparatedStems:
    """Container for separated audio stems.
    
    Attributes:
        vocals: Vocal track audio data
        bass: Bass track audio data
        drums: Drum track audio data
        other: Other instruments audio data
        sample_rate: Sample rate of audio
        source_path: Path to original source file
        metadata: Additional metadata
    """
    
    vocals: Optional[np.ndarray] = None
    bass: Optional[np.ndarray] = None
    drums: Optional[np.ndarray] = None
    other: Optional[np.ndarray] = None
    sample_rate: int = 44100
    source_path: Optional[Path] = None
    metadata: Optional[Dict] = None
    
    def get_stem(self, stem_type: StemType) -> Optional[np.ndarray]:
        """Get a specific stem by type.
        
        Args:
            stem_type: Type of stem to retrieve
            
        Returns:
            Audio data for the stem, or None if not available
        """
        stem_map = {
            StemType.VOCALS: self.vocals,
            StemType.BASS: self.bass,
            StemType.DRUMS: self.drums,
            StemType.OTHER: self.other,
        }
        return stem_map.get(stem_type)
    
    def set_stem(self, stem_type: StemType, audio: np.ndarray) -> None:
        """Set audio data for a specific stem.
        
        Args:
            stem_type: Type of stem to set
            audio: Audio data
        """
        if stem_type == StemType.VOCALS:
            self.vocals = audio
        elif stem_type == StemType.BASS:
            self.bass = audio
        elif stem_type == StemType.DRUMS:
            self.drums = audio
        elif stem_type == StemType.OTHER:
            self.other = audio
    
    def available_stems(self) -> list[StemType]:
        """Get list of available stems.
        
        Returns:
            List of StemType for available stems
        """
        available = []
        if self.vocals is not None:
            available.append(StemType.VOCALS)
        if self.bass is not None:
            available.append(StemType.BASS)
        if self.drums is not None:
            available.append(StemType.DRUMS)
        if self.other is not None:
            available.append(StemType.OTHER)
        return available
    
    def mix_stems(self, stem_types: Optional[list[StemType]] = None) -> np.ndarray:
        """Mix multiple stems together.
        
        Args:
            stem_types: List of stems to mix (None = mix all available)
            
        Returns:
            Mixed audio data
        """
        if stem_types is None:
            stem_types = self.available_stems()
        
        mixed = None
        for stem_type in stem_types:
            stem_audio = self.get_stem(stem_type)
            if stem_audio is not None:
                if mixed is None:
                    mixed = stem_audio.copy()
                else:
                    # Ensure same shape
                    if mixed.shape != stem_audio.shape:
                        min_len = min(mixed.shape[-1], stem_audio.shape[-1])
                        mixed = mixed[..., :min_len]
                        stem_audio = stem_audio[..., :min_len]
                    mixed += stem_audio
        
        return mixed if mixed is not None else np.array([])
