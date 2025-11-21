"""Core audio generation engine for MAGE.

This module contains the main MAGE class that orchestrates audio generation,
processing, and export functionality.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import soundfile as sf

from mage.config import Config
from mage.exceptions import (
    AudioGenerationError,
    InvalidParameterError,
    ExportError,
)
from mage.utils import MAGELogger, log_function_call, log_performance
from mage.models.generator import AudioGenerator
from mage.processors.audio_processor import AudioProcessor

logger = MAGELogger.get_logger(__name__)


class GeneratedAudio:
    """Container for generated audio data with processing and export capabilities.
    
    Attributes:
        data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        metadata: Additional metadata about the generation
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize generated audio container.
        
        Args:
            data: Audio data as numpy array (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dictionary
        """
        self.data = data
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        
        # Calculate duration
        if data.ndim == 1:
            self.duration = len(data) / sample_rate
        else:
            self.duration = data.shape[1] / sample_rate
        
        logger.debug(
            f"Created GeneratedAudio: {self.duration:.2f}s @ {sample_rate}Hz",
            extra={"duration": self.duration, "sample_rate": sample_rate}
        )
    
    @log_function_call(logger)
    def apply_effects(
        self,
        reverb: float = 0.0,
        compression: float = 0.0,
        eq: Optional[Dict[str, float]] = None
    ) -> "GeneratedAudio":
        """Apply audio effects to the generated audio.
        
        Args:
            reverb: Reverb amount (0.0 to 1.0)
            compression: Compression amount (0.0 to 1.0)
            eq: Equalization parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            InvalidParameterError: If parameters are out of range
        """
        if not 0.0 <= reverb <= 1.0:
            raise InvalidParameterError(
                f"Reverb must be between 0.0 and 1.0, got {reverb}"
            )
        
        if not 0.0 <= compression <= 1.0:
            raise InvalidParameterError(
                f"Compression must be between 0.0 and 1.0, got {compression}"
            )
        
        processor = AudioProcessor()
        
        if reverb > 0.0:
            self.data = processor.apply_reverb(self.data, reverb, self.sample_rate)
            logger.info(f"Applied reverb: {reverb}")
        
        if compression > 0.0:
            self.data = processor.apply_compression(self.data, compression)
            logger.info(f"Applied compression: {compression}")
        
        if eq:
            self.data = processor.apply_eq(self.data, eq, self.sample_rate)
            logger.info(f"Applied EQ: {eq}")
        
        return self
    
    @log_function_call(logger)
    def normalize(self, target_level: float = -3.0) -> "GeneratedAudio":
        """Normalize audio to target level.
        
        Args:
            target_level: Target level in dB
            
        Returns:
            Self for method chaining
        """
        processor = AudioProcessor()
        self.data = processor.normalize(self.data, target_level)
        logger.info(f"Normalized to {target_level} dB")
        return self
    
    @log_performance(logger)
    def export(
        self,
        output_path: str | Path,
        format: Optional[str] = None,
        bitrate: Optional[str] = None
    ) -> None:
        """Export audio to file.
        
        Args:
            output_path: Path where audio should be saved
            format: Audio format (inferred from extension if not provided)
            bitrate: Bitrate for compressed formats (e.g., "320k")
            
        Raises:
            ExportError: If export fails
        """
        output_path = Path(output_path)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure data is in correct format
            audio_data = self.data
            if audio_data.ndim == 1:
                # Mono to stereo if needed
                audio_data = np.stack([audio_data, audio_data])
            
            # Transpose to (samples, channels) for soundfile
            if audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            
            # Write audio file
            sf.write(
                str(output_path),
                audio_data,
                self.sample_rate,
                format=format
            )
            
            logger.info(
                f"Exported audio to {output_path}",
                extra={
                    "path": str(output_path),
                    "duration": self.duration,
                    "sample_rate": self.sample_rate
                }
            )
            
        except Exception as e:
            raise ExportError(
                f"Failed to export audio to {output_path}: {e}",
                details={
                    "path": str(output_path),
                    "error": str(e),
                    "format": format
                }
            )


class MAGE:
    """Main Mixed Audio Generation Engine class.
    
    This class provides the primary interface for generating audio using
    AI models with comprehensive error handling and logging.
    
    Attributes:
        config: Configuration object
        generator: Audio generator instance
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the MAGE engine.
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or Config()
        
        # Configure logging
        MAGELogger.get_logger(
            __name__,
            log_level=self.config.logging.level,
            log_dir=Path(self.config.logging.log_dir)
        )
        
        logger.info("Initializing MAGE engine")
        logger.debug(f"Configuration: {self.config.to_dict()}")
        
        # Initialize components
        try:
            self.generator = AudioGenerator(self.config)
            logger.info("MAGE engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MAGE engine: {e}")
            raise
    
    @log_performance(logger)
    def generate(
        self,
        duration: Optional[float] = None,
        style: Optional[str] = None,
        tempo: Optional[int] = None,
        key: Optional[str] = None,
        complexity: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> GeneratedAudio:
        """Generate audio with specified parameters.
        
        Args:
            duration: Duration in seconds (uses config default if not provided)
            style: Music style/genre
            tempo: Tempo in BPM
            key: Musical key
            complexity: Generation complexity (0.0 to 1.0)
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            GeneratedAudio object containing the generated audio
            
        Raises:
            AudioGenerationError: If generation fails
            InvalidParameterError: If parameters are invalid
        """
        # Use defaults from config if not provided
        duration = duration or self.config.audio.default_duration
        style = style or self.config.generation.default_style
        tempo = tempo or self.config.generation.default_tempo
        key = key or self.config.generation.default_key
        complexity = complexity or self.config.generation.complexity
        seed = seed or self.config.generation.seed
        
        # Validate parameters
        if duration <= 0 or duration > self.config.audio.max_duration:
            raise InvalidParameterError(
                f"Duration must be between 0 and {self.config.audio.max_duration}",
                details={"duration": duration}
            )
        
        if tempo < 20 or tempo > 300:
            raise InvalidParameterError(
                f"Tempo must be between 20 and 300 BPM",
                details={"tempo": tempo}
            )
        
        if not 0.0 <= complexity <= 1.0:
            raise InvalidParameterError(
                f"Complexity must be between 0.0 and 1.0",
                details={"complexity": complexity}
            )
        
        logger.info(
            f"Generating audio: {duration}s, {style}, {tempo} BPM, {key}",
            extra={
                "duration": duration,
                "style": style,
                "tempo": tempo,
                "key": key,
                "complexity": complexity
            }
        )
        
        try:
            # Generate audio using the generator
            audio_data = self.generator.generate(
                duration=duration,
                style=style,
                tempo=tempo,
                key=key,
                complexity=complexity,
                seed=seed,
                **kwargs
            )
            
            # Create GeneratedAudio object with metadata
            metadata = {
                "style": style,
                "tempo": tempo,
                "key": key,
                "complexity": complexity,
                "seed": seed,
                **kwargs
            }
            
            generated = GeneratedAudio(
                data=audio_data,
                sample_rate=self.config.audio.sample_rate,
                metadata=metadata
            )
            
            logger.info("Audio generation completed successfully")
            return generated
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}", exc_info=True)
            raise AudioGenerationError(
                f"Failed to generate audio: {e}",
                details={
                    "duration": duration,
                    "style": style,
                    "tempo": tempo,
                    "error": str(e)
                }
            )
    
    def get_available_styles(self) -> list[str]:
        """Get list of available music styles.
        
        Returns:
            List of style names
        """
        return self.generator.get_available_styles()
    
    def get_available_keys(self) -> list[str]:
        """Get list of available musical keys.
        
        Returns:
            List of key names
        """
        return self.generator.get_available_keys()
