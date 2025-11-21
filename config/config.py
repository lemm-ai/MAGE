"""Configuration management for MAGE.

This module provides a flexible configuration system using YAML files
with validation, defaults, and runtime updates.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, field, asdict

from mage.exceptions import ConfigurationError
from mage.utils import MAGELogger
from mage.platform import get_optimal_device

logger = MAGELogger.get_logger(__name__)


@dataclass
class AudioConfig:
    """Audio generation configuration."""
    
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2
    max_duration: float = 300.0  # Maximum 5 minutes
    default_duration: float = 30.0
    
    def validate(self) -> None:
        """Validate audio configuration parameters."""
        if self.sample_rate not in [22050, 44100, 48000, 96000]:
            raise ConfigurationError(
                f"Invalid sample rate: {self.sample_rate}",
                details={"valid_rates": [22050, 44100, 48000, 96000]}
            )
        
        if self.bit_depth not in [16, 24, 32]:
            raise ConfigurationError(
                f"Invalid bit depth: {self.bit_depth}",
                details={"valid_depths": [16, 24, 32]}
            )
        
        if self.channels not in [1, 2]:
            raise ConfigurationError(
                f"Invalid channel count: {self.channels}",
                details={"valid_channels": [1, 2]}
            )
        
        if self.max_duration <= 0 or self.max_duration > 3600:
            raise ConfigurationError(
                f"Max duration must be between 0 and 3600 seconds: {self.max_duration}"
            )


@dataclass
class ModelConfig:
    """Model configuration."""
    
    model_type: str = "default"
    model_path: Optional[str] = None
    cache_dir: str = "models"
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: str = "float32"  # float16, float32
    
    def validate(self) -> None:
        """Validate model configuration parameters."""
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ConfigurationError(
                f"Invalid device: {self.device}",
                details={"valid_devices": valid_devices}
            )
        
        valid_precisions = ["float16", "float32"]
        if self.precision not in valid_precisions:
            raise ConfigurationError(
                f"Invalid precision: {self.precision}",
                details={"valid_precisions": valid_precisions}
            )
    
    def get_device(self) -> str:
        """Get the resolved device string for PyTorch.
        
        If device is "auto", this will detect the optimal device.
        Otherwise, returns the configured device.
        
        Returns:
            Device string ("cuda", "mps", "cpu")
        """
        if self.device == "auto":
            try:
                resolved_device = get_optimal_device()
                logger.debug(f"Auto-detected device: {resolved_device}")
                return resolved_device
            except Exception as e:
                logger.warning(f"Failed to auto-detect device, falling back to CPU: {e}")
                return "cpu"
        else:
            return self.device


@dataclass
class LyricsConfig:
    """Lyrics generation configuration."""
    
    model_path: Optional[str] = None
    cache_dir: str = "models/lyricmind"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1
    
    def validate(self) -> None:
        """Validate lyrics configuration parameters."""
        if self.max_length < 1 or self.max_length > 2048:
            raise ConfigurationError(
                f"max_length must be between 1 and 2048: {self.max_length}"
            )
        
        if not 0.1 <= self.temperature <= 2.0:
            raise ConfigurationError(
                f"Temperature must be between 0.1 and 2.0: {self.temperature}"
            )
        
        if self.top_k < 1 or self.top_k > 1000:
            raise ConfigurationError(
                f"top_k must be between 1 and 1000: {self.top_k}"
            )
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ConfigurationError(
                f"top_p must be between 0.0 and 1.0: {self.top_p}"
            )
    
    def get_device(self) -> str:
        """Get the resolved device string for PyTorch.
        
        Returns:
            Device string ("cuda", "mps", "cpu")
        """
        if self.device == "auto":
            try:
                resolved_device = get_optimal_device()
                logger.debug(f"Auto-detected lyrics device: {resolved_device}")
                return resolved_device
            except Exception as e:
                logger.warning(f"Failed to auto-detect device, falling back to CPU: {e}")
                return "cpu"
        else:
            return self.device


@dataclass
class StemsConfig:
    """Stem separation configuration."""
    
    model_name: str = "htdemucs"
    cache_dir: str = "models/demucs"
    output_dir: str = "output/stems"
    device: str = "auto"
    use_cache: bool = True
    
    def validate(self) -> None:
        """Validate stems configuration parameters."""
        valid_models = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"]
        if self.model_name not in valid_models:
            logger.warning(f"Unknown model '{self.model_name}', using 'htdemucs'")
            self.model_name = "htdemucs"
        
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ConfigurationError(
                f"Invalid device: {self.device}",
                details={"valid_devices": valid_devices}
            )
    
    def get_device(self) -> str:
        """Get the resolved device string for PyTorch.
        
        Returns:
            Device string ("cuda", "mps", "cpu")
        """
        if self.device == "auto":
            try:
                resolved_device = get_optimal_device()
                logger.debug(f"Auto-detected stems device: {resolved_device}")
                return resolved_device
            except Exception as e:
                logger.warning(f"Failed to auto-detect device, falling back to CPU: {e}")
                return "cpu"
        else:
            return self.device


@dataclass
class GenerationConfig:
    """Audio generation parameters."""
    
    default_style: str = "ambient"
    default_tempo: int = 120
    default_key: str = "C_major"
    complexity: float = 0.5
    seed: Optional[int] = None
    
    def validate(self) -> None:
        """Validate generation configuration parameters."""
        if self.default_tempo < 20 or self.default_tempo > 300:
            raise ConfigurationError(
                f"Tempo must be between 20 and 300 BPM: {self.default_tempo}"
            )
        
        if not 0.0 <= self.complexity <= 1.0:
            raise ConfigurationError(
                f"Complexity must be between 0.0 and 1.0: {self.complexity}"
            )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    log_dir: str = "logs"
    console_output: bool = True
    file_output: bool = True
    
    def validate(self) -> None:
        """Validate logging configuration parameters."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.level}",
                details={"valid_levels": valid_levels}
            )


@dataclass
class TimelineConfig:
    """Timeline and arrangement configuration."""
    
    default_tempo: float = 120.0  # BPM
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4
    default_crossfade: float = 2.0  # seconds
    quantize_grid: int = 4  # quarter notes
    auto_normalize: bool = True
    prevent_clipping: bool = True
    
    # Section durations (in seconds)
    intro_duration: float = 8.0
    verse_duration: float = 16.0
    chorus_duration: float = 16.0
    bridge_duration: float = 12.0
    outro_duration: float = 8.0
    
    def validate(self) -> None:
        """Validate timeline configuration parameters."""
        if self.default_tempo <= 0 or self.default_tempo > 300:
            raise ConfigurationError(
                f"Invalid tempo: {self.default_tempo}",
                details={"valid_range": "1-300 BPM"}
            )
        
        if self.time_signature_numerator <= 0 or self.time_signature_numerator > 16:
            raise ConfigurationError(
                f"Invalid time signature numerator: {self.time_signature_numerator}",
                details={"valid_range": "1-16"}
            )
        
        if self.time_signature_denominator not in [2, 4, 8, 16]:
            raise ConfigurationError(
                f"Invalid time signature denominator: {self.time_signature_denominator}",
                details={"valid_values": [2, 4, 8, 16]}
            )
        
        if self.default_crossfade < 0 or self.default_crossfade > 10:
            raise ConfigurationError(
                f"Invalid crossfade duration: {self.default_crossfade}",
                details={"valid_range": "0-10 seconds"}
            )
        
        if self.quantize_grid not in [1, 2, 4, 8, 16, 32]:
            raise ConfigurationError(
                f"Invalid quantize grid: {self.quantize_grid}",
                details={"valid_values": [1, 2, 4, 8, 16, 32]}
            )


@dataclass
class EffectsConfig:
    """Audio effects processing configuration."""
    
    # EQ defaults
    eq_low_shelf_gain_db: float = 0.0
    eq_low_shelf_freq: float = 100.0
    eq_mid_gain_db: float = 0.0
    eq_mid_freq: float = 1000.0
    eq_mid_q: float = 1.0
    eq_high_shelf_gain_db: float = 0.0
    eq_high_shelf_freq: float = 8000.0
    
    # Compressor defaults
    comp_threshold_db: float = -20.0
    comp_ratio: float = 4.0
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 100.0
    
    # Reverb defaults
    reverb_room_size: float = 0.5
    reverb_damping: float = 0.5
    reverb_wet_level: float = 0.33
    reverb_dry_level: float = 0.4
    reverb_width: float = 1.0
    
    # Limiter defaults
    limiter_threshold_db: float = -1.0
    limiter_release_ms: float = 100.0
    
    # Enable/disable effects by default
    enable_eq: bool = False
    enable_compressor: bool = False
    enable_reverb: bool = False
    enable_limiter: bool = True
    
    def validate(self) -> None:
        """Validate effects configuration parameters."""
        # EQ validation
        if not (-24 <= self.eq_low_shelf_gain_db <= 24):
            raise ConfigurationError(
                f"EQ low shelf gain must be -24 to +24 dB: {self.eq_low_shelf_gain_db}"
            )
        
        if not (-24 <= self.eq_mid_gain_db <= 24):
            raise ConfigurationError(
                f"EQ mid gain must be -24 to +24 dB: {self.eq_mid_gain_db}"
            )
        
        if not (-24 <= self.eq_high_shelf_gain_db <= 24):
            raise ConfigurationError(
                f"EQ high shelf gain must be -24 to +24 dB: {self.eq_high_shelf_gain_db}"
            )
        
        # Compressor validation
        if not (-60 <= self.comp_threshold_db <= 0):
            raise ConfigurationError(
                f"Compressor threshold must be -60 to 0 dB: {self.comp_threshold_db}"
            )
        
        if not (1.0 <= self.comp_ratio <= 20.0):
            raise ConfigurationError(
                f"Compressor ratio must be 1.0 to 20.0: {self.comp_ratio}"
            )
        
        # Reverb validation
        if not (0.0 <= self.reverb_room_size <= 1.0):
            raise ConfigurationError(
                f"Reverb room size must be 0.0 to 1.0: {self.reverb_room_size}"
            )


@dataclass
class VocalConfig:
    """Vocal enhancement configuration."""
    model_path: Optional[str] = None
    cache_dir: str = "models/vocals"
    device: str = "auto"
    denoise_strength: float = 0.7
    brightness: float = 0.2
    warmth: float = 0.1
    clarity: float = 0.3
    target_level_db: float = -18.0
    compression_ratio: float = 3.0
    
    def validate(self) -> None:
        """Validate vocal enhancement configuration parameters."""
        if not (0.0 <= self.denoise_strength <= 1.0):
            raise ConfigurationError(
                f"Denoise strength must be 0.0 to 1.0: {self.denoise_strength}"
            )
        
        if not (-1.0 <= self.brightness <= 1.0):
            raise ConfigurationError(
                f"Brightness must be -1.0 to 1.0: {self.brightness}"
            )
        
        if not (-1.0 <= self.warmth <= 1.0):
            raise ConfigurationError(
                f"Warmth must be -1.0 to 1.0: {self.warmth}"
            )
        
        if not (-1.0 <= self.clarity <= 1.0):
            raise ConfigurationError(
                f"Clarity must be -1.0 to 1.0: {self.clarity}"
            )
        
        if not (-40.0 <= self.target_level_db <= 0.0):
            raise ConfigurationError(
                f"Target level must be -40.0 to 0.0 dB: {self.target_level_db}"
            )
        
        if not (1.0 <= self.compression_ratio <= 10.0):
            raise ConfigurationError(
                f"Compression ratio must be 1.0 to 10.0: {self.compression_ratio}"
            )


@dataclass
class Config:
    """Main configuration class for MAGE.
    
    This class manages all configuration settings for the MAGE system,
    providing validation, serialization, and easy access to settings.
    """
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lyrics: LyricsConfig = field(default_factory=LyricsConfig)
    stems: StemsConfig = field(default_factory=StemsConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    vocals: VocalConfig = field(default_factory=VocalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: str | Path) -> "Config":
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config instance with loaded settings
            
        Raises:
            ConfigurationError: If file cannot be loaded or is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                error_code="CONFIG_NOT_FOUND",
                details={"path": str(config_path)}
            )
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            
            return cls.from_dict(config_dict or {})
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                error_code="INVALID_YAML",
                details={"path": str(config_path), "error": str(e)}
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                error_code="CONFIG_LOAD_ERROR",
                details={"path": str(config_path), "error": str(e)}
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Config instance
        """
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        lyrics_config = LyricsConfig(**config_dict.get("lyrics", {}))
        stems_config = StemsConfig(**config_dict.get("stems", {}))
        timeline_config = TimelineConfig(**config_dict.get("timeline", {}))
        effects_config = EffectsConfig(**config_dict.get("effects", {}))
        vocals_config = VocalConfig(**config_dict.get("vocals", {}))
        generation_config = GenerationConfig(**config_dict.get("generation", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        config = cls(
            audio=audio_config,
            model=model_config,
            lyrics=lyrics_config,
            stems=stems_config,
            timeline=timeline_config,
            effects=effects_config,
            vocals=vocals_config,
            generation=generation_config,
            logging=logging_config
        )
        
        config.validate()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary containing all configuration values
        """
        return {
            "audio": asdict(self.audio),
            "model": asdict(self.model),
            "lyrics": asdict(self.lyrics),
            "stems": asdict(self.stems),
            "timeline": asdict(self.timeline),
            "effects": asdict(self.effects),
            "vocals": asdict(self.vocals),
            "generation": asdict(self.generation),
            "logging": asdict(self.logging)
        }
    
    def to_file(self, config_path: str | Path) -> None:
        """Save configuration to a YAML file.
        
        Args:
            config_path: Path where the configuration should be saved
            
        Raises:
            ConfigurationError: If file cannot be written
        """
        config_path = Path(config_path)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                error_code="CONFIG_SAVE_ERROR",
                details={"path": str(config_path), "error": str(e)}
            )
    
    def validate(self) -> None:
        """Validate all configuration sections.
        
        Raises:
            ConfigurationError: If any validation fails
        """
        try:
            self.audio.validate()
            self.model.validate()
            self.lyrics.validate()
            self.stems.validate()
            self.timeline.validate()
            self.effects.validate()
            self.vocals.validate()
            self.generation.validate()
            self.logging.validate()
            logger.debug("Configuration validation successful")
        except ConfigurationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary containing configuration updates
            
        Raises:
            ConfigurationError: If updates are invalid
        """
        if "audio" in updates:
            for key, value in updates["audio"].items():
                if hasattr(self.audio, key):
                    setattr(self.audio, key, value)
        
        if "model" in updates:
            for key, value in updates["model"].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if "generation" in updates:
            for key, value in updates["generation"].items():
                if hasattr(self.generation, key):
                    setattr(self.generation, key, value)
        
        if "logging" in updates:
            for key, value in updates["logging"].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        self.validate()
        logger.info("Configuration updated successfully")
