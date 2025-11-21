"""
Professional audio effects using Pedalboard.

This module provides high-quality audio effects processing using Spotify's
Pedalboard library, including EQ, compression, reverb, chorus, delay, and more.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

from ..utils.logger import MAGELogger
from ..exceptions.exceptions import (
    AudioProcessingError,
    InvalidParameterError
)

logger = MAGELogger.get_logger(__name__)

# Try to import pedalboard
try:
    from pedalboard import (
        Pedalboard,
        Reverb,
        Compressor,
        Gain,
        Chorus,
        Delay,
        Distortion,
        HighpassFilter,
        LowpassFilter,
        PeakFilter,
        LowShelfFilter,
        HighShelfFilter,
        Limiter,
        Phaser,
        NoiseGate
    )
    PEDALBOARD_AVAILABLE = True
    logger.debug("Pedalboard library loaded successfully")
except ImportError as e:
    PEDALBOARD_AVAILABLE = False
    logger.warning(f"Pedalboard not available: {e}")
    # Create dummy classes for type hints
    Pedalboard = None
    Reverb = Compressor = Gain = Chorus = Delay = None
    Distortion = HighpassFilter = LowpassFilter = PeakFilter = None
    LowShelfFilter = HighShelfFilter = Limiter = Phaser = NoiseGate = None


class EffectsProcessor:
    """
    Professional audio effects processor using Pedalboard.
    
    This class provides high-quality audio effects including:
    - Parametric EQ with multiple bands
    - Dynamic range compression
    - Reverb (room, hall, plate)
    - Chorus, delay, phaser
    - Limiting and noise gating
    - Distortion and saturation
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the effects processor.
        
        Args:
            sample_rate: Sample rate in Hz
            
        Raises:
            AudioProcessingError: If Pedalboard is not available
        """
        if not PEDALBOARD_AVAILABLE:
            raise AudioProcessingError(
                "Pedalboard library not available. Install with: pip install pedalboard",
                error_code="PEDALBOARD_NOT_FOUND"
            )
        
        self.sample_rate = sample_rate
        self.board = Pedalboard([])
        
        logger.info(f"Initialized EffectsProcessor at {sample_rate} Hz")
    
    def apply_eq(
        self,
        audio: np.ndarray,
        low_shelf_gain_db: float = 0.0,
        low_shelf_freq: float = 100.0,
        mid_gain_db: float = 0.0,
        mid_freq: float = 1000.0,
        mid_q: float = 1.0,
        high_shelf_gain_db: float = 0.0,
        high_shelf_freq: float = 8000.0
    ) -> np.ndarray:
        """
        Apply 3-band parametric EQ.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            low_shelf_gain_db: Low shelf gain in dB (-24 to +24)
            low_shelf_freq: Low shelf frequency in Hz
            mid_gain_db: Mid peak gain in dB (-24 to +24)
            mid_freq: Mid peak frequency in Hz
            mid_q: Mid peak Q factor (0.1 to 10.0)
            high_shelf_gain_db: High shelf gain in dB (-24 to +24)
            high_shelf_freq: High shelf frequency in Hz
            
        Returns:
            Processed audio with same shape as input
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying EQ: low={low_shelf_gain_db}dB@{low_shelf_freq}Hz, "
                f"mid={mid_gain_db}dB@{mid_freq}Hz, "
                f"high={high_shelf_gain_db}dB@{high_shelf_freq}Hz"
            )
            
            # Validate parameters
            if not (-24 <= low_shelf_gain_db <= 24):
                raise InvalidParameterError(f"Low shelf gain must be -24 to +24 dB, got {low_shelf_gain_db}")
            
            if not (-24 <= mid_gain_db <= 24):
                raise InvalidParameterError(f"Mid gain must be -24 to +24 dB, got {mid_gain_db}")
            
            if not (-24 <= high_shelf_gain_db <= 24):
                raise InvalidParameterError(f"High shelf gain must be -24 to +24 dB, got {high_shelf_gain_db}")
            
            # Create EQ chain
            effects = []
            
            if low_shelf_gain_db != 0.0:
                effects.append(
                    LowShelfFilter(
                        cutoff_frequency_hz=low_shelf_freq,
                        gain_db=low_shelf_gain_db,
                        q=0.707
                    )
                )
            
            if mid_gain_db != 0.0:
                effects.append(
                    PeakFilter(
                        cutoff_frequency_hz=mid_freq,
                        gain_db=mid_gain_db,
                        q=mid_q
                    )
                )
            
            if high_shelf_gain_db != 0.0:
                effects.append(
                    HighShelfFilter(
                        cutoff_frequency_hz=high_shelf_freq,
                        gain_db=high_shelf_gain_db,
                        q=0.707
                    )
                )
            
            if not effects:
                logger.debug("No EQ adjustments needed, returning original audio")
                return audio
            
            # Apply effects
            board = Pedalboard(effects)
            processed = board(audio, self.sample_rate)
            
            logger.debug(f"EQ applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"EQ processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply EQ: {e}",
                error_code="EQ_ERROR",
                details={
                    "low_shelf": f"{low_shelf_gain_db}dB @ {low_shelf_freq}Hz",
                    "mid": f"{mid_gain_db}dB @ {mid_freq}Hz Q={mid_q}",
                    "high_shelf": f"{high_shelf_gain_db}dB @ {high_shelf_freq}Hz",
                    "error": str(e)
                }
            )
    
    def apply_compressor(
        self,
        audio: np.ndarray,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 100.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            threshold_db: Threshold in dB (-60 to 0)
            ratio: Compression ratio (1.0 to 20.0)
            attack_ms: Attack time in milliseconds (0.1 to 100)
            release_ms: Release time in milliseconds (10 to 1000)
            
        Returns:
            Compressed audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying compressor: threshold={threshold_db}dB, "
                f"ratio={ratio}:1, attack={attack_ms}ms, release={release_ms}ms"
            )
            
            # Validate parameters
            if not (-60 <= threshold_db <= 0):
                raise InvalidParameterError(f"Threshold must be -60 to 0 dB, got {threshold_db}")
            
            if not (1.0 <= ratio <= 20.0):
                raise InvalidParameterError(f"Ratio must be 1.0 to 20.0, got {ratio}")
            
            if not (0.1 <= attack_ms <= 100):
                raise InvalidParameterError(f"Attack must be 0.1 to 100 ms, got {attack_ms}")
            
            if not (10 <= release_ms <= 1000):
                raise InvalidParameterError(f"Release must be 10 to 1000 ms, got {release_ms}")
            
            # Create compressor
            compressor = Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms
            )
            
            board = Pedalboard([compressor])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Compression applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply compression: {e}",
                error_code="COMPRESSION_ERROR",
                details={
                    "threshold_db": threshold_db,
                    "ratio": ratio,
                    "attack_ms": attack_ms,
                    "release_ms": release_ms,
                    "error": str(e)
                }
            )
    
    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.33,
        dry_level: float = 0.4,
        width: float = 1.0
    ) -> np.ndarray:
        """
        Apply reverb effect.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            room_size: Room size (0.0 to 1.0)
            damping: High frequency damping (0.0 to 1.0)
            wet_level: Wet signal level (0.0 to 1.0)
            dry_level: Dry signal level (0.0 to 1.0)
            width: Stereo width (0.0 to 1.0)
            
        Returns:
            Reverberated audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying reverb: room_size={room_size}, damping={damping}, "
                f"wet={wet_level}, dry={dry_level}, width={width}"
            )
            
            # Validate parameters
            if not (0.0 <= room_size <= 1.0):
                raise InvalidParameterError(f"Room size must be 0.0 to 1.0, got {room_size}")
            
            if not (0.0 <= damping <= 1.0):
                raise InvalidParameterError(f"Damping must be 0.0 to 1.0, got {damping}")
            
            if not (0.0 <= wet_level <= 1.0):
                raise InvalidParameterError(f"Wet level must be 0.0 to 1.0, got {wet_level}")
            
            if not (0.0 <= dry_level <= 1.0):
                raise InvalidParameterError(f"Dry level must be 0.0 to 1.0, got {dry_level}")
            
            # Create reverb
            reverb = Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
                width=width
            )
            
            board = Pedalboard([reverb])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Reverb applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Reverb processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply reverb: {e}",
                error_code="REVERB_ERROR",
                details={
                    "room_size": room_size,
                    "damping": damping,
                    "wet_level": wet_level,
                    "dry_level": dry_level,
                    "width": width,
                    "error": str(e)
                }
            )
    
    def apply_chorus(
        self,
        audio: np.ndarray,
        rate_hz: float = 1.0,
        depth: float = 0.25,
        centre_delay_ms: float = 7.0,
        feedback: float = 0.0,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Apply chorus effect.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            rate_hz: LFO rate in Hz (0.1 to 10)
            depth: Modulation depth (0.0 to 1.0)
            centre_delay_ms: Center delay in milliseconds (1 to 20)
            feedback: Feedback amount (0.0 to 1.0)
            mix: Dry/wet mix (0.0 to 1.0)
            
        Returns:
            Chorused audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying chorus: rate={rate_hz}Hz, depth={depth}, "
                f"delay={centre_delay_ms}ms, feedback={feedback}, mix={mix}"
            )
            
            # Validate parameters
            if not (0.1 <= rate_hz <= 10):
                raise InvalidParameterError(f"Rate must be 0.1 to 10 Hz, got {rate_hz}")
            
            if not (0.0 <= depth <= 1.0):
                raise InvalidParameterError(f"Depth must be 0.0 to 1.0, got {depth}")
            
            if not (1.0 <= centre_delay_ms <= 20.0):
                raise InvalidParameterError(f"Centre delay must be 1 to 20 ms, got {centre_delay_ms}")
            
            # Create chorus
            chorus = Chorus(
                rate_hz=rate_hz,
                depth=depth,
                centre_delay_ms=centre_delay_ms,
                feedback=feedback,
                mix=mix
            )
            
            board = Pedalboard([chorus])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Chorus applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Chorus processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply chorus: {e}",
                error_code="CHORUS_ERROR",
                details={
                    "rate_hz": rate_hz,
                    "depth": depth,
                    "centre_delay_ms": centre_delay_ms,
                    "feedback": feedback,
                    "mix": mix,
                    "error": str(e)
                }
            )
    
    def apply_delay(
        self,
        audio: np.ndarray,
        delay_seconds: float = 0.5,
        feedback: float = 0.3,
        mix: float = 0.5
    ) -> np.ndarray:
        """
        Apply delay effect.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            delay_seconds: Delay time in seconds (0.001 to 2.0)
            feedback: Feedback amount (0.0 to 0.95)
            mix: Dry/wet mix (0.0 to 1.0)
            
        Returns:
            Delayed audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying delay: time={delay_seconds}s, "
                f"feedback={feedback}, mix={mix}"
            )
            
            # Validate parameters
            if not (0.001 <= delay_seconds <= 2.0):
                raise InvalidParameterError(f"Delay must be 0.001 to 2.0 s, got {delay_seconds}")
            
            if not (0.0 <= feedback <= 0.95):
                raise InvalidParameterError(f"Feedback must be 0.0 to 0.95, got {feedback}")
            
            if not (0.0 <= mix <= 1.0):
                raise InvalidParameterError(f"Mix must be 0.0 to 1.0, got {mix}")
            
            # Create delay
            delay = Delay(
                delay_seconds=delay_seconds,
                feedback=feedback,
                mix=mix
            )
            
            board = Pedalboard([delay])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Delay applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Delay processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply delay: {e}",
                error_code="DELAY_ERROR",
                details={
                    "delay_seconds": delay_seconds,
                    "feedback": feedback,
                    "mix": mix,
                    "error": str(e)
                }
            )
    
    def apply_limiter(
        self,
        audio: np.ndarray,
        threshold_db: float = -1.0,
        release_ms: float = 100.0
    ) -> np.ndarray:
        """
        Apply limiter to prevent clipping.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            threshold_db: Limiting threshold in dB (-20 to 0)
            release_ms: Release time in milliseconds (10 to 1000)
            
        Returns:
            Limited audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Applying limiter: threshold={threshold_db}dB, release={release_ms}ms")
            
            # Validate parameters
            if not (-20 <= threshold_db <= 0):
                raise InvalidParameterError(f"Threshold must be -20 to 0 dB, got {threshold_db}")
            
            if not (10 <= release_ms <= 1000):
                raise InvalidParameterError(f"Release must be 10 to 1000 ms, got {release_ms}")
            
            # Create limiter
            limiter = Limiter(
                threshold_db=threshold_db,
                release_ms=release_ms
            )
            
            board = Pedalboard([limiter])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Limiter applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Limiter processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply limiter: {e}",
                error_code="LIMITER_ERROR",
                details={
                    "threshold_db": threshold_db,
                    "release_ms": release_ms,
                    "error": str(e)
                }
            )
    
    def apply_noise_gate(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        ratio: float = 10.0,
        attack_ms: float = 1.0,
        release_ms: float = 100.0
    ) -> np.ndarray:
        """
        Apply noise gate to reduce background noise.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            threshold_db: Gate threshold in dB (-100 to 0)
            ratio: Gate ratio (1.0 to 20.0)
            attack_ms: Attack time in milliseconds (0.1 to 100)
            release_ms: Release time in milliseconds (10 to 1000)
            
        Returns:
            Gated audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(
                f"Applying noise gate: threshold={threshold_db}dB, "
                f"ratio={ratio}:1, attack={attack_ms}ms, release={release_ms}ms"
            )
            
            # Validate parameters
            if not (-100 <= threshold_db <= 0):
                raise InvalidParameterError(f"Threshold must be -100 to 0 dB, got {threshold_db}")
            
            if not (1.0 <= ratio <= 20.0):
                raise InvalidParameterError(f"Ratio must be 1.0 to 20.0, got {ratio}")
            
            # Create noise gate
            gate = NoiseGate(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms
            )
            
            board = Pedalboard([gate])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Noise gate applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Noise gate processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply noise gate: {e}",
                error_code="GATE_ERROR",
                details={
                    "threshold_db": threshold_db,
                    "ratio": ratio,
                    "attack_ms": attack_ms,
                    "release_ms": release_ms,
                    "error": str(e)
                }
            )
    
    def apply_gain(
        self,
        audio: np.ndarray,
        gain_db: float
    ) -> np.ndarray:
        """
        Apply gain adjustment.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            gain_db: Gain in dB (-60 to +60)
            
        Returns:
            Gain-adjusted audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Applying gain: {gain_db}dB")
            
            # Validate parameters
            if not (-60 <= gain_db <= 60):
                raise InvalidParameterError(f"Gain must be -60 to +60 dB, got {gain_db}")
            
            # Create gain
            gain = Gain(gain_db=gain_db)
            
            board = Pedalboard([gain])
            processed = board(audio, self.sample_rate)
            
            logger.debug("Gain applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Gain processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply gain: {e}",
                error_code="GAIN_ERROR",
                details={"gain_db": gain_db, "error": str(e)}
            )
    
    def apply_chain(
        self,
        audio: np.ndarray,
        effects_chain: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Apply a chain of effects in sequence.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            effects_chain: List of effect configs, each with 'type' and parameters
            
        Example:
            effects_chain = [
                {"type": "eq", "low_shelf_gain_db": 3.0, "high_shelf_gain_db": -2.0},
                {"type": "compressor", "threshold_db": -20.0, "ratio": 4.0},
                {"type": "reverb", "room_size": 0.7, "wet_level": 0.3}
            ]
            
        Returns:
            Processed audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.info(f"Applying effects chain with {len(effects_chain)} effects")
            
            processed = audio.copy()
            
            for i, effect_config in enumerate(effects_chain):
                effect_type = effect_config.get("type", "").lower()
                
                logger.debug(f"Applying effect {i+1}/{len(effects_chain)}: {effect_type}")
                
                if effect_type == "eq":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_eq(processed, **params)
                
                elif effect_type == "compressor":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_compressor(processed, **params)
                
                elif effect_type == "reverb":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_reverb(processed, **params)
                
                elif effect_type == "chorus":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_chorus(processed, **params)
                
                elif effect_type == "delay":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_delay(processed, **params)
                
                elif effect_type == "limiter":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_limiter(processed, **params)
                
                elif effect_type == "gate" or effect_type == "noise_gate":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_noise_gate(processed, **params)
                
                elif effect_type == "gain":
                    params = {k: v for k, v in effect_config.items() if k != "type"}
                    processed = self.apply_gain(processed, **params)
                
                else:
                    logger.warning(f"Unknown effect type: {effect_type}, skipping")
            
            logger.info("Effects chain applied successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Effects chain processing failed: {e}")
            raise AudioProcessingError(
                f"Failed to apply effects chain: {e}",
                error_code="CHAIN_ERROR",
                details={"num_effects": len(effects_chain), "error": str(e)}
            )


def is_pedalboard_available() -> bool:
    """Check if Pedalboard library is available."""
    return PEDALBOARD_AVAILABLE
