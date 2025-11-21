"""Audio processing utilities.

This module provides audio processing capabilities including effects,
normalization, and other post-processing operations.
"""

import logging
import numpy as np
from typing import Dict, Optional

from mage.exceptions import AudioProcessingError
from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


class AudioProcessor:
    """Audio processor for applying effects and transformations.
    
    This class provides various audio processing operations including
    reverb, compression, EQ, and normalization.
    """
    
    def __init__(self):
        """Initialize the audio processor."""
        logger.debug("Initialized AudioProcessor")
    
    def apply_reverb(
        self,
        audio: np.ndarray,
        amount: float,
        sample_rate: int
    ) -> np.ndarray:
        """Apply reverb effect to audio.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
            amount: Reverb amount (0.0 to 1.0)
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Applying reverb: amount={amount}")
            
            # Simple reverb using delay and feedback
            # This is a placeholder - replace with better reverb algorithm
            delay_samples = int(sample_rate * 0.05)  # 50ms delay
            feedback = amount * 0.5
            
            if audio.ndim == 1:
                # Mono
                output = audio.copy()
                for i in range(delay_samples, len(audio)):
                    output[i] += audio[i - delay_samples] * feedback
            else:
                # Stereo
                output = audio.copy()
                for ch in range(audio.shape[0]):
                    for i in range(delay_samples, audio.shape[1]):
                        output[ch, i] += audio[ch, i - delay_samples] * feedback
            
            # Mix with original
            output = audio * (1 - amount) + output * amount
            
            return output
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to apply reverb: {e}",
                details={"amount": amount, "error": str(e)}
            )
    
    def apply_compression(
        self,
        audio: np.ndarray,
        amount: float
    ) -> np.ndarray:
        """Apply dynamic range compression.
        
        Args:
            audio: Audio data
            amount: Compression amount (0.0 to 1.0)
            
        Returns:
            Compressed audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Applying compression: amount={amount}")
            
            # Simple soft-knee compression
            threshold = 1.0 - amount * 0.5
            ratio = 1.0 + amount * 3.0  # Up to 4:1 ratio
            
            compressed = audio.copy()
            
            # Apply compression
            if audio.ndim == 1:
                mask = np.abs(compressed) > threshold
                over = np.abs(compressed[mask]) - threshold
                compressed[mask] = np.sign(compressed[mask]) * (
                    threshold + over / ratio
                )
            else:
                for ch in range(audio.shape[0]):
                    mask = np.abs(compressed[ch]) > threshold
                    over = np.abs(compressed[ch, mask]) - threshold
                    compressed[ch, mask] = np.sign(compressed[ch, mask]) * (
                        threshold + over / ratio
                    )
            
            return compressed
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to apply compression: {e}",
                details={"amount": amount, "error": str(e)}
            )
    
    def apply_eq(
        self,
        audio: np.ndarray,
        eq_params: Dict[str, float],
        sample_rate: int
    ) -> np.ndarray:
        """Apply equalization to audio.
        
        Args:
            audio: Audio data
            eq_params: EQ parameters (e.g., {"bass": 0.5, "treble": -0.2})
            sample_rate: Sample rate in Hz
            
        Returns:
            Equalized audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Applying EQ: {eq_params}")
            
            # Simple EQ using frequency-domain filtering
            # This is a placeholder - replace with proper EQ implementation
            output = audio.copy()
            
            # For now, just apply simple gain adjustments
            # TODO: Implement proper parametric EQ
            
            if "bass" in eq_params:
                # Boost/cut bass
                output *= (1.0 + eq_params["bass"] * 0.3)
            
            if "treble" in eq_params:
                # Boost/cut treble
                output *= (1.0 + eq_params["treble"] * 0.3)
            
            return output
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to apply EQ: {e}",
                details={"eq_params": eq_params, "error": str(e)}
            )
    
    def normalize(
        self,
        audio: np.ndarray,
        target_db: float = -3.0
    ) -> np.ndarray:
        """Normalize audio to target level.
        
        Args:
            audio: Audio data
            target_db: Target level in dB
            
        Returns:
            Normalized audio
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.debug(f"Normalizing to {target_db} dB")
            
            # Calculate current peak level
            peak = np.max(np.abs(audio))
            
            if peak == 0:
                logger.warning("Audio is silent, skipping normalization")
                return audio
            
            # Convert target dB to linear scale
            target_linear = 10 ** (target_db / 20.0)
            
            # Calculate gain needed
            gain = target_linear / peak
            
            # Apply gain
            normalized = audio * gain
            
            logger.debug(f"Applied gain: {20 * np.log10(gain):.2f} dB")
            
            return normalized
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to normalize audio: {e}",
                details={"target_db": target_db, "error": str(e)}
            )
