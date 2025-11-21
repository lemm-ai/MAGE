"""Vocal enhancement using AI-based denoising and quality improvement.

This module provides the VocalEnhancer class for improving vocal quality through
denoising, spectral enhancement, and dynamic processing.

Based on FullSubNet architecture from:
https://github.com/wimmerb/singing-quality-enhancement
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    T = None

try:
    import librosa
    import scipy.signal
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    scipy = None

from mage.utils import MAGELogger
from mage.exceptions import (
    ModelLoadError,
    AudioProcessingError,
    InvalidParameterError,
    ResourceNotFoundError
)

logger = MAGELogger.get_logger(__name__)


def is_enhancement_available() -> bool:
    """Check if vocal enhancement is available.
    
    Returns:
        True if all required dependencies are installed
    """
    return TORCH_AVAILABLE and TORCHAUDIO_AVAILABLE and LIBROSA_AVAILABLE


class FullSubNetModel(nn.Module):
    """Simplified FullSubNet model for vocal enhancement.
    
    This is a lightweight implementation inspired by FullSubNet architecture
    for singing quality enhancement. In production, this would load the actual
    pretrained model from singing-quality-enhancement repository.
    
    Architecture:
    - Sub-band feature extraction
    - Full-band feature extraction
    - Attention-based fusion
    - Mask estimation
    """
    
    def __init__(self, 
                 n_fft: int = 512,
                 hop_length: int = 256,
                 num_freqs: int = 257):
        """Initialize FullSubNet model.
        
        Args:
            n_fft: FFT size
            hop_length: Hop length for STFT
            num_freqs: Number of frequency bins
        """
        if not TORCH_AVAILABLE:
            raise ModelLoadError(
                "PyTorch is required but not installed",
                details={"module": "torch"}
            )
        
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freqs = num_freqs
        
        # Sub-band LSTM
        self.subband_lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Full-band LSTM
        self.fullband_lstm = nn.LSTM(
            input_size=num_freqs,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # Fusion and mask estimation
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        logger.debug("Initialized FullSubNet model")
    
    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            noisy_mag: Noisy magnitude spectrogram [batch, freq, time]
            
        Returns:
            Estimated mask [batch, freq, time]
        """
        batch_size, num_freqs, num_frames = noisy_mag.shape
        
        # Transpose for processing: [batch, time, freq]
        noisy_mag_t = noisy_mag.transpose(1, 2)
        
        # Full-band processing
        fullband_out, _ = self.fullband_lstm(noisy_mag_t)  # [batch, time, 256]
        
        # Process each frequency band
        masks = []
        for freq_idx in range(num_freqs):
            # Get this frequency's time series
            freq_band = noisy_mag[:, freq_idx:freq_idx+1, :].transpose(1, 2)  # [batch, time, 1]
            
            # Sub-band LSTM
            subband_out, _ = self.subband_lstm(freq_band)  # [batch, time, 256]
            
            # Combine with full-band features
            combined = torch.cat([subband_out, fullband_out], dim=-1)  # [batch, time, 512]
            
            # Estimate mask for this frequency
            freq_mask = self.fusion(combined)  # [batch, time, 1]
            masks.append(freq_mask.squeeze(-1))  # [batch, time]
        
        # Stack frequency masks
        mask = torch.stack(masks, dim=1)  # [batch, freq, time]
        
        return mask


class VocalEnhancer:
    """Vocal enhancement using AI-based denoising and quality improvement.
    
    This class provides methods for:
    - Background noise removal
    - Spectral enhancement
    - Dynamic range optimization
    - Vocal clarity improvement
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 device: Optional[str] = None,
                 model_path: Optional[str | Path] = None,
                 cache_dir: str = "models/vocals"):
        """Initialize vocal enhancer.
        
        Args:
            sample_rate: Sample rate for processing
            device: Device to use ("cuda", "cpu", "mps", or None for auto)
            model_path: Path to pretrained model (None = use default)
            cache_dir: Directory to cache models
            
        Raises:
            ModelLoadError: If initialization fails
        """
        if not is_enhancement_available():
            raise ModelLoadError(
                "Vocal enhancement requires torch, torchaudio, and librosa",
                details={
                    "torch": TORCH_AVAILABLE,
                    "torchaudio": TORCHAUDIO_AVAILABLE,
                    "librosa": LIBROSA_AVAILABLE
                }
            )
        
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir)
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._model_loaded = False
        
        logger.info(f"Initializing VocalEnhancer (sample_rate: {sample_rate}Hz)")
        
        # Resolve device
        self._device = self._resolve_device(device)
        logger.info(f"Using device: {self._device}")
        
        # STFT parameters
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve device for processing.
        
        Args:
            device: Requested device or None for auto
            
        Returns:
            Device string
        """
        if device and device != "auto":
            return device
        
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded.
        
        Raises:
            ModelLoadError: If model fails to load
        """
        if self._model_loaded:
            return
        
        logger.info("Loading vocal enhancement model...")
        
        try:
            # Try to load pretrained model if path provided
            if self.model_path and self.model_path.exists():
                logger.info(f"Loading pretrained model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self._device)
                self._model = FullSubNetModel(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    num_freqs=self.n_fft // 2 + 1
                )
                self._model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            else:
                # Use default model (placeholder)
                logger.info("Using default model (placeholder)")
                self._model = FullSubNetModel(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    num_freqs=self.n_fft // 2 + 1
                )
            
            self._model.to(self._device)
            self._model.eval()
            
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load vocal enhancement model: {e}",
                details={"device": self._device, "model_path": str(self.model_path)}
            )
    
    def _validate_audio(self, audio: np.ndarray) -> None:
        """Validate audio array.
        
        Args:
            audio: Audio array to validate
            
        Raises:
            InvalidParameterError: If audio is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise InvalidParameterError(
                "Audio must be numpy array",
                parameter="audio",
                value=type(audio).__name__
            )
        
        if audio.dtype not in [np.float32, np.float64]:
            raise InvalidParameterError(
                "Audio must be float32 or float64",
                parameter="audio.dtype",
                value=str(audio.dtype)
            )
        
        if len(audio.shape) > 2:
            raise InvalidParameterError(
                "Audio must be 1D (mono) or 2D (stereo)",
                parameter="audio.shape",
                value=audio.shape
            )
    
    def denoise(self,
                audio: np.ndarray,
                noise_reduction: float = 0.8) -> np.ndarray:
        """Remove background noise from vocal audio.
        
        Args:
            audio: Input audio (mono or stereo)
            noise_reduction: Noise reduction strength (0.0 to 1.0)
            
        Returns:
            Denoised audio with same shape as input
            
        Raises:
            AudioProcessingError: If denoising fails
            InvalidParameterError: If parameters are invalid
        """
        logger.debug(f"Starting denoise (reduction: {noise_reduction})")
        
        # Validate inputs
        self._validate_audio(audio)
        
        if not 0.0 <= noise_reduction <= 1.0:
            raise InvalidParameterError(
                "noise_reduction must be between 0.0 and 1.0",
                parameter="noise_reduction",
                value=noise_reduction
            )
        
        try:
            # Ensure model loaded
            self._ensure_model_loaded()
            
            # Handle stereo
            is_stereo = len(audio.shape) == 2
            if is_stereo:
                logger.debug(f"Processing stereo audio: {audio.shape}")
                left = self._denoise_mono(audio[0], noise_reduction)
                right = self._denoise_mono(audio[1], noise_reduction)
                result = np.stack([left, right], axis=0)
            else:
                logger.debug(f"Processing mono audio: {audio.shape}")
                result = self._denoise_mono(audio, noise_reduction)
            
            logger.info("Denoising completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to denoise audio: {e}",
                details={"shape": audio.shape, "reduction": noise_reduction}
            )
    
    def _denoise_mono(self, audio: np.ndarray, noise_reduction: float) -> np.ndarray:
        """Denoise mono audio using FullSubNet model.
        
        Args:
            audio: Mono audio array
            noise_reduction: Noise reduction strength
            
        Returns:
            Denoised mono audio
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self._device)
        
        # Compute STFT
        stft = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(self._device),
            return_complex=True
        )
        
        # Get magnitude and phase
        noisy_mag = torch.abs(stft).unsqueeze(0)  # [1, freq, time]
        phase = torch.angle(stft)
        
        # Estimate mask using model
        with torch.no_grad():
            mask = self._model(noisy_mag).squeeze(0)  # [freq, time]
        
        # Apply noise reduction scaling
        mask = mask * noise_reduction + (1.0 - noise_reduction)
        
        # Apply mask to magnitude
        enhanced_mag = torch.abs(stft) * mask
        
        # Reconstruct complex spectrogram
        enhanced_stft = enhanced_mag * torch.exp(1j * phase)
        
        # Inverse STFT
        enhanced = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(self._device),
            length=len(audio)
        )
        
        return enhanced.cpu().numpy()
    
    def enhance_spectral(self,
                         audio: np.ndarray,
                         brightness: float = 0.0,
                         warmth: float = 0.0,
                         clarity: float = 0.0) -> np.ndarray:
        """Enhance spectral characteristics of vocal audio.
        
        Args:
            audio: Input audio (mono or stereo)
            brightness: High frequency boost (-1.0 to 1.0, dB)
            warmth: Low-mid frequency boost (-1.0 to 1.0, dB)
            clarity: Mid-high frequency boost (-1.0 to 1.0, dB)
            
        Returns:
            Spectrally enhanced audio
            
        Raises:
            AudioProcessingError: If enhancement fails
            InvalidParameterError: If parameters are invalid
        """
        logger.debug(f"Spectral enhancement (bright: {brightness}, warm: {warmth}, clear: {clarity})")
        
        # Validate inputs
        self._validate_audio(audio)
        
        for param, value in [("brightness", brightness), ("warmth", warmth), ("clarity", clarity)]:
            if not -1.0 <= value <= 1.0:
                raise InvalidParameterError(
                    f"{param} must be between -1.0 and 1.0",
                    parameter=param,
                    value=value
                )
        
        try:
            # Handle stereo
            is_stereo = len(audio.shape) == 2
            if is_stereo:
                left = self._enhance_spectral_mono(audio[0], brightness, warmth, clarity)
                right = self._enhance_spectral_mono(audio[1], brightness, warmth, clarity)
                result = np.stack([left, right], axis=0)
            else:
                result = self._enhance_spectral_mono(audio, brightness, warmth, clarity)
            
            logger.info("Spectral enhancement completed")
            return result
            
        except Exception as e:
            logger.error(f"Spectral enhancement failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to enhance audio spectrally: {e}",
                details={"brightness": brightness, "warmth": warmth, "clarity": clarity}
            )
    
    def _enhance_spectral_mono(self,
                               audio: np.ndarray,
                               brightness: float,
                               warmth: float,
                               clarity: float) -> np.ndarray:
        """Enhance spectral characteristics of mono audio.
        
        Args:
            audio: Mono audio
            brightness: High frequency boost
            warmth: Low-mid frequency boost
            clarity: Mid-high frequency boost
            
        Returns:
            Enhanced mono audio
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # Create frequency-dependent gains
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        gains = np.ones_like(freqs)
        
        # Warmth (boost 200-800 Hz)
        if warmth != 0.0:
            warmth_mask = (freqs >= 200) & (freqs <= 800)
            gains[warmth_mask] *= (1.0 + warmth * 0.15)  # Up to ±15%
        
        # Clarity (boost 2-5 kHz)
        if clarity != 0.0:
            clarity_mask = (freqs >= 2000) & (freqs <= 5000)
            gains[clarity_mask] *= (1.0 + clarity * 0.2)  # Up to ±20%
        
        # Brightness (boost 5-12 kHz)
        if brightness != 0.0:
            brightness_mask = (freqs >= 5000) & (freqs <= 12000)
            gains[brightness_mask] *= (1.0 + brightness * 0.25)  # Up to ±25%
        
        # Apply gains
        gains_2d = gains[:, np.newaxis]
        enhanced_mag = mag * gains_2d
        
        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=len(audio)
        )
        
        return enhanced
    
    def optimize_dynamics(self,
                          audio: np.ndarray,
                          target_level: float = -20.0,
                          compression_ratio: float = 3.0) -> np.ndarray:
        """Optimize dynamic range of vocal audio.
        
        Args:
            audio: Input audio (mono or stereo)
            target_level: Target RMS level in dB (-40 to 0)
            compression_ratio: Compression ratio (1.0 to 10.0)
            
        Returns:
            Dynamically optimized audio
            
        Raises:
            AudioProcessingError: If optimization fails
            InvalidParameterError: If parameters are invalid
        """
        logger.debug(f"Dynamic optimization (target: {target_level}dB, ratio: {compression_ratio})")
        
        # Validate inputs
        self._validate_audio(audio)
        
        if not -40.0 <= target_level <= 0.0:
            raise InvalidParameterError(
                "target_level must be between -40.0 and 0.0 dB",
                parameter="target_level",
                value=target_level
            )
        
        if not 1.0 <= compression_ratio <= 10.0:
            raise InvalidParameterError(
                "compression_ratio must be between 1.0 and 10.0",
                parameter="compression_ratio",
                value=compression_ratio
            )
        
        try:
            # Simple compression and normalization
            # In production, use more sophisticated dynamics processing
            
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio ** 2))
            current_db = 20 * np.log10(rms + 1e-10)
            
            logger.debug(f"Current RMS: {current_db:.2f} dB")
            
            # Apply simple compression (reduce peaks)
            threshold = 0.5
            compressed = np.copy(audio)
            
            # Compress peaks above threshold
            peaks = np.abs(compressed) > threshold
            if np.any(peaks):
                excess = np.abs(compressed[peaks]) - threshold
                compressed[peaks] = np.sign(compressed[peaks]) * (
                    threshold + excess / compression_ratio
                )
            
            # Normalize to target level
            target_linear = 10 ** (target_level / 20)
            current_rms = np.sqrt(np.mean(compressed ** 2))
            gain = target_linear / (current_rms + 1e-10)
            
            # Limit gain to prevent clipping
            max_gain = 0.95 / (np.max(np.abs(compressed)) + 1e-10)
            gain = min(gain, max_gain)
            
            result = compressed * gain
            
            final_rms = np.sqrt(np.mean(result ** 2))
            final_db = 20 * np.log10(final_rms + 1e-10)
            logger.info(f"Dynamic optimization complete (RMS: {final_db:.2f} dB)")
            
            return result
            
        except Exception as e:
            logger.error(f"Dynamic optimization failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to optimize dynamics: {e}",
                details={"target_level": target_level, "compression_ratio": compression_ratio}
            )
    
    def enhance(self,
                audio: np.ndarray,
                denoise_strength: float = 0.7,
                brightness: float = 0.2,
                warmth: float = 0.1,
                clarity: float = 0.3,
                target_level: float = -18.0) -> np.ndarray:
        """Full vocal enhancement pipeline.
        
        Applies denoising, spectral enhancement, and dynamic optimization.
        
        Args:
            audio: Input audio (mono or stereo)
            denoise_strength: Noise reduction strength (0.0 to 1.0)
            brightness: High frequency boost (-1.0 to 1.0)
            warmth: Low-mid frequency boost (-1.0 to 1.0)
            clarity: Mid-high frequency boost (-1.0 to 1.0)
            target_level: Target RMS level in dB (-40 to 0)
            
        Returns:
            Fully enhanced vocal audio
            
        Raises:
            AudioProcessingError: If enhancement fails
        """
        logger.info("Starting full vocal enhancement pipeline")
        
        try:
            # Step 1: Denoise
            if denoise_strength > 0.0:
                logger.debug("Step 1: Denoising")
                audio = self.denoise(audio, noise_reduction=denoise_strength)
            
            # Step 2: Spectral enhancement
            if brightness != 0.0 or warmth != 0.0 or clarity != 0.0:
                logger.debug("Step 2: Spectral enhancement")
                audio = self.enhance_spectral(
                    audio,
                    brightness=brightness,
                    warmth=warmth,
                    clarity=clarity
                )
            
            # Step 3: Dynamic optimization
            logger.debug("Step 3: Dynamic optimization")
            audio = self.optimize_dynamics(
                audio,
                target_level=target_level,
                compression_ratio=3.0
            )
            
            logger.info("Full enhancement pipeline completed successfully")
            return audio
            
        except Exception as e:
            logger.error(f"Enhancement pipeline failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to enhance vocal audio: {e}",
                details={
                    "denoise_strength": denoise_strength,
                    "brightness": brightness,
                    "warmth": warmth,
                    "clarity": clarity,
                    "target_level": target_level
                }
            )
