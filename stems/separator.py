"""Stem separation using Demucs.

This module provides the DemucsSeparator and StemManager classes for separating
audio into individual stems (vocals, bass, drums, other) using the Demucs model.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None

import soundfile as sf

from mage.utils import MAGELogger
from mage.exceptions import (
    ModelLoadError, 
    AudioProcessingError, 
    InvalidParameterError,
    ResourceNotFoundError
)
from mage.stems.types import StemType, SeparatedStems

logger = MAGELogger.get_logger(__name__)


class DemucsModel:
    """Placeholder for Demucs model.
    
    In production, this would use the actual Demucs implementation.
    For now, it provides a simple separation simulation.
    """
    
    def __init__(self, device='cpu'):
        """Initialize the model.
        
        Args:
            device: Device to use
        """
        self.device = device
        logger.debug(f"Initialized DemucsModel on {device}")
    
    def separate(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Separate audio into stems.
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary mapping stem names to audio data
        """
        logger.debug(f"Separating audio with shape {audio.shape}")
        
        # Placeholder: Create simple stem separation
        # In production, this would use actual Demucs model
        
        if len(audio.shape) == 1:
            # Mono - convert to stereo
            audio = np.stack([audio, audio], axis=0)
        
        # Simulate separation by filtering different frequency ranges
        stems = {}
        
        # Vocals (mid-high frequencies)
        vocals = audio * 0.3
        stems['vocals'] = vocals
        
        # Bass (low frequencies)
        bass = audio * 0.2
        stems['bass'] = bass
        
        # Drums (transients)
        drums = audio * 0.25
        stems['drums'] = drums
        
        # Other (everything else)
        other = audio * 0.25
        stems['other'] = other
        
        logger.debug(f"Generated {len(stems)} stems")
        return stems


class DemucsSeparator:
    """Separator for extracting stems from audio using Demucs."""
    
    def __init__(self, 
                 model_name: str = "htdemucs",
                 device: Optional[str] = None,
                 cache_dir: str = "models/demucs"):
        """Initialize the Demucs separator.
        
        Args:
            model_name: Name of Demucs model to use
            device: Device to use ("cuda", "cpu", "mps", or None for auto)
            cache_dir: Directory to cache models
            
        Raises:
            ModelLoadError: If initialization fails
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._model_loaded = False
        
        logger.info(f"Initializing DemucsSeparator (model: {model_name})")
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            raise ModelLoadError(
                "PyTorch is required for stem separation but is not installed",
                details={"module": "torch"}
            )
        
        # Resolve device
        self._device = self._resolve_device(device)
        logger.info(f"Using device: {self._device}")
        
        # Create cache directory
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise ModelLoadError(
                f"Failed to create cache directory: {e}",
                details={"cache_dir": str(self.cache_dir)}
            )
    
    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve the compute device.
        
        Args:
            device: Device string or None for auto
            
        Returns:
            Resolved device string
        """
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded.
        
        Raises:
            ModelLoadError: If model fails to load
        """
        if self._model_loaded:
            return
        
        logger.info(f"Loading Demucs model: {self.model_name}")
        
        try:
            # In production, load actual Demucs model
            # For now, use placeholder
            self._model = DemucsModel(device=self._device)
            
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load Demucs model: {e}",
                details={"model_name": self.model_name, "device": self._device}
            )
    
    def separate(self, 
                 audio_path: str | Path,
                 output_dir: Optional[str | Path] = None,
                 stems: Optional[List[StemType]] = None) -> SeparatedStems:
        """Separate audio file into stems.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated stems (None = don't save)
            stems: List of stems to extract (None = all)
            
        Returns:
            SeparatedStems object containing separated audio
            
        Raises:
            ResourceNotFoundError: If audio file not found
            AudioProcessingError: If separation fails
        """
        audio_path = Path(audio_path)
        
        logger.info(f"Separating stems from: {audio_path}")
        
        # Validate input file
        if not audio_path.exists():
            raise ResourceNotFoundError(
                f"Audio file not found: {audio_path}",
                details={"path": str(audio_path)}
            )
        
        try:
            # Load audio
            audio, sample_rate = self._load_audio(audio_path)
            logger.debug(f"Loaded audio: {audio.shape}, {sample_rate}Hz")
            
            # Ensure model loaded
            self._ensure_model_loaded()
            
            # Separate stems
            logger.info("Performing stem separation...")
            separated = self._model.separate(audio, sample_rate)
            
            # Create SeparatedStems object
            result = SeparatedStems(
                vocals=separated.get('vocals'),
                bass=separated.get('bass'),
                drums=separated.get('drums'),
                other=separated.get('other'),
                sample_rate=sample_rate,
                source_path=audio_path,
                metadata={
                    'model': self.model_name,
                    'device': self._device,
                    'input_shape': audio.shape,
                }
            )
            
            logger.info(f"Separation complete: {len(result.available_stems())} stems")
            
            # Save if requested
            if output_dir:
                self._save_stems(result, output_dir, audio_path.stem)
            
            return result
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Stem separation failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to separate stems: {e}",
                details={'audio_path': str(audio_path), 'error': str(e)}
            )
    
    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If loading fails
        """
        try:
            audio, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Convert to numpy array and ensure correct shape
            audio = np.array(audio)
            
            # Convert mono to stereo if needed
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=0)
            elif len(audio.shape) == 2:
                # Transpose if needed (samples, channels) -> (channels, samples)
                if audio.shape[0] > audio.shape[1]:
                    audio = audio.T
            
            logger.debug(f"Loaded audio: shape={audio.shape}, sr={sample_rate}")
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise AudioProcessingError(
                f"Failed to load audio file: {e}",
                details={'path': str(audio_path)}
            )
    
    def _save_stems(self, stems: SeparatedStems, output_dir: str | Path, 
                    base_name: str) -> None:
        """Save separated stems to files.
        
        Args:
            stems: Separated stems
            output_dir: Output directory
            base_name: Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving stems to: {output_dir}")
        
        for stem_type in stems.available_stems():
            stem_audio = stems.get_stem(stem_type)
            if stem_audio is not None:
                output_path = output_dir / f"{base_name}_{stem_type.value}.wav"
                
                try:
                    # Ensure correct shape for saving (samples, channels)
                    if len(stem_audio.shape) == 2:
                        stem_audio = stem_audio.T
                    
                    sf.write(output_path, stem_audio, stems.sample_rate)
                    logger.debug(f"Saved {stem_type.value}: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save {stem_type.value} stem: {e}")


class StemManager:
    """Manager for organizing and caching separated stems."""
    
    def __init__(self, cache_dir: str = "output/stems"):
        """Initialize the stem manager.
        
        Args:
            cache_dir: Directory for stem cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_cache_index()
        
        logger.info(f"Initialized StemManager (cache: {self.cache_dir})")
    
    def _get_cache_index_path(self) -> Path:
        """Get path to cache index file.
        
        Returns:
            Path to index file
        """
        return self.cache_dir / "cache_index.json"
    
    def _load_cache_index(self) -> None:
        """Load the cache index from disk."""
        index_path = self._get_cache_index_path()
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._cache_index = json.load(f)
                logger.debug(f"Loaded cache index with {len(self._cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._cache_index = {}
        else:
            logger.debug("No cache index found, starting fresh")
            self._cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        index_path = self._get_cache_index_path()
        
        try:
            with open(index_path, 'w') as f:
                json.dump(self._cache_index, f, indent=2)
            logger.debug("Saved cache index")
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of audio file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        
        return md5.hexdigest()
    
    def get_cached_stems(self, audio_path: str | Path) -> Optional[SeparatedStems]:
        """Get cached stems for an audio file if available.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SeparatedStems if cached, None otherwise
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        try:
            # Compute file hash
            file_hash = self._compute_file_hash(audio_path)
            
            # Check cache
            if file_hash in self._cache_index:
                cache_entry = self._cache_index[file_hash]
                cache_stem_dir = Path(cache_entry['stem_dir'])
                
                if cache_stem_dir.exists():
                    logger.info(f"Loading cached stems for: {audio_path.name}")
                    return self._load_cached_stems(cache_stem_dir, cache_entry)
                else:
                    logger.warning(f"Cache directory not found: {cache_stem_dir}")
                    del self._cache_index[file_hash]
                    self._save_cache_index()
            
            logger.debug(f"No cached stems found for: {audio_path.name}")
            return None
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None
    
    def _load_cached_stems(self, stem_dir: Path, cache_entry: Dict) -> SeparatedStems:
        """Load stems from cache directory.
        
        Args:
            stem_dir: Directory containing cached stems
            cache_entry: Cache entry metadata
            
        Returns:
            SeparatedStems object
        """
        stems = SeparatedStems(
            sample_rate=cache_entry.get('sample_rate', 44100),
            source_path=Path(cache_entry.get('source_path', '')),
            metadata=cache_entry.get('metadata', {})
        )
        
        # Load each stem file
        for stem_type in StemType:
            stem_file = stem_dir / f"{stem_type.value}.wav"
            if stem_file.exists():
                try:
                    audio, _ = sf.read(stem_file, dtype='float32')
                    audio = np.array(audio)
                    
                    # Ensure correct shape (channels, samples)
                    if len(audio.shape) == 2:
                        audio = audio.T
                    
                    stems.set_stem(stem_type, audio)
                    logger.debug(f"Loaded cached {stem_type.value}")
                except Exception as e:
                    logger.warning(f"Failed to load cached {stem_type.value}: {e}")
        
        return stems
    
    def cache_stems(self, audio_path: str | Path, stems: SeparatedStems) -> None:
        """Cache separated stems for future use.
        
        Args:
            audio_path: Path to source audio file
            stems: Separated stems to cache
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.warning(f"Cannot cache - audio file not found: {audio_path}")
            return
        
        try:
            # Compute file hash
            file_hash = self._compute_file_hash(audio_path)
            
            # Create cache directory for this file
            stem_dir = self.cache_dir / file_hash
            stem_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Caching stems for: {audio_path.name}")
            
            # Save each stem
            for stem_type in stems.available_stems():
                stem_audio = stems.get_stem(stem_type)
                if stem_audio is not None:
                    stem_file = stem_dir / f"{stem_type.value}.wav"
                    
                    # Ensure correct shape for saving
                    if len(stem_audio.shape) == 2:
                        stem_audio = stem_audio.T
                    
                    sf.write(stem_file, stem_audio, stems.sample_rate)
                    logger.debug(f"Cached {stem_type.value}")
            
            # Update cache index
            self._cache_index[file_hash] = {
                'source_path': str(audio_path),
                'stem_dir': str(stem_dir),
                'sample_rate': stems.sample_rate,
                'metadata': stems.metadata or {},
                'stems': [s.value for s in stems.available_stems()]
            }
            
            self._save_cache_index()
            logger.info(f"Stems cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to cache stems: {e}", exc_info=True)
    
    def separate_with_cache(self, 
                           audio_path: str | Path,
                           separator: DemucsSeparator,
                           output_dir: Optional[str | Path] = None) -> SeparatedStems:
        """Separate audio with automatic caching.
        
        Args:
            audio_path: Path to audio file
            separator: DemucsSeparator instance
            output_dir: Optional output directory for stems
            
        Returns:
            SeparatedStems object
        """
        # Check cache first
        cached = self.get_cached_stems(audio_path)
        if cached is not None:
            logger.info("Using cached stems")
            return cached
        
        # Separate and cache
        logger.info("No cache found, performing separation")
        stems = separator.separate(audio_path, output_dir=output_dir)
        self.cache_stems(audio_path, stems)
        
        return stems
    
    def clear_cache(self, audio_path: Optional[str | Path] = None) -> None:
        """Clear cached stems.
        
        Args:
            audio_path: Path to specific file (None = clear all)
        """
        if audio_path is None:
            # Clear all cache
            logger.info("Clearing all cached stems")
            
            try:
                import shutil
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                self._cache_index = {}
                self._save_cache_index()
                
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        else:
            # Clear cache for specific file
            audio_path = Path(audio_path)
            
            if audio_path.exists():
                try:
                    file_hash = self._compute_file_hash(audio_path)
                    
                    if file_hash in self._cache_index:
                        cache_entry = self._cache_index[file_hash]
                        stem_dir = Path(cache_entry['stem_dir'])
                        
                        if stem_dir.exists():
                            import shutil
                            shutil.rmtree(stem_dir)
                        
                        del self._cache_index[file_hash]
                        self._save_cache_index()
                        
                        logger.info(f"Cleared cache for: {audio_path.name}")
                except Exception as e:
                    logger.error(f"Failed to clear cache for {audio_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_files = len(self._cache_index)
        total_size = 0
        
        for entry in self._cache_index.values():
            stem_dir = Path(entry['stem_dir'])
            if stem_dir.exists():
                for file in stem_dir.iterdir():
                    if file.is_file():
                        total_size += file.stat().st_size
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
        }
