"""Tests for audio processing functionality."""

import pytest
import numpy as np

from mage.processors.audio_processor import AudioProcessor
from mage.exceptions import AudioProcessingError


class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create an AudioProcessor instance."""
        return AudioProcessor()
    
    @pytest.fixture
    def test_audio_mono(self):
        """Create test mono audio data."""
        return np.random.randn(44100).astype(np.float32)
    
    @pytest.fixture
    def test_audio_stereo(self):
        """Create test stereo audio data."""
        return np.random.randn(2, 44100).astype(np.float32)
    
    def test_apply_reverb_mono(self, processor, test_audio_mono):
        """Test applying reverb to mono audio."""
        processed = processor.apply_reverb(
            test_audio_mono,
            amount=0.5,
            sample_rate=44100
        )
        
        assert processed.shape == test_audio_mono.shape
        assert not np.array_equal(processed, test_audio_mono)
    
    def test_apply_reverb_stereo(self, processor, test_audio_stereo):
        """Test applying reverb to stereo audio."""
        processed = processor.apply_reverb(
            test_audio_stereo,
            amount=0.5,
            sample_rate=44100
        )
        
        assert processed.shape == test_audio_stereo.shape
        assert not np.array_equal(processed, test_audio_stereo)
    
    def test_apply_compression_mono(self, processor, test_audio_mono):
        """Test applying compression to mono audio."""
        processed = processor.apply_compression(test_audio_mono, amount=0.5)
        
        assert processed.shape == test_audio_mono.shape
        # Peak should be reduced
        assert np.max(np.abs(processed)) <= np.max(np.abs(test_audio_mono))
    
    def test_apply_compression_stereo(self, processor, test_audio_stereo):
        """Test applying compression to stereo audio."""
        processed = processor.apply_compression(test_audio_stereo, amount=0.5)
        
        assert processed.shape == test_audio_stereo.shape
    
    def test_apply_eq(self, processor, test_audio_mono):
        """Test applying EQ to audio."""
        eq_params = {"bass": 0.5, "treble": -0.2}
        processed = processor.apply_eq(
            test_audio_mono,
            eq_params,
            sample_rate=44100
        )
        
        assert processed.shape == test_audio_mono.shape
    
    def test_normalize(self, processor, test_audio_mono):
        """Test audio normalization."""
        # Scale down the audio
        quiet_audio = test_audio_mono * 0.1
        
        # Normalize to -6 dB
        normalized = processor.normalize(quiet_audio, target_db=-6.0)
        
        # Check that peak is near -6 dB
        peak_db = 20 * np.log10(np.max(np.abs(normalized)))
        assert peak_db == pytest.approx(-6.0, abs=1.0)
    
    def test_normalize_silent_audio(self, processor):
        """Test normalizing silent audio."""
        silent = np.zeros(44100, dtype=np.float32)
        normalized = processor.normalize(silent, target_db=-6.0)
        
        # Should return unchanged
        np.testing.assert_array_equal(normalized, silent)
