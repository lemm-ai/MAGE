"""Tests for the MAGE core engine."""

import pytest
import numpy as np

from mage import MAGE, Config
from mage.core.engine import GeneratedAudio
from mage.exceptions import AudioGenerationError, InvalidParameterError


class TestGeneratedAudio:
    """Test GeneratedAudio functionality."""
    
    def test_initialization(self):
        """Test GeneratedAudio initialization."""
        data = np.random.randn(2, 44100)  # 1 second stereo
        audio = GeneratedAudio(data, sample_rate=44100)
        
        assert audio.sample_rate == 44100
        assert audio.duration == pytest.approx(1.0, rel=0.01)
        assert audio.data.shape == (2, 44100)
    
    def test_mono_audio(self):
        """Test GeneratedAudio with mono data."""
        data = np.random.randn(44100)  # 1 second mono
        audio = GeneratedAudio(data, sample_rate=44100)
        
        assert audio.duration == pytest.approx(1.0, rel=0.01)
    
    def test_normalize(self):
        """Test audio normalization."""
        data = np.random.randn(2, 44100) * 0.1  # Quiet audio
        audio = GeneratedAudio(data, sample_rate=44100)
        
        audio.normalize(target_level=-6.0)
        
        # Check that peak is near target
        peak_db = 20 * np.log10(np.max(np.abs(audio.data)))
        assert peak_db == pytest.approx(-6.0, abs=1.0)


class TestMAGE:
    """Test main MAGE engine functionality."""
    
    def test_initialization(self):
        """Test MAGE initialization with default config."""
        engine = MAGE()
        assert engine.config is not None
        assert engine.generator is not None
    
    def test_initialization_with_custom_config(self):
        """Test MAGE initialization with custom config."""
        config = Config()
        config.audio.sample_rate = 48000
        
        engine = MAGE(config=config)
        assert engine.config.audio.sample_rate == 48000
    
    def test_generate_default_parameters(self):
        """Test audio generation with default parameters."""
        engine = MAGE()
        audio = engine.generate()
        
        assert isinstance(audio, GeneratedAudio)
        assert audio.duration > 0
        assert audio.sample_rate == engine.config.audio.sample_rate
    
    def test_generate_custom_duration(self):
        """Test audio generation with custom duration."""
        engine = MAGE()
        duration = 5.0
        audio = engine.generate(duration=duration)
        
        assert audio.duration == pytest.approx(duration, rel=0.01)
    
    def test_generate_custom_style(self):
        """Test audio generation with custom style."""
        engine = MAGE()
        audio = engine.generate(style="electronic", tempo=140)
        
        assert audio.metadata["style"] == "electronic"
        assert audio.metadata["tempo"] == 140
    
    def test_invalid_duration(self):
        """Test that invalid duration raises InvalidParameterError."""
        engine = MAGE()
        
        with pytest.raises(InvalidParameterError):
            engine.generate(duration=-1.0)
        
        with pytest.raises(InvalidParameterError):
            engine.generate(duration=1000.0)  # Exceeds max
    
    def test_invalid_tempo(self):
        """Test that invalid tempo raises InvalidParameterError."""
        engine = MAGE()
        
        with pytest.raises(InvalidParameterError):
            engine.generate(tempo=10)  # Too low
        
        with pytest.raises(InvalidParameterError):
            engine.generate(tempo=400)  # Too high
    
    def test_invalid_complexity(self):
        """Test that invalid complexity raises InvalidParameterError."""
        engine = MAGE()
        
        with pytest.raises(InvalidParameterError):
            engine.generate(complexity=-0.1)
        
        with pytest.raises(InvalidParameterError):
            engine.generate(complexity=1.5)
    
    def test_get_available_styles(self):
        """Test getting available styles."""
        engine = MAGE()
        styles = engine.get_available_styles()
        
        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "ambient" in styles
    
    def test_get_available_keys(self):
        """Test getting available keys."""
        engine = MAGE()
        keys = engine.get_available_keys()
        
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "C_major" in keys
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same output."""
        engine = MAGE()
        
        audio1 = engine.generate(duration=1.0, seed=42)
        audio2 = engine.generate(duration=1.0, seed=42)
        
        np.testing.assert_array_almost_equal(audio1.data, audio2.data)
