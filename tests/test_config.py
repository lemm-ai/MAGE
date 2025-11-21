"""Tests for the MAGE configuration system."""

import pytest
from pathlib import Path
import tempfile
import yaml

from mage.config import Config, AudioConfig, ModelConfig, GenerationConfig
from mage.exceptions import ConfigurationError


class TestAudioConfig:
    """Test AudioConfig validation and functionality."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AudioConfig()
        assert config.sample_rate == 44100
        assert config.bit_depth == 16
        assert config.channels == 2
    
    def test_valid_sample_rates(self):
        """Test valid sample rates pass validation."""
        valid_rates = [22050, 44100, 48000, 96000]
        for rate in valid_rates:
            config = AudioConfig(sample_rate=rate)
            config.validate()  # Should not raise
    
    def test_invalid_sample_rate(self):
        """Test that invalid sample rates raise ConfigurationError."""
        config = AudioConfig(sample_rate=12000)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_invalid_bit_depth(self):
        """Test that invalid bit depths raise ConfigurationError."""
        config = AudioConfig(bit_depth=8)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_invalid_channels(self):
        """Test that invalid channel counts raise ConfigurationError."""
        config = AudioConfig(channels=5)
        with pytest.raises(ConfigurationError):
            config.validate()


class TestModelConfig:
    """Test ModelConfig validation and functionality."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()
        assert config.model_type == "default"
        assert config.device == "auto"
        assert config.precision == "float32"
    
    def test_valid_devices(self):
        """Test valid device settings pass validation."""
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        for device in valid_devices:
            config = ModelConfig(device=device)
            config.validate()  # Should not raise
    
    def test_invalid_device(self):
        """Test that invalid devices raise ConfigurationError."""
        config = ModelConfig(device="invalid")
        with pytest.raises(ConfigurationError):
            config.validate()


class TestGenerationConfig:
    """Test GenerationConfig validation and functionality."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GenerationConfig()
        assert config.default_style == "ambient"
        assert config.default_tempo == 120
        assert config.complexity == 0.5
    
    def test_valid_tempo_range(self):
        """Test valid tempo values pass validation."""
        config = GenerationConfig(default_tempo=120)
        config.validate()  # Should not raise
    
    def test_invalid_tempo_low(self):
        """Test that too low tempo raises ConfigurationError."""
        config = GenerationConfig(default_tempo=10)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_invalid_tempo_high(self):
        """Test that too high tempo raises ConfigurationError."""
        config = GenerationConfig(default_tempo=400)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_invalid_complexity(self):
        """Test that invalid complexity raises ConfigurationError."""
        config = GenerationConfig(complexity=1.5)
        with pytest.raises(ConfigurationError):
            config.validate()


class TestConfig:
    """Test main Config class functionality."""
    
    def test_default_initialization(self):
        """Test that Config initializes with defaults."""
        config = Config()
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generation, GenerationConfig)
    
    def test_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {
            "audio": {"sample_rate": 48000},
            "generation": {"default_tempo": 140}
        }
        config = Config.from_dict(config_dict)
        assert config.audio.sample_rate == 48000
        assert config.generation.default_tempo == 140
    
    def test_to_dict(self):
        """Test converting Config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        assert "audio" in config_dict
        assert "model" in config_dict
        assert "generation" in config_dict
        assert "logging" in config_dict
    
    def test_save_and_load(self):
        """Test saving and loading configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Create and save config
            config1 = Config()
            config1.audio.sample_rate = 48000
            config1.generation.default_tempo = 140
            config1.to_file(temp_path)
            
            # Load config
            config2 = Config.from_file(temp_path)
            assert config2.audio.sample_rate == 48000
            assert config2.generation.default_tempo == 140
        
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            Config.from_file("nonexistent_file.yaml")
    
    def test_update_config(self):
        """Test updating configuration values."""
        config = Config()
        updates = {
            "audio": {"sample_rate": 48000},
            "generation": {"default_tempo": 140}
        }
        config.update(updates)
        assert config.audio.sample_rate == 48000
        assert config.generation.default_tempo == 140
    
    def test_validation_on_update(self):
        """Test that validation occurs on update."""
        config = Config()
        invalid_updates = {
            "audio": {"sample_rate": 12000}  # Invalid sample rate
        }
        with pytest.raises(ConfigurationError):
            config.update(invalid_updates)
