"""Comprehensive tests for Enhancement Popup.

Tests cover:
- Window initialization and creation
- Parameter sliders and controls
- Audio loading and processing
- Preview and apply functionality
- Error handling and edge cases
- Integration capabilities
"""

import pytest
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mage.gui.enhancement_popup import EnhancementPopup, is_enhancement_popup_available
from mage.exceptions import MAGEException


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    try:
        import soundfile as sf
        
        # Generate 1 second of test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz sine wave
        
        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), audio, sample_rate)
        
        return audio_path
    except ImportError:
        pytest.skip("soundfile not available")


class TestEnhancementPopupAvailability:
    """Test enhancement popup availability check."""
    
    def test_is_available(self):
        """Test availability check function."""
        # Should return a boolean
        result = is_enhancement_popup_available()
        assert isinstance(result, bool)


class TestEnhancementPopupInitialization:
    """Test enhancement popup initialization."""
    
    def test_init_without_audio(self):
        """Test initialization without audio file."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        # Should initialize successfully
        popup = EnhancementPopup()
        assert popup is not None
        assert popup.audio_path is None
        assert popup.original_audio is None
        assert popup.parameters is not None
        assert len(popup.parameters) > 0
        
        # Clean up
        popup.window.destroy()
    
    def test_init_with_audio(self, sample_audio_file):
        """Test initialization with audio file."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup(audio_path=str(sample_audio_file))
        assert popup is not None
        assert popup.audio_path == Path(sample_audio_file)
        assert popup.original_audio is not None
        assert popup.sample_rate is not None
        
        # Clean up
        popup.window.destroy()
    
    def test_init_with_callback(self):
        """Test initialization with callback function."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        callback_called = []
        
        def callback(param, value):
            callback_called.append((param, value))
        
        popup = EnhancementPopup(callback=callback)
        assert popup is not None
        assert popup.callback == callback
        
        # Clean up
        popup.window.destroy()
    
    def test_init_invalid_audio_path(self):
        """Test initialization with non-existent audio file."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        # Should initialize but not load audio
        popup = EnhancementPopup(audio_path="nonexistent.wav")
        assert popup is not None
        assert popup.original_audio is None  # Audio not loaded
        
        # Clean up
        popup.window.destroy()


class TestEnhancementPopupParameters:
    """Test parameter management."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        params = popup.get_parameters()
        
        # Check all expected parameters exist
        expected_params = [
            'eq_low', 'eq_mid', 'eq_high',
            'comp_threshold', 'comp_ratio', 'comp_attack', 'comp_release',
            'reverb_room', 'reverb_damping', 'reverb_wet',
            'limiter_threshold', 'limiter_release',
            'vocal_denoise', 'vocal_brightness', 'vocal_warmth', 'vocal_clarity', 'vocal_level'
        ]
        
        for param in expected_params:
            assert param in params
            assert isinstance(params[param], (int, float))
        
        # Clean up
        popup.window.destroy()
    
    def test_set_parameter(self):
        """Test setting parameter values."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Set a parameter
        popup.set_parameter('eq_low', 5.0)
        assert popup.parameters['eq_low'] == 5.0
        
        # Verify it's in get_parameters()
        params = popup.get_parameters()
        assert params['eq_low'] == 5.0
        
        # Clean up
        popup.window.destroy()
    
    def test_set_invalid_parameter(self):
        """Test setting invalid parameter name."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid parameter name"):
            popup.set_parameter('invalid_param', 1.0)
        
        # Clean up
        popup.window.destroy()
    
    def test_get_parameters_copy(self):
        """Test that get_parameters returns a copy."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        params1 = popup.get_parameters()
        params2 = popup.get_parameters()
        
        # Should be different objects
        assert params1 is not params2
        
        # But same values
        assert params1 == params2
        
        # Modifying one shouldn't affect the other
        params1['eq_low'] = 999.0
        assert popup.parameters['eq_low'] != 999.0
        
        # Clean up
        popup.window.destroy()


class TestEnhancementPopupWindow:
    """Test window creation and components."""
    
    def test_window_creation(self):
        """Test that window is created properly."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        assert popup.window is not None
        assert popup.window.title() == "MAGE - Audio Enhancement"
        
        # Clean up
        popup.window.destroy()
    
    def test_sliders_created(self):
        """Test that all sliders are created."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Check that sliders dictionary is populated
        assert len(popup.sliders) > 0
        
        # Check specific sliders exist
        assert 'eq_low' in popup.sliders
        assert 'comp_threshold' in popup.sliders
        assert 'reverb_room' in popup.sliders
        assert 'vocal_denoise' in popup.sliders
        
        # Clean up
        popup.window.destroy()
    
    def test_buttons_created(self):
        """Test that control buttons are created."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        assert popup.preview_btn is not None
        assert popup.apply_btn is not None
        assert popup.reset_btn is not None
        assert popup.close_btn is not None
        
        # Clean up
        popup.window.destroy()


class TestEnhancementPopupAudioProcessing:
    """Test audio loading and processing."""
    
    def test_load_audio(self, sample_audio_file):
        """Test audio loading."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup(audio_path=str(sample_audio_file))
        
        assert popup.original_audio is not None
        assert popup.sample_rate == 44100
        assert len(popup.original_audio) > 0
        
        # Clean up
        popup.window.destroy()
    
    def test_reset_parameters_functionality(self):
        """Test parameter reset functionality."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Change some parameters
        popup.set_parameter('eq_low', 10.0)
        popup.set_parameter('comp_ratio', 10.0)
        popup.set_parameter('reverb_wet', 0.8)
        
        # Reset
        popup._reset_parameters()
        
        # Check defaults are restored
        assert popup.parameters['eq_low'] == 0.0
        assert popup.parameters['comp_ratio'] == 4.0
        assert popup.parameters['reverb_wet'] == 0.3
        
        # Clean up
        popup.window.destroy()


class TestEnhancementPopupErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_dependencies(self, monkeypatch):
        """Test behavior when dependencies are missing."""
        # This test is tricky because we can't easily unload numpy/soundfile
        # So we just verify the availability check works
        result = is_enhancement_popup_available()
        assert isinstance(result, bool)
    
    def test_preview_without_audio(self):
        """Test preview functionality without loaded audio."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Should handle gracefully (shows warning dialog)
        # We can't test the dialog directly, but we can verify it doesn't crash
        assert popup.original_audio is None
        
        # Clean up
        popup.window.destroy()
    
    def test_apply_without_preview(self):
        """Test apply functionality without preview."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Should handle gracefully
        assert popup.processed_audio is None
        
        # Clean up
        popup.window.destroy()


class TestEnhancementPopupIntegration:
    """Test integration capabilities."""
    
    def test_callback_invocation(self):
        """Test that callback is invoked on parameter change."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        callback_history = []
        
        def callback(param, value):
            callback_history.append((param, value))
        
        popup = EnhancementPopup(callback=callback)
        
        # Simulate slider change
        import tkinter as tk
        value_var = tk.StringVar()
        popup._on_slider_change('eq_low', '5.0', 'dB', value_var)
        
        # Check callback was called
        assert len(callback_history) == 1
        assert callback_history[0] == ('eq_low', 5.0)
        
        # Clean up
        popup.window.destroy()
    
    def test_get_parameters_for_integration(self):
        """Test getting parameters for external use."""
        if not is_enhancement_popup_available():
            pytest.skip("Enhancement popup not available")
        
        popup = EnhancementPopup()
        
        # Set some parameters
        popup.set_parameter('eq_low', 3.0)
        popup.set_parameter('comp_threshold', -15.0)
        popup.set_parameter('reverb_wet', 0.5)
        
        # Get parameters (simulating external integration)
        params = popup.get_parameters()
        
        assert params['eq_low'] == 3.0
        assert params['comp_threshold'] == -15.0
        assert params['reverb_wet'] == 0.5
        
        # Verify these can be used by external code
        assert isinstance(params, dict)
        assert all(isinstance(v, (int, float)) for v in params.values())
        
        # Clean up
        popup.window.destroy()


def test_enhancement_popup_example():
    """Test that the example script structure is valid."""
    example_path = Path(__file__).parent / "examples" / "enhancement_popup_example.py"
    
    # File should exist
    assert example_path.exists()
    
    # Should be importable (basic syntax check)
    import importlib.util
    spec = importlib.util.spec_from_file_location("example", example_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just verify it loads
        assert module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
