"""Tkinter Enhancement Popup for real-time parameter adjustment.

This module provides a desktop GUI popup window for adjusting audio processing
parameters in real-time with:
- Visual sliders for all effect parameters
- Real-time audio preview
- Integration with main Gradio interface
- Comprehensive error handling and logging
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

try:
    import numpy as np
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    np = None
    sf = None

from mage.utils import MAGELogger
from mage.exceptions import AudioProcessingError, MAGEException

logger = MAGELogger.get_logger(__name__)


class EnhancementPopup:
    """Desktop GUI popup for real-time audio enhancement parameter adjustment.
    
    This popup provides intuitive sliders and controls for adjusting:
    - 3-Band EQ (Low, Mid, High)
    - Compressor (Threshold, Ratio, Attack, Release)
    - Reverb (Room Size, Damping, Wet Level)
    - Limiter (Threshold, Release)
    - Vocal Enhancement (Denoise, Brightness, Warmth, Clarity, Level)
    
    Features:
    - Real-time preview of changes
    - Apply/Reset functionality
    - Integration with Gradio interface
    - Comprehensive error handling
    """
    
    def __init__(self, audio_path: Optional[str] = None, 
                 callback: Optional[Callable] = None):
        """Initialize the enhancement popup.
        
        Args:
            audio_path: Path to audio file to enhance (optional)
            callback: Callback function to notify when parameters change
            
        Raises:
            MAGEException: If initialization fails
        """
        logger.info("Initializing Enhancement Popup")
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available (numpy/soundfile)")
            raise MAGEException("Enhancement popup requires numpy and soundfile")
        
        self.audio_path = Path(audio_path) if audio_path else None
        self.callback = callback
        
        # Audio state
        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None
        
        # Parameter state
        self.parameters = {
            # EQ
            'eq_low': 0.0,
            'eq_mid': 0.0,
            'eq_high': 0.0,
            # Compressor
            'comp_threshold': -20.0,
            'comp_ratio': 4.0,
            'comp_attack': 5.0,
            'comp_release': 100.0,
            # Reverb
            'reverb_room': 0.5,
            'reverb_damping': 0.5,
            'reverb_wet': 0.3,
            # Limiter
            'limiter_threshold': -1.0,
            'limiter_release': 100.0,
            # Vocal Enhancement
            'vocal_denoise': 0.0,
            'vocal_brightness': 0.0,
            'vocal_warmth': 0.0,
            'vocal_clarity': 0.0,
            'vocal_level': 0.0,
        }
        
        # Thread-safe queue for audio processing
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # Lazy-loaded processors
        self._effects_processor = None
        self._vocal_enhancer = None
        
        # Create GUI
        self.window = None
        self.sliders = {}
        self.labels = {}
        
        try:
            self._create_window()
            if self.audio_path and self.audio_path.exists():
                self._load_audio()
            logger.info("Enhancement popup initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhancement popup: {e}")
            raise MAGEException(f"Enhancement popup initialization failed: {e}")
    
    def _create_window(self):
        """Create the main Tkinter window with all controls."""
        try:
            logger.info("Creating enhancement popup window")
            
            self.window = tk.Tk()
            self.window.title("MAGE - Audio Enhancement")
            self.window.geometry("600x800")
            self.window.resizable(False, False)
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            
            # Main container
            main_frame = ttk.Frame(self.window, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Title
            title_label = ttk.Label(
                main_frame,
                text="🎛️ Audio Enhancement Parameters",
                font=('Arial', 16, 'bold')
            )
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
            
            row = 1
            
            # === EQ Section ===
            row = self._create_section(main_frame, "3-Band Equalizer", row)
            row = self._create_slider(main_frame, 'eq_low', "Low (Bass)", -12, 12, 0.5, "dB", row)
            row = self._create_slider(main_frame, 'eq_mid', "Mid", -12, 12, 0.5, "dB", row)
            row = self._create_slider(main_frame, 'eq_high', "High (Treble)", -12, 12, 0.5, "dB", row)
            
            # === Compressor Section ===
            row = self._create_section(main_frame, "Compressor", row)
            row = self._create_slider(main_frame, 'comp_threshold', "Threshold", -60, 0, 1, "dB", row)
            row = self._create_slider(main_frame, 'comp_ratio', "Ratio", 1, 20, 0.5, ":1", row)
            row = self._create_slider(main_frame, 'comp_attack', "Attack", 0.1, 100, 0.1, "ms", row)
            row = self._create_slider(main_frame, 'comp_release', "Release", 10, 1000, 10, "ms", row)
            
            # === Reverb Section ===
            row = self._create_section(main_frame, "Reverb", row)
            row = self._create_slider(main_frame, 'reverb_room', "Room Size", 0, 1, 0.05, "", row)
            row = self._create_slider(main_frame, 'reverb_damping', "Damping", 0, 1, 0.05, "", row)
            row = self._create_slider(main_frame, 'reverb_wet', "Wet Level", 0, 1, 0.05, "", row)
            
            # === Limiter Section ===
            row = self._create_section(main_frame, "Limiter", row)
            row = self._create_slider(main_frame, 'limiter_threshold', "Threshold", -20, 0, 0.5, "dB", row)
            row = self._create_slider(main_frame, 'limiter_release', "Release", 10, 500, 10, "ms", row)
            
            # === Vocal Enhancement Section ===
            row = self._create_section(main_frame, "Vocal Enhancement", row)
            row = self._create_slider(main_frame, 'vocal_denoise', "Denoise", 0, 1, 0.05, "", row)
            row = self._create_slider(main_frame, 'vocal_brightness', "Brightness", -1, 1, 0.1, "", row)
            row = self._create_slider(main_frame, 'vocal_warmth', "Warmth", -1, 1, 0.1, "", row)
            row = self._create_slider(main_frame, 'vocal_clarity', "Clarity", 0, 1, 0.05, "", row)
            row = self._create_slider(main_frame, 'vocal_level', "Level", -12, 12, 0.5, "dB", row)
            
            # === Control Buttons ===
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=row, column=0, columnspan=2, pady=20)
            
            self.preview_btn = ttk.Button(
                button_frame,
                text="🎧 Preview",
                command=self._preview_changes,
                width=12
            )
            self.preview_btn.grid(row=0, column=0, padx=5)
            
            self.apply_btn = ttk.Button(
                button_frame,
                text="✅ Apply",
                command=self._apply_changes,
                width=12
            )
            self.apply_btn.grid(row=0, column=1, padx=5)
            
            self.reset_btn = ttk.Button(
                button_frame,
                text="🔄 Reset",
                command=self._reset_parameters,
                width=12
            )
            self.reset_btn.grid(row=0, column=2, padx=5)
            
            self.close_btn = ttk.Button(
                button_frame,
                text="❌ Close",
                command=self._close_window,
                width=12
            )
            self.close_btn.grid(row=0, column=3, padx=5)
            
            # Status bar
            self.status_var = tk.StringVar(value="Ready")
            status_label = ttk.Label(
                main_frame,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W
            )
            status_label.grid(row=row+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
            
            logger.info("Enhancement popup window created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create window: {e}")
            raise MAGEException(f"Window creation failed: {e}")
    
    def _create_section(self, parent, title: str, row: int) -> int:
        """Create a section header.
        
        Args:
            parent: Parent frame
            title: Section title
            row: Current row number
            
        Returns:
            Next row number
        """
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 5))
        
        label = ttk.Label(parent, text=title, font=('Arial', 12, 'bold'))
        label.grid(row=row+1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        return row + 2
    
    def _create_slider(self, parent, param_name: str, label: str,
                      min_val: float, max_val: float, resolution: float,
                      unit: str, row: int) -> int:
        """Create a parameter slider with label.
        
        Args:
            parent: Parent frame
            param_name: Parameter name in self.parameters
            label: Display label
            min_val: Minimum value
            max_val: Maximum value
            resolution: Step size
            unit: Unit label (e.g., "dB", "ms")
            row: Current row number
            
        Returns:
            Next row number
        """
        # Label frame
        label_frame = ttk.Frame(parent)
        label_frame.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        ttk.Label(label_frame, text=label, width=15).pack(side=tk.LEFT)
        
        value_var = tk.StringVar(value=f"{self.parameters[param_name]:.2f}{unit}")
        self.labels[param_name] = ttk.Label(label_frame, textvariable=value_var, width=10)
        self.labels[param_name].pack(side=tk.LEFT)
        
        # Slider
        slider = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v, p=param_name, u=unit, var=value_var: self._on_slider_change(p, v, u, var)
        )
        slider.set(self.parameters[param_name])
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        self.sliders[param_name] = slider
        
        return row + 1
    
    def _on_slider_change(self, param_name: str, value: str, unit: str, label_var: tk.StringVar):
        """Handle slider value change.
        
        Args:
            param_name: Parameter name
            value: New value (as string from Tkinter)
            unit: Unit label
            label_var: Label StringVar to update
        """
        try:
            float_value = float(value)
            self.parameters[param_name] = float_value
            label_var.set(f"{float_value:.2f}{unit}")
            
            # Notify callback if set
            if self.callback:
                self.callback(param_name, float_value)
                
        except Exception as e:
            logger.error(f"Slider change error for {param_name}: {e}")
    
    def _load_audio(self):
        """Load audio file for processing."""
        try:
            logger.info(f"Loading audio from {self.audio_path}")
            self.status_var.set(f"Loading {self.audio_path.name}...")
            
            self.original_audio, self.sample_rate = sf.read(str(self.audio_path))
            logger.info(f"Audio loaded: {len(self.original_audio)} samples @ {self.sample_rate}Hz")
            
            self.status_var.set(f"Loaded: {self.audio_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            self.status_var.set(f"Error loading audio: {str(e)}")
            messagebox.showerror("Audio Load Error", f"Failed to load audio:\n{str(e)}")
    
    def _get_effects_processor(self):
        """Lazy-load effects processor."""
        if self._effects_processor is None:
            try:
                from mage.processors import EffectsProcessor
                self._effects_processor = EffectsProcessor()
                logger.info("Effects processor loaded")
            except Exception as e:
                logger.error(f"Failed to load effects processor: {e}")
                raise MAGEException(f"Could not load effects processor: {e}")
        return self._effects_processor
    
    def _get_vocal_enhancer(self):
        """Lazy-load vocal enhancer."""
        if self._vocal_enhancer is None:
            try:
                from mage.vocals import VocalEnhancer
                self._vocal_enhancer = VocalEnhancer()
                logger.info("Vocal enhancer loaded")
            except Exception as e:
                logger.error(f"Failed to load vocal enhancer: {e}")
                raise MAGEException(f"Could not load vocal enhancer: {e}")
        return self._vocal_enhancer
    
    def _preview_changes(self):
        """Preview audio with current parameters."""
        try:
            if self.original_audio is None:
                messagebox.showwarning("No Audio", "Please load an audio file first")
                return
            
            logger.info("Previewing audio with current parameters")
            self.status_var.set("Processing preview...")
            
            # Disable buttons during processing
            self.preview_btn.config(state='disabled')
            self.apply_btn.config(state='disabled')
            
            # Process in background thread
            threading.Thread(target=self._process_audio_preview, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            self.status_var.set(f"Preview error: {str(e)}")
            messagebox.showerror("Preview Error", f"Failed to preview:\n{str(e)}")
            self.preview_btn.config(state='normal')
            self.apply_btn.config(state='normal')
    
    def _process_audio_preview(self):
        """Process audio with current parameters (runs in background thread)."""
        try:
            audio = self.original_audio.copy()
            
            # Apply effects in order
            effects = self._get_effects_processor()
            
            # EQ
            if any(self.parameters[k] != 0 for k in ['eq_low', 'eq_mid', 'eq_high']):
                audio = effects.apply_eq(
                    audio,
                    low_shelf_gain_db=self.parameters['eq_low'],
                    mid_gain_db=self.parameters['eq_mid'],
                    high_shelf_gain_db=self.parameters['eq_high']
                )
            
            # Compressor
            if self.parameters['comp_threshold'] != -20.0 or self.parameters['comp_ratio'] != 4.0:
                audio = effects.apply_compressor(
                    audio,
                    threshold_db=self.parameters['comp_threshold'],
                    ratio=self.parameters['comp_ratio'],
                    attack_ms=self.parameters['comp_attack'],
                    release_ms=self.parameters['comp_release']
                )
            
            # Reverb
            if self.parameters['reverb_wet'] > 0:
                audio = effects.apply_reverb(
                    audio,
                    room_size=self.parameters['reverb_room'],
                    damping=self.parameters['reverb_damping'],
                    wet_level=self.parameters['reverb_wet']
                )
            
            # Limiter
            audio = effects.apply_limiter(
                audio,
                threshold_db=self.parameters['limiter_threshold'],
                release_ms=self.parameters['limiter_release']
            )
            
            # Vocal enhancement (if any parameters are non-zero)
            if any(self.parameters[k] != 0 for k in ['vocal_denoise', 'vocal_brightness', 
                                                      'vocal_warmth', 'vocal_clarity', 'vocal_level']):
                vocal_enhancer = self._get_vocal_enhancer()
                audio = vocal_enhancer.enhance(
                    audio,
                    denoise=self.parameters['vocal_denoise'],
                    brightness=self.parameters['vocal_brightness'],
                    warmth=self.parameters['vocal_warmth'],
                    clarity=self.parameters['vocal_clarity'],
                    level_adjust_db=self.parameters['vocal_level']
                )
            
            self.processed_audio = audio
            
            # Save preview to temp file
            preview_path = Path("output") / "temp" / "preview.wav"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(preview_path), audio, self.sample_rate)
            
            logger.info("Preview processing complete")
            self.window.after(0, lambda: self.status_var.set("Preview ready (saved to output/temp/preview.wav)"))
            self.window.after(0, lambda: self.preview_btn.config(state='normal'))
            self.window.after(0, lambda: self.apply_btn.config(state='normal'))
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            self.window.after(0, lambda: self.status_var.set(f"Processing error: {str(e)}"))
            self.window.after(0, lambda: messagebox.showerror("Processing Error", f"Failed to process:\n{str(e)}"))
            self.window.after(0, lambda: self.preview_btn.config(state='normal'))
            self.window.after(0, lambda: self.apply_btn.config(state='normal'))
    
    def _apply_changes(self):
        """Apply current parameters and save."""
        try:
            if self.processed_audio is None:
                messagebox.showwarning("No Preview", "Please preview changes first")
                return
            
            logger.info("Applying changes to audio file")
            self.status_var.set("Saving changes...")
            
            # Save to original path with _enhanced suffix
            output_path = self.audio_path.parent / f"{self.audio_path.stem}_enhanced{self.audio_path.suffix}"
            sf.write(str(output_path), self.processed_audio, self.sample_rate)
            
            logger.info(f"Enhanced audio saved to {output_path}")
            self.status_var.set(f"Saved: {output_path.name}")
            messagebox.showinfo("Success", f"Enhanced audio saved to:\n{output_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            self.status_var.set(f"Save error: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")
    
    def _reset_parameters(self):
        """Reset all parameters to default values."""
        try:
            logger.info("Resetting all parameters to defaults")
            
            # Reset parameter values
            defaults = {
                'eq_low': 0.0, 'eq_mid': 0.0, 'eq_high': 0.0,
                'comp_threshold': -20.0, 'comp_ratio': 4.0,
                'comp_attack': 5.0, 'comp_release': 100.0,
                'reverb_room': 0.5, 'reverb_damping': 0.5, 'reverb_wet': 0.3,
                'limiter_threshold': -1.0, 'limiter_release': 100.0,
                'vocal_denoise': 0.0, 'vocal_brightness': 0.0,
                'vocal_warmth': 0.0, 'vocal_clarity': 0.0, 'vocal_level': 0.0,
            }
            
            for param, value in defaults.items():
                self.parameters[param] = value
                self.sliders[param].set(value)
            
            self.status_var.set("Parameters reset to defaults")
            logger.info("Parameters reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset parameters: {e}")
            messagebox.showerror("Reset Error", f"Failed to reset:\n{str(e)}")
    
    def _close_window(self):
        """Close the popup window."""
        logger.info("Closing enhancement popup")
        self.window.destroy()
    
    def show(self):
        """Show the popup window and start event loop."""
        try:
            logger.info("Showing enhancement popup")
            self.window.mainloop()
        except Exception as e:
            logger.error(f"Window event loop error: {e}")
            raise MAGEException(f"Window display failed: {e}")
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values.
        
        Returns:
            Dictionary of parameter names to values
        """
        return self.parameters.copy()
    
    def set_parameter(self, name: str, value: float):
        """Set a parameter value programmatically.
        
        Args:
            name: Parameter name
            value: New value
            
        Raises:
            ValueError: If parameter name is invalid
        """
        if name not in self.parameters:
            raise ValueError(f"Invalid parameter name: {name}")
        
        self.parameters[name] = value
        if name in self.sliders:
            self.sliders[name].set(value)
        
        logger.debug(f"Parameter set: {name} = {value}")


def is_enhancement_popup_available() -> bool:
    """Check if Tkinter and audio libraries are available.
    
    Returns:
        True if enhancement popup can be used
    """
    try:
        import tkinter
        return AUDIO_AVAILABLE
    except ImportError:
        return False
