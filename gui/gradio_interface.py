"""Gradio web interface for MAGE.

This module provides a comprehensive web-based GUI for music generation,
vocal enhancement, stem separation, and audio processing using Gradio.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import soundfile as sf

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

from mage.utils import MAGELogger
from mage.exceptions import (
    MAGEException,
    AudioProcessingError,
    ModelLoadError,
    ConfigurationError
)
from mage.config import Config

logger = MAGELogger.get_logger(__name__)


def is_gradio_available() -> bool:
    """Check if Gradio is available.
    
    Returns:
        True if Gradio is installed
    """
    return GRADIO_AVAILABLE


class GradioInterface:
    """Gradio web interface for MAGE.
    
    Provides a comprehensive web UI for:
    - Lyrics generation
    - Stem separation
    - Vocal enhancement
    - Audio effects processing
    - Timeline arrangement
    - Audio export
    """
    
    def __init__(self, config_path: str | Path = "config/config.yaml"):
        """Initialize Gradio interface.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ModelLoadError: If Gradio is not available
            ConfigurationError: If config cannot be loaded
        """
        if not GRADIO_AVAILABLE:
            raise ModelLoadError(
                "Gradio is required but not installed",
                details={"install": "pip install gradio>=4.0.0"}
            )
        
        logger.info("Initializing Gradio interface")
        
        try:
            self.config = Config.from_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                details={"config_path": str(config_path)}
            )
        
        # Output directory
        self.output_dir = Path("output/gui")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load modules
        self._lyrics_generator = None
        self._stem_separator = None
        self._vocal_enhancer = None
        self._effects_processor = None
        
        logger.debug("Gradio interface initialized")
    
    def _get_lyrics_generator(self):
        """Get or create lyrics generator."""
        if self._lyrics_generator is None:
            try:
                from mage.lyrics import LyricGenerator
                self._lyrics_generator = LyricGenerator(
                    device=self.config.lyrics.device
                )
                logger.info("Lyrics generator loaded")
            except Exception as e:
                logger.error(f"Failed to load lyrics generator: {e}")
                raise ModelLoadError(f"Failed to load lyrics generator: {e}")
        return self._lyrics_generator
    
    def _get_stem_separator(self):
        """Get or create stem separator."""
        if self._stem_separator is None:
            try:
                from mage.stems import DemucsSeparator
                self._stem_separator = DemucsSeparator(
                    model_name=self.config.stems.model_name,
                    device=self.config.stems.device
                )
                logger.info("Stem separator loaded")
            except Exception as e:
                logger.error(f"Failed to load stem separator: {e}")
                raise ModelLoadError(f"Failed to load stem separator: {e}")
        return self._stem_separator
    
    def _get_vocal_enhancer(self):
        """Get or create vocal enhancer."""
        if self._vocal_enhancer is None:
            try:
                from mage.vocals import VocalEnhancer
                self._vocal_enhancer = VocalEnhancer(
                    sample_rate=self.config.audio.sample_rate,
                    device=self.config.vocals.device
                )
                logger.info("Vocal enhancer loaded")
            except Exception as e:
                logger.error(f"Failed to load vocal enhancer: {e}")
                raise ModelLoadError(f"Failed to load vocal enhancer: {e}")
        return self._vocal_enhancer
    
    def _get_effects_processor(self):
        """Get or create effects processor."""
        if self._effects_processor is None:
            try:
                from mage.processors import EffectsProcessor
                self._effects_processor = EffectsProcessor(
                    sample_rate=self.config.audio.sample_rate
                )
                logger.info("Effects processor loaded")
            except Exception as e:
                logger.error(f"Failed to load effects processor: {e}")
                raise ModelLoadError(f"Failed to load effects processor: {e}")
        return self._effects_processor
    
    def generate_lyrics(
        self,
        genre: str,
        theme: str,
        num_lines: int,
        temperature: float
    ) -> str:
        """Generate lyrics.
        
        Args:
            genre: Music genre
            theme: Lyric theme/topic
            num_lines: Number of lines to generate
            temperature: Generation temperature
            
        Returns:
            Generated lyrics text
        """
        logger.info(f"Generating lyrics: genre={genre}, theme={theme}, lines={num_lines}")
        
        try:
            generator = self._get_lyrics_generator()
            
            prompt = f"{genre} song about {theme}"
            lyrics = generator.generate(
                prompt=prompt,
                max_new_tokens=num_lines * 20,
                temperature=temperature
            )
            
            logger.info(f"Generated {len(lyrics)} characters of lyrics")
            return lyrics
            
        except Exception as e:
            error_msg = f"Failed to generate lyrics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"
    
    def separate_stems(
        self,
        audio_file: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        """Separate audio into stems.
        
        Args:
            audio_file: Path to audio file
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (vocals_path, drums_path, bass_path, other_path, status_message)
        """
        if not audio_file:
            return None, None, None, None, "Please upload an audio file"
        
        logger.info(f"Separating stems from: {audio_file}")
        progress(0, desc="Loading audio...")
        
        try:
            separator = self._get_stem_separator()
            
            progress(0.2, desc="Separating stems...")
            stems = separator.separate(audio_file, output_dir=self.output_dir / "stems")
            
            # Save individual stems
            progress(0.6, desc="Saving stems...")
            vocals_path = str(self.output_dir / "stems" / "vocals.wav")
            drums_path = str(self.output_dir / "stems" / "drums.wav")
            bass_path = str(self.output_dir / "stems" / "bass.wav")
            other_path = str(self.output_dir / "stems" / "other.wav")
            
            if stems.vocals is not None:
                sf.write(vocals_path, stems.vocals.T, stems.sample_rate)
            if stems.drums is not None:
                sf.write(drums_path, stems.drums.T, stems.sample_rate)
            if stems.bass is not None:
                sf.write(bass_path, stems.bass.T, stems.sample_rate)
            if stems.other is not None:
                sf.write(other_path, stems.other.T, stems.sample_rate)
            
            progress(1.0, desc="Complete!")
            
            status = f"Stems separated successfully! Available: {', '.join([s.value for s in stems.available_stems()])}"
            logger.info(status)
            
            return vocals_path, drums_path, bass_path, other_path, f"✅ {status}"
            
        except Exception as e:
            error_msg = f"Failed to separate stems: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, None, None, f"Error: {error_msg}"
    
    def enhance_vocals(
        self,
        audio_file: str,
        denoise_strength: float,
        brightness: float,
        warmth: float,
        clarity: float,
        target_level: float,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """Enhance vocal audio.
        
        Args:
            audio_file: Path to audio file
            denoise_strength: Noise reduction strength (0-1)
            brightness: High frequency boost (-1 to 1)
            warmth: Low-mid frequency boost (-1 to 1)
            clarity: Mid-high frequency boost (-1 to 1)
            target_level: Target RMS level in dB
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (enhanced_audio_path, status_message)
        """
        if not audio_file:
            return None, "Please upload an audio file"
        
        logger.info(f"Enhancing vocals from: {audio_file}")
        progress(0, desc="Loading audio...")
        
        try:
            # Load audio
            audio, sr = sf.read(audio_file)
            if len(audio.shape) == 1:
                # Mono
                pass
            else:
                # Stereo - transpose to (channels, samples)
                audio = audio.T
            
            progress(0.3, desc="Enhancing vocals...")
            enhancer = self._get_vocal_enhancer()
            
            enhanced = enhancer.enhance(
                audio,
                denoise_strength=denoise_strength,
                brightness=brightness,
                warmth=warmth,
                clarity=clarity,
                target_level=target_level
            )
            
            progress(0.8, desc="Saving enhanced audio...")
            output_path = str(self.output_dir / "enhanced_vocals.wav")
            
            if len(enhanced.shape) == 1:
                sf.write(output_path, enhanced, sr)
            else:
                sf.write(output_path, enhanced.T, sr)
            
            progress(1.0, desc="Complete!")
            
            status = f"Vocals enhanced successfully! Saved to: {output_path}"
            logger.info(status)
            
            return output_path, f"✅ {status}"
            
        except Exception as e:
            error_msg = f"Failed to enhance vocals: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, f"Error: {error_msg}"
    
    def apply_effects(
        self,
        audio_file: str,
        effect_type: str,
        param1: float,
        param2: float,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], str]:
        """Apply audio effects.
        
        Args:
            audio_file: Path to audio file
            effect_type: Type of effect to apply
            param1: First effect parameter
            param2: Second effect parameter
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (processed_audio_path, status_message)
        """
        if not audio_file:
            return None, "Please upload an audio file"
        
        logger.info(f"Applying {effect_type} effect to: {audio_file}")
        progress(0, desc="Loading audio...")
        
        try:
            # Load audio
            audio, sr = sf.read(audio_file)
            if len(audio.shape) > 1:
                audio = audio.T
            
            progress(0.3, desc=f"Applying {effect_type}...")
            processor = self._get_effects_processor()
            
            # Apply effect based on type
            if effect_type == "EQ":
                processed = processor.apply_eq(
                    audio,
                    low_shelf_gain_db=param1,
                    high_shelf_gain_db=param2
                )
            elif effect_type == "Compressor":
                processed = processor.apply_compressor(
                    audio,
                    threshold_db=param1,
                    ratio=param2
                )
            elif effect_type == "Reverb":
                processed = processor.apply_reverb(
                    audio,
                    room_size=param1,
                    wet_level=param2
                )
            elif effect_type == "Limiter":
                processed = processor.apply_limiter(
                    audio,
                    threshold_db=param1
                )
            else:
                return None, f"Unknown effect type: {effect_type}"
            
            progress(0.8, desc="Saving processed audio...")
            output_path = str(self.output_dir / f"processed_{effect_type.lower()}.wav")
            
            if len(processed.shape) == 1:
                sf.write(output_path, processed, sr)
            else:
                sf.write(output_path, processed.T, sr)
            
            progress(1.0, desc="Complete!")
            
            status = f"{effect_type} applied successfully! Saved to: {output_path}"
            logger.info(status)
            
            return output_path, f"✅ {status}"
            
        except Exception as e:
            error_msg = f"Failed to apply effect: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, f"Error: {error_msg}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        logger.info("Creating Gradio interface")
        
        with gr.Blocks(title="MAGE - Mixed Audio Generation Engine", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🎵 MAGE - Mixed Audio Generation Engine
                
                A comprehensive AI-based music generation and audio processing system.
                """
            )
            
            with gr.Tabs():
                # Tab 1: Lyrics Generation
                with gr.Tab("📝 Lyrics Generation"):
                    gr.Markdown("### Generate song lyrics using AI")
                    
                    with gr.Row():
                        with gr.Column():
                            lyrics_genre = gr.Dropdown(
                                choices=["Pop", "Rock", "Hip-Hop", "Country", "Jazz", "Classical", "Electronic"],
                                value="Pop",
                                label="Genre"
                            )
                            lyrics_theme = gr.Textbox(
                                label="Theme/Topic",
                                placeholder="love, adventure, freedom, etc.",
                                value="love"
                            )
                            lyrics_lines = gr.Slider(
                                minimum=4,
                                maximum=32,
                                value=16,
                                step=4,
                                label="Number of Lines"
                            )
                            lyrics_temp = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.8,
                                step=0.1,
                                label="Temperature (creativity)"
                            )
                            lyrics_btn = gr.Button("🎤 Generate Lyrics", variant="primary")
                        
                        with gr.Column():
                            lyrics_output = gr.Textbox(
                                label="Generated Lyrics",
                                lines=20,
                                placeholder="Generated lyrics will appear here..."
                            )
                    
                    lyrics_btn.click(
                        fn=self.generate_lyrics,
                        inputs=[lyrics_genre, lyrics_theme, lyrics_lines, lyrics_temp],
                        outputs=lyrics_output
                    )
                
                # Tab 2: Stem Separation
                with gr.Tab("🎼 Stem Separation"):
                    gr.Markdown("### Separate audio into individual stems (vocals, drums, bass, other)")
                    
                    with gr.Row():
                        with gr.Column():
                            stem_input = gr.Audio(
                                label="Upload Audio File",
                                type="filepath"
                            )
                            stem_btn = gr.Button("🔀 Separate Stems", variant="primary")
                        
                        with gr.Column():
                            stem_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Row():
                        stem_vocals = gr.Audio(label="Vocals", type="filepath")
                        stem_drums = gr.Audio(label="Drums", type="filepath")
                    
                    with gr.Row():
                        stem_bass = gr.Audio(label="Bass", type="filepath")
                        stem_other = gr.Audio(label="Other", type="filepath")
                    
                    stem_btn.click(
                        fn=self.separate_stems,
                        inputs=stem_input,
                        outputs=[stem_vocals, stem_drums, stem_bass, stem_other, stem_status]
                    )
                
                # Tab 3: Vocal Enhancement
                with gr.Tab("🎙️ Vocal Enhancement"):
                    gr.Markdown("### Enhance vocal quality with AI-powered processing")
                    
                    with gr.Row():
                        with gr.Column():
                            vocal_input = gr.Audio(
                                label="Upload Vocal Audio",
                                type="filepath"
                            )
                            
                            gr.Markdown("#### Enhancement Settings")
                            vocal_denoise = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.05,
                                label="Denoise Strength"
                            )
                            vocal_brightness = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.2,
                                step=0.1,
                                label="Brightness (high frequencies)"
                            )
                            vocal_warmth = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.1,
                                step=0.1,
                                label="Warmth (low-mid frequencies)"
                            )
                            vocal_clarity = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                label="Clarity (mid-high frequencies)"
                            )
                            vocal_target = gr.Slider(
                                minimum=-40.0,
                                maximum=0.0,
                                value=-18.0,
                                step=1.0,
                                label="Target Level (dB)"
                            )
                            
                            vocal_btn = gr.Button("✨ Enhance Vocals", variant="primary")
                        
                        with gr.Column():
                            vocal_output = gr.Audio(label="Enhanced Vocals", type="filepath")
                            vocal_status = gr.Textbox(label="Status", lines=3)
                    
                    vocal_btn.click(
                        fn=self.enhance_vocals,
                        inputs=[
                            vocal_input,
                            vocal_denoise,
                            vocal_brightness,
                            vocal_warmth,
                            vocal_clarity,
                            vocal_target
                        ],
                        outputs=[vocal_output, vocal_status]
                    )
                
                # Tab 4: Audio Effects
                with gr.Tab("🎚️ Audio Effects"):
                    gr.Markdown("### Apply professional audio effects")
                    
                    with gr.Row():
                        with gr.Column():
                            effect_input = gr.Audio(
                                label="Upload Audio File",
                                type="filepath"
                            )
                            
                            effect_type = gr.Dropdown(
                                choices=["EQ", "Compressor", "Reverb", "Limiter"],
                                value="EQ",
                                label="Effect Type"
                            )
                            
                            effect_param1 = gr.Slider(
                                minimum=-24.0,
                                maximum=24.0,
                                value=0.0,
                                step=0.5,
                                label="Parameter 1 (varies by effect)"
                            )
                            
                            effect_param2 = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=1.0,
                                step=0.1,
                                label="Parameter 2 (varies by effect)"
                            )
                            
                            gr.Markdown(
                                """
                                **Parameter Guide:**
                                - **EQ**: Param1=Low Shelf Gain, Param2=High Shelf Gain
                                - **Compressor**: Param1=Threshold (dB), Param2=Ratio
                                - **Reverb**: Param1=Room Size (0-1), Param2=Wet Level (0-1)
                                - **Limiter**: Param1=Threshold (dB), Param2=unused
                                """
                            )
                            
                            effect_btn = gr.Button("🎛️ Apply Effect", variant="primary")
                        
                        with gr.Column():
                            effect_output = gr.Audio(label="Processed Audio", type="filepath")
                            effect_status = gr.Textbox(label="Status", lines=3)
                    
                    effect_btn.click(
                        fn=self.apply_effects,
                        inputs=[effect_input, effect_type, effect_param1, effect_param2],
                        outputs=[effect_output, effect_status]
                    )
                
                # Tab 5: About
                with gr.Tab("ℹ️ About"):
                    gr.Markdown(
                        """
                        ## About MAGE
                        
                        **MAGE (Mixed Audio Generation Engine)** is a comprehensive AI-based music generation
                        and audio processing system.
                        
                        ### Features
                        
                        - **Lyrics Generation**: AI-powered lyrics generation using LyricMind-AI
                        - **Stem Separation**: High-quality stem separation using Demucs
                        - **Vocal Enhancement**: AI-based vocal quality improvement with FullSubNet
                        - **Audio Effects**: Professional effects using Spotify's Pedalboard
                        - **Timeline Arrangement**: Multi-track composition and mixing
                        
                        ### System Information
                        
                        - Sample Rate: {} Hz
                        - GPU Device: {}
                        - Output Directory: {}
                        
                        ### Version
                        
                        MAGE v0.1.0
                        
                        ### Credits
                        
                        Built with: PyTorch, Gradio, Demucs, Pedalboard, LibROSA
                        """.format(
                            self.config.audio.sample_rate,
                            self.config.model.device,
                            self.output_dir
                        )
                    )
        
        logger.info("Gradio interface created successfully")
        return interface
    
    def launch(
        self,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False
    ) -> None:
        """Launch the Gradio interface.
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Create public share link
            debug: Enable debug mode
            
        Raises:
            MAGEException: If launch fails
        """
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        try:
            interface = self.create_interface()
            
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=share,
                debug=debug,
                show_error=True
            )
            
        except Exception as e:
            error_msg = f"Failed to launch Gradio interface: {e}"
            logger.error(error_msg, exc_info=True)
            raise MAGEException(error_msg, details={"error": str(e)})
