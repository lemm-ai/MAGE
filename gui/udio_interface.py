"""Udio-style Gradio interface for MAGE.

This module provides a professional DAW-like web interface with:
- Timeline-based workflow with waveform display
- Clip library management
- Advanced controls for effects and vocal enhancement
- Intelligent clip merging and positioning
"""

import os
import logging
import json
import uuid
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

import numpy as np
import soundfile as sf
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from mage.config import Config
from mage.exceptions import AudioProcessingError, ModelLoadError, MAGEException
from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


class Clip:
    """Represents an audio clip in the library and timeline."""
    
    def __init__(self, clip_id: str, name: str, filepath: str, duration: float, 
                 sample_rate: int, position: float = 0.0):
        """Initialize a clip.
        
        Args:
            clip_id: Unique identifier for the clip
            name: Display name for the clip
            filepath: Path to the audio file
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            position: Position in timeline (seconds)
        """
        self.id = clip_id
        self.name = name
        self.filepath = filepath
        self.duration = duration
        self.sample_rate = sample_rate
        self.position = position
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert clip to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'filepath': self.filepath,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'position': self.position
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Clip':
        """Create clip from dictionary."""
        return cls(
            clip_id=data['id'],
            name=data['name'],
            filepath=data['filepath'],
            duration=data['duration'],
            sample_rate=data['sample_rate'],
            position=data.get('position', 0.0)
        )


class UdioInterface:
    """Udio-style web interface for MAGE."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Udio interface.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Udio-style interface")
        
        # Load configuration
        if config_path:
            self.config = Config.from_file(config_path)
        else:
            self.config = Config()
        logger.info(f"Loaded configuration from {config_path or 'default'}")
        
        # Set up paths
        self.output_dir = Path('output') / 'udio'
        self.clips_dir = self.output_dir / 'clips'
        self.timeline_dir = self.output_dir / 'timeline'
        self.library_file = self.output_dir / 'library.json'
        
        # Create directories
        for directory in [self.output_dir, self.clips_dir, self.timeline_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.clip_library: Dict[str, Clip] = {}
        self.timeline_clips: List[str] = []  # List of clip IDs in timeline order
        
        # Lazy-loaded modules
        self._lyrics_generator = None
        self._stem_separator = None
        self._vocal_enhancer = None
        self._effects_processor = None
        self._music_generator = None  # Placeholder for future MusicControlNet
        
        # Load library
        self._load_library()
        
        logger.info(f"Udio interface initialized (output: {self.output_dir})")
    
    def _load_library(self):
        """Load clip library from disk."""
        try:
            if self.library_file.exists():
                with open(self.library_file, 'r') as f:
                    data = json.load(f)
                    self.clip_library = {
                        cid: Clip.from_dict(cdata) 
                        for cid, cdata in data.get('clips', {}).items()
                    }
                    self.timeline_clips = data.get('timeline', [])
                logger.info(f"Loaded {len(self.clip_library)} clips from library")
        except Exception as e:
            logger.error(f"Failed to load library: {e}")
            self.clip_library = {}
            self.timeline_clips = []
    
    def _save_library(self):
        """Save clip library to disk."""
        try:
            data = {
                'clips': {cid: clip.to_dict() for cid, clip in self.clip_library.items()},
                'timeline': self.timeline_clips
            }
            with open(self.library_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Library saved successfully")
        except Exception as e:
            logger.error(f"Failed to save library: {e}")
    
    def _get_lyrics_generator(self):
        """Lazy-load lyrics generator."""
        if self._lyrics_generator is None:
            try:
                from mage.lyrics import LyricGenerator
                self._lyrics_generator = LyricGenerator()
                logger.info("Lyrics generator loaded")
            except Exception as e:
                logger.error(f"Failed to load lyrics generator: {e}")
                raise ModelLoadError(f"Could not load lyrics generator: {e}")
        return self._lyrics_generator
    
    def _get_stem_separator(self):
        """Lazy-load stem separator."""
        if self._stem_separator is None:
            try:
                from mage.stems import DemucsSeparator
                self._stem_separator = DemucsSeparator()
                logger.info("Stem separator loaded")
            except Exception as e:
                logger.error(f"Failed to load stem separator: {e}")
                raise ModelLoadError(f"Could not load stem separator: {e}")
        return self._stem_separator
    
    def _get_vocal_enhancer(self):
        """Lazy-load vocal enhancer."""
        if self._vocal_enhancer is None:
            try:
                from mage.vocals import VocalEnhancer
                self._vocal_enhancer = VocalEnhancer()
                logger.info("Vocal enhancer loaded")
            except Exception as e:
                logger.error(f"Failed to load vocal enhancer: {e}")
                raise ModelLoadError(f"Could not load vocal enhancer: {e}")
        return self._vocal_enhancer
    
    def _get_effects_processor(self):
        """Lazy-load effects processor."""
        if self._effects_processor is None:
            try:
                from mage.processors import EffectsProcessor
                self._effects_processor = EffectsProcessor()
                logger.info("Effects processor loaded")
            except Exception as e:
                logger.error(f"Failed to load effects processor: {e}")
                raise ModelLoadError(f"Could not load effects processor: {e}")
        return self._effects_processor
    
    def generate_lyrics(self, prompt: str, lines: int) -> str:
        """Generate lyrics with AI from prompt.
        
        Args:
            prompt: Description of desired lyrics (genre/theme/mood extracted from this)
            lines: Number of lines to generate
            
        Returns:
            Generated lyrics text
        """
        try:
            logger.info(f"Generating lyrics from prompt: '{prompt}', lines={lines}")
            generator = self._get_lyrics_generator()
            
            # ACE-Step and LyricsMindAI can interpret genre/theme/mood from prompt
            lyrics = generator.generate(
                prompt=prompt,
                max_length=lines * 20  # Approximate tokens per line
            )
            
            logger.info("Lyrics generated successfully")
            return lyrics
            
        except Exception as e:
            logger.error(f"Failed to generate lyrics: {e}")
            return f"Error: Failed to generate lyrics: {str(e)}"
    
    def generate_clip(self, prompt: str, lyrics: str, bpm: int, 
                     position: str, context_length: float,
                     progress=None) -> Tuple[str, str, str, str]:
        """Generate a new audio clip and add to library/timeline.
        
        Args:
            prompt: Text description (genre/mood/style interpreted from this)
            lyrics: Lyrics to use (optional)
            bpm: Beats per minute
            position: Where to place clip (Intro/Previous/Next/Outro)
            context_length: Seconds of context for MusicControlNet
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status_message, library_html, timeline_html, waveform_html)
        """
        try:
            logger.info(f"Generating clip: prompt='{prompt}', bpm={bpm}, position={position}")
            if progress:
                progress(0, desc="Initializing generation...")
            
            # TODO: Replace with actual MusicControlNet generation
            # ACE-Step and MusicControlNet can interpret genre/mood/style from prompt
            if progress:
                progress(0.2, desc="Loading AI models...")
            
            import time
            time.sleep(1.0)  # Simulate model loading
            
            if progress:
                progress(0.3, desc="Generating audio (10-30 seconds)...")
            
            time.sleep(2.0)  # Simulate AI generation
            
            # Create placeholder audio (8 seconds of more complex audio based on BPM)
            duration = 8.0  # Longer clips
            sample_rate = self.config.audio.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = bpm / 60.0 * 2  # Simple frequency based on BPM
            
            # Create more musical placeholder (will be replaced by real AI)
            audio = np.sin(2 * np.pi * frequency * t) * 0.3  # Base tone
            audio += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.2  # Fifth
            audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.15  # Octave
            
            # Extract mood hint from prompt for audio variation
            prompt_lower = prompt.lower()
            if 'energetic' in prompt_lower or 'upbeat' in prompt_lower or 'fast' in prompt_lower:
                audio += np.sin(2 * np.pi * frequency * 3 * t) * 0.15
            elif 'calm' in prompt_lower or 'slow' in prompt_lower or 'gentle' in prompt_lower:
                audio *= 0.6
            
            # Add some variation
            audio += np.random.normal(0, 0.02, len(audio))
            
            if progress:
                progress(0.6, desc="Saving clip...")
            
            # Create clip
            clip_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Extract genre hint from prompt for naming
            clip_name = f"clip_{timestamp}"
            if any(word in prompt_lower for word in ['rock', 'pop', 'jazz', 'electronic', 'classical']):
                for genre in ['rock', 'pop', 'jazz', 'electronic', 'classical']:
                    if genre in prompt_lower:
                        clip_name = f"{genre}_{timestamp}"
                        break
            
            clip_path = self.clips_dir / f"{clip_id}.wav"
            
            # Save audio
            sf.write(str(clip_path), audio, sample_rate)
            
            # Create clip object
            clip = Clip(
                clip_id=clip_id,
                name=clip_name,
                filepath=str(clip_path),
                duration=duration,
                sample_rate=sample_rate
            )
            
            # Add to library
            self.clip_library[clip_id] = clip
            
            if progress:
                progress(0.8, desc="Adding to timeline...")
            
            # Add to timeline based on position
            self._add_clip_to_timeline(clip_id, position)
            
            # Save library
            self._save_library()
            
            if progress:
                progress(1.0, desc="Complete!")
            
            logger.info(f"Clip generated and added: {clip_name} at {clip_path}")
            
            # Return updated UI components
            library_html = self._render_library()
            timeline_html, waveform_html = self._render_timeline()
            
            success_msg = f"✅ Generated: {clip_name}\n📁 File: {clip_path.name}\n⏱️ Duration: {duration:.1f}s @ {bpm} BPM"
            return (success_msg, library_html, timeline_html, waveform_html)
            
        except Exception as e:
            logger.error(f"Failed to generate clip: {e}")
            timeline_html, waveform_html = self._render_timeline()
            return (
                f"Error: {str(e)}",
                self._render_library(),
                timeline_html,
                waveform_html
            )
    
    def _add_clip_to_timeline(self, clip_id: str, position: str):
        """Add clip to timeline at specified position.
        
        Args:
            clip_id: Clip ID to add
            position: Where to add (Intro/Previous/Next/Outro)
        """
        if position == "Intro":
            self.timeline_clips.insert(0, clip_id)
            logger.info(f"Added clip {clip_id} at start of timeline")
        elif position == "Previous":
            if self.timeline_clips:
                insert_pos = max(0, len(self.timeline_clips) - 1)
                self.timeline_clips.insert(insert_pos, clip_id)
            else:
                self.timeline_clips.append(clip_id)
            logger.info(f"Added clip {clip_id} before last clip")
        elif position == "Next":
            self.timeline_clips.append(clip_id)
            logger.info(f"Added clip {clip_id} after last clip")
        elif position == "Outro":
            self.timeline_clips.append(clip_id)
            logger.info(f"Added clip {clip_id} at end of timeline")
    
    def _merge_timeline(self) -> Optional[str]:
        """Merge all clips on timeline into a single audio file.
        
        Returns:
            Path to merged file, or None if failed
        """
        try:
            if len(self.timeline_clips) < 2:
                return None
            
            logger.info(f"Merging {len(self.timeline_clips)} clips on timeline")
            
            # Load all clips
            audio_segments = []
            sample_rate = None
            
            for clip_id in self.timeline_clips:
                if clip_id not in self.clip_library:
                    logger.warning(f"Clip {clip_id} not found in library, skipping")
                    continue
                    
                clip = self.clip_library[clip_id]
                audio, sr = sf.read(clip.filepath)
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
                    # TODO: Resample if needed
                
                audio_segments.append(audio)
            
            if not audio_segments:
                logger.error("No audio segments to merge")
                return None
            
            # Concatenate audio
            merged_audio = np.concatenate(audio_segments)
            
            # Save merged file to temp directory (NOT to library)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_path = self.timeline_dir / f"merged_{timestamp}.wav"
            sf.write(str(merged_path), merged_audio, sample_rate)
            
            logger.info(f"Timeline merged successfully: {merged_path}")
            logger.info("Merged clip is temporary and not added to library")
            return str(merged_path)
            
        except Exception as e:
            logger.error(f"Failed to merge timeline: {e}")
            return None
    
    def _render_library(self) -> str:
        """Render clip library as HTML.
        
        Returns:
            HTML string for clip library
        """
        if not self.clip_library:
            return "<p style='color: #888; text-align: center; padding: 20px;'>No clips in library</p>"
        
        html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
        
        for clip_id, clip in self.clip_library.items():
            in_timeline = "✓" if clip_id in self.timeline_clips else ""
            html += f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #f9f9f9;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <strong>{clip.name}</strong> {in_timeline}
                        <br><small>{clip.duration:.1f}s @ {clip.sample_rate}Hz</small>
                    </div>
                    <div style='display: flex; gap: 5px;'>
                        <button onclick='deleteClip("{clip_id}")' title='Delete'>🗑️</button>
                        <button onclick='renameClip("{clip_id}")' title='Rename'>✏️</button>
                        <button onclick='downloadClip("{clip_id}")' title='Download'>⬇️</button>
                        <button onclick='extendClip("{clip_id}")' title='Extend'>↗️</button>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _render_timeline(self) -> Tuple[str, str]:
        """Render timeline and waveforms as separate HTML components.
        
        Returns:
            Tuple of (timeline_html, waveform_html) for DAW-like stacked display
        """
        if not self.timeline_clips:
            empty_msg = "<p style='color: #888; text-align: center; padding: 20px;'>No clips on timeline</p>"
            return (empty_msg, empty_msg)
        
        try:
            # ===== Timeline Track View (DAW-style) =====
            timeline_html = "<div style='background: #1a1a1a; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>"
            timeline_html += "<div style='color: #fff; font-weight: bold; margin-bottom: 10px;'>Timeline</div>"
            timeline_html += "<div style='display: flex; flex-direction: row; align-items: center; background: #2a2a2a; padding: 10px; border-radius: 3px; min-height: 60px;'>"
            
            # Calculate total duration for scaling
            total_duration = 0.0
            for clip_id in self.timeline_clips:
                if clip_id in self.clip_library:
                    total_duration += self.clip_library[clip_id].duration
            
            if total_duration == 0:
                total_duration = 1.0  # Avoid division by zero
            
            # Render clips as blocks
            colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#6366f1']
            current_pos = 0.0
            
            for i, clip_id in enumerate(self.timeline_clips):
                if clip_id not in self.clip_library:
                    continue
                    
                clip = self.clip_library[clip_id]
                width_percent = (clip.duration / total_duration) * 100
                color = colors[i % len(colors)]
                
                timeline_html += f"""
                <div style='
                    flex: 0 0 {width_percent}%;
                    background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                    height: 40px;
                    margin: 0 2px;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 11px;
                    font-weight: 600;
                    text-overflow: ellipsis;
                    overflow: hidden;
                    white-space: nowrap;
                    padding: 0 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                '>
                    {clip.name[:20]}
                </div>
                """
                current_pos += clip.duration
            
            timeline_html += "</div>"
            
            # Add time ruler
            timeline_html += "<div style='display: flex; justify-content: space-between; margin-top: 5px; color: #888; font-size: 10px;'>"
            num_markers = 5
            for i in range(num_markers + 1):
                time_sec = (total_duration / num_markers) * i
                timeline_html += f"<span>{time_sec:.1f}s</span>"
            timeline_html += "</div></div>"
            
            # ===== Waveform Display =====
            fig, ax = plt.subplots(figsize=(14, 4), facecolor='#1a1a1a')
            ax.set_facecolor('#2a2a2a')
            
            current_pos = 0.0
            
            for i, clip_id in enumerate(self.timeline_clips):
                if clip_id not in self.clip_library:
                    continue
                    
                clip = self.clip_library[clip_id]
                
                # Load audio for waveform
                audio, sr = sf.read(clip.filepath)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                
                # Downsample for visualization
                downsample_factor = max(1, len(audio) // 2000)
                audio_viz = audio[::downsample_factor]
                
                # Time array
                time = np.linspace(current_pos, current_pos + clip.duration, len(audio_viz))
                
                # Plot waveform
                color = colors[i % len(colors)]
                ax.fill_between(time, audio_viz, alpha=0.7, color=color, linewidth=0)
                ax.plot(time, audio_viz, linewidth=0.8, color=color, alpha=0.9)
                
                current_pos += clip.duration
            
            ax.set_xlabel('Time (seconds)', color='#888', fontsize=10)
            ax.set_ylabel('Amplitude', color='#888', fontsize=10)
            ax.tick_params(colors='#888', labelsize=9)
            ax.grid(True, alpha=0.15, color='#666')
            ax.spines['bottom'].set_color('#666')
            ax.spines['top'].set_color('#666')
            ax.spines['left'].set_color('#666')
            ax.spines['right'].set_color('#666')
            
            # Save to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            
            waveform_html = f"""
            <div style='background: #1a1a1a; padding: 15px; border-radius: 5px;'>
                <div style='color: #fff; font-weight: bold; margin-bottom: 10px;'>Waveform Display</div>
                <img src='data:image/png;base64,{img_base64}' style='width: 100%; height: auto; border-radius: 4px;'>
            </div>
            """
            
            return (timeline_html, waveform_html)
            
        except Exception as e:
            logger.error(f"Failed to render timeline/waveform: {e}")
            error_msg = f"<p style='color: red;'>Error rendering: {e}</p>"
            return (error_msg, error_msg)
    
    def delete_clip(self, clip_id: str) -> Tuple[str, str, str]:
        """Delete a clip from library.
        
        Args:
            clip_id: Clip ID to delete
            
        Returns:
            Tuple of (library_html, timeline_html, waveform_html)
        """
        try:
            if clip_id not in self.clip_library:
                logger.warning(f"Clip {clip_id} not found")
                timeline_html, waveform_html = self._render_timeline()
                return self._render_library(), timeline_html, waveform_html
            
            clip = self.clip_library[clip_id]
            
            # Remove from timeline
            if clip_id in self.timeline_clips:
                self.timeline_clips.remove(clip_id)
            
            # Delete file
            if os.path.exists(clip.filepath):
                os.remove(clip.filepath)
            
            # Remove from library
            del self.clip_library[clip_id]
            
            # Save library
            self._save_library()
            
            logger.info(f"Deleted clip: {clip.name}")
            
            timeline_html, waveform_html = self._render_timeline()
            return self._render_library(), timeline_html, waveform_html
            
        except Exception as e:
            logger.error(f"Failed to delete clip: {e}")
            timeline_html, waveform_html = self._render_timeline()
            return self._render_library(), timeline_html, waveform_html
    
    def extend_clip(self, clip_id: str) -> Tuple[str, str, str]:
        """Extend mode: Clear timeline and add clip at start.
        
        Args:
            clip_id: Clip ID to extend from
            
        Returns:
            Tuple of (library_html, timeline_html, waveform_html)
        """
        try:
            if clip_id not in self.clip_library:
                logger.warning(f"Clip {clip_id} not found")
                timeline_html, waveform_html = self._render_timeline()
                return self._render_library(), timeline_html, waveform_html
            
            # Clear timeline and add clip
            self.timeline_clips = [clip_id]
            self._save_library()
            
            logger.info(f"Extended from clip: {self.clip_library[clip_id].name}")
            
            timeline_html, waveform_html = self._render_timeline()
            return self._render_library(), timeline_html, waveform_html
            
        except Exception as e:
            logger.error(f"Failed to extend clip: {e}")
            timeline_html, waveform_html = self._render_timeline()
            return self._render_library(), timeline_html, waveform_html
    
    def apply_advanced_effects(self, eq_low: float, eq_mid: float, eq_high: float,
                              compressor_threshold: float, compressor_ratio: float,
                              reverb_room_size: float, reverb_damping: float,
                              limiter_threshold: float, limiter_release: float,
                              progress=gr.Progress()) -> str:
        """Apply advanced DAW-like effects to merged timeline.
        
        Args:
            eq_low: Low frequency gain (-12 to +12 dB)
            eq_mid: Mid frequency gain (-12 to +12 dB)
            eq_high: High frequency gain (-12 to +12 dB)
            compressor_threshold: Threshold in dB (-60 to 0)
            compressor_ratio: Compression ratio (1:1 to 20:1)
            reverb_room_size: Room size (0.0 to 1.0)
            reverb_damping: Damping (0.0 to 1.0)
            limiter_threshold: Limiter threshold in dB (-20 to 0)
            limiter_release: Release time in ms (10 to 1000)
            progress: Gradio progress tracker
            
        Returns:
            Status message
        """
        try:
            if not self.timeline_clips:
                return "Error: No clips on timeline to process"
            
            logger.info("Applying advanced effects to timeline")
            if progress:
                progress(0, desc="Loading effects processor...")
            
            processor = self._get_effects_processor()
            
            # Get merged timeline audio
            if len(self.timeline_clips) == 1:
                clip_id = self.timeline_clips[0]
                audio_path = self.clip_library[clip_id].filepath
            else:
                if progress:
                    progress(0.2, desc="Merging timeline...")
                audio_path = self._merge_timeline()
                if not audio_path:
                    return "Error: Failed to merge timeline"
            
            # Load audio
            if progress:
                progress(0.4, desc="Loading audio...")
            audio, sr = sf.read(audio_path)
            
            # Apply EQ
            if progress:
                progress(0.5, desc="Applying EQ...")
            audio = processor.apply_eq(
                audio,
                low_shelf_gain_db=eq_low,
                mid_gain_db=eq_mid,
                high_shelf_gain_db=eq_high
            )
            
            # Apply compressor
            if progress:
                progress(0.6, desc="Applying compressor...")
            audio = processor.apply_compressor(
                audio,
                threshold_db=compressor_threshold,
                ratio=compressor_ratio
            )
            
            # Apply reverb
            if progress:
                progress(0.7, desc="Applying reverb...")
            audio = processor.apply_reverb(
                audio,
                room_size=reverb_room_size,
                damping=reverb_damping
            )
            
            # Apply limiter
            if progress:
                progress(0.85, desc="Applying limiter...")
            audio = processor.apply_limiter(
                audio,
                threshold_db=limiter_threshold,
                release_ms=limiter_release
            )
            
            # Save processed audio
            if progress:
                progress(0.95, desc="Saving...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.timeline_dir / f"mastered_{timestamp}.wav"
            sf.write(str(output_path), audio, sr)
            
            if progress:
                progress(1.0, desc="Complete!")
            
            logger.info(f"Effects applied successfully: {output_path}")
            return f"Effects applied successfully! Saved to: {output_path}"
            
        except Exception as e:
            logger.error(f"Failed to apply effects: {e}")
            return f"Error: {str(e)}"
    
    def apply_vocal_enhancement(self, denoise_amount: float, brightness: float,
                               warmth: float, clarity: float, target_level: float,
                               progress=gr.Progress()) -> str:
        """Apply vocal enhancement to merged timeline.
        
        Args:
            denoise_amount: Denoising strength (0.0 to 1.0)
            brightness: High frequency enhancement (0.0 to 2.0)
            warmth: Low-mid frequency enhancement (0.0 to 2.0)
            clarity: Presence boost (0.0 to 2.0)
            target_level: Target RMS level in dB (-30 to 0)
            progress: Gradio progress tracker
            
        Returns:
            Status message
        """
        try:
            if not self.timeline_clips:
                return "Error: No clips on timeline to process"
            
            logger.info("Applying vocal enhancement to timeline")
            progress(0, desc="Loading vocal enhancer...")
            
            enhancer = self._get_vocal_enhancer()
            
            # Get merged timeline audio
            if len(self.timeline_clips) == 1:
                clip_id = self.timeline_clips[0]
                audio_path = self.clip_library[clip_id].filepath
            else:
                progress(0.2, desc="Merging timeline...")
                audio_path = self._merge_timeline()
                if not audio_path:
                    return "Error: Failed to merge timeline"
            
            # Load audio
            progress(0.3, desc="Loading audio...")
            audio, sr = sf.read(audio_path)
            
            # Apply enhancement
            progress(0.5, desc="Enhancing vocals...")
            enhanced = enhancer.enhance(
                audio, sr,
                denoise_amount=denoise_amount,
                brightness=brightness,
                warmth=warmth,
                clarity=clarity,
                target_level=target_level
            )
            
            # Save enhanced audio
            progress(0.9, desc="Saving...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.timeline_dir / f"enhanced_{timestamp}.wav"
            sf.write(str(output_path), enhanced, sr)
            
            progress(1.0, desc="Complete!")
            
            logger.info(f"Vocal enhancement applied: {output_path}")
            return f"Vocal enhancement applied! Saved to: {output_path}"
            
        except Exception as e:
            logger.error(f"Failed to apply vocal enhancement: {e}")
            return f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Udio-style Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        logger.info("Creating Udio-style Gradio interface")
        
        # Get initial displays
        timeline_html, waveform_html = self._render_timeline()
        
        with gr.Blocks(title="MAGE - Udio Style", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 MAGE - Mixed Audio Generation Engine")
            gr.Markdown("### Professional DAW-Style Workflow")
            
            with gr.Row():
                # Left column: Main controls
                with gr.Column(scale=2):
                    # Prompt and lyrics
                    with gr.Group():
                        gr.Markdown("### 🎼 Generation Controls")
                        prompt_box = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your music (genre, mood, style will be interpreted)...",
                            lines=3,
                            info="Example: 'Upbeat electronic dance track with energetic bassline'"
                        )
                        
                        with gr.Row():
                            lyrics_box = gr.Textbox(
                                label="Lyrics (optional)",
                                placeholder="Enter lyrics or auto-generate...",
                                lines=5
                            )
                            with gr.Column(scale=0, min_width=180):
                                gr.Markdown("**Auto-Generate Lyrics**")
                                lyrics_prompt = gr.Textbox(
                                    label="Lyrics Prompt",
                                    placeholder="Describe the lyrics theme...",
                                    lines=2,
                                    info="Genre/mood/theme interpreted from prompt"
                                )
                                lyrics_lines = gr.Slider(4, 24, value=8, step=1, label="Lines")
                                gen_lyrics_btn = gr.Button("Generate Lyrics", size="sm")
                    
                    # Generation parameters
                    with gr.Group():
                        gr.Markdown("### ⚙️ Parameters")
                        with gr.Row():
                            bpm = gr.Slider(60, 200, value=120, step=1, label="BPM")
                            context_length = gr.Slider(0, 30, value=10, step=1, 
                                                      label="Context Length (s)",
                                                      info="MusicControlNet reference length")
                        
                        position = gr.Radio(
                            ["Intro", "Previous", "Next", "Outro"],
                            value="Next",
                            label="Timeline Position",
                            info="Where to place the generated clip"
                        )
                    
                    # Generate button
                    generate_btn = gr.Button("🎵 Generate", variant="primary", size="lg")
                    status_text = gr.Textbox(label="Status", interactive=False)
                    
                    # DAW-style timeline and waveform (stacked vertically)
                    with gr.Group():
                        gr.Markdown("### 🎹 Song Timeline & Waveform")
                        
                        # Playback controls
                        with gr.Row():
                            play_btn = gr.Button("▶️ Play", size="sm", scale=1)
                            pause_btn = gr.Button("⏸️ Pause", size="sm", scale=1)
                            stop_btn = gr.Button("⏹️ Stop", size="sm", scale=1)
                            export_btn = gr.Button("💾 Export Timeline", size="sm", scale=2, variant="secondary")
                        
                        export_status = gr.Textbox(label="Export Status", interactive=False, visible=True)
                        timeline_display = gr.HTML(value=timeline_html, label="Timeline")
                        waveform_display = gr.HTML(value=waveform_html, label="Waveform")
                        
                        # Audio player (hidden, for playback)
                        audio_player = gr.Audio(label="Timeline Audio", visible=False, autoplay=False)
                
                # Right column: Clip library
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Clip Library")
                    library_display = gr.HTML(value=self._render_library())
            
            # Advanced controls (collapsible)
            with gr.Accordion("🎚️ Advanced Controls", open=False):
                gr.Markdown("### DAW-Style Effects & Enhancement")
                
                with gr.Tab("🎛️ EQ & Effects"):
                    gr.Markdown("#### 3-Band EQ")
                    with gr.Row():
                        eq_low = gr.Slider(-12, 12, value=0, step=0.5, label="Low (Bass)")
                        eq_mid = gr.Slider(-12, 12, value=0, step=0.5, label="Mid")
                        eq_high = gr.Slider(-12, 12, value=0, step=0.5, label="High (Treble)")
                    
                    gr.Markdown("#### Compressor")
                    with gr.Row():
                        comp_threshold = gr.Slider(-60, 0, value=-20, step=1, label="Threshold (dB)")
                        comp_ratio = gr.Slider(1, 20, value=4, step=0.5, label="Ratio")
                    
                    gr.Markdown("#### Reverb")
                    with gr.Row():
                        reverb_size = gr.Slider(0, 1, value=0.5, step=0.05, label="Room Size")
                        reverb_damp = gr.Slider(0, 1, value=0.5, step=0.05, label="Damping")
                    
                    gr.Markdown("#### Limiter")
                    with gr.Row():
                        limiter_threshold = gr.Slider(-20, 0, value=-1, step=0.5, label="Threshold (dB)")
                        limiter_release = gr.Slider(10, 1000, value=100, step=10, label="Release (ms)")
                    
                    apply_effects_btn = gr.Button("Apply Effects to Timeline", variant="primary")
                    effects_status = gr.Textbox(label="Effects Status", interactive=False)
                
                with gr.Tab("🎙️ Vocal Enhancement"):
                    gr.Markdown("#### Vocal Processing")
                    vocal_denoise = gr.Slider(0, 1, value=0.7, step=0.05, label="Denoise Amount")
                    vocal_brightness = gr.Slider(0, 2, value=1.0, step=0.1, label="Brightness")
                    vocal_warmth = gr.Slider(0, 2, value=1.0, step=0.1, label="Warmth")
                    vocal_clarity = gr.Slider(0, 2, value=1.0, step=0.1, label="Clarity/Presence")
                    vocal_level = gr.Slider(-30, 0, value=-18, step=1, label="Target Level (dB)")
                    
                    apply_vocal_btn = gr.Button("Apply Vocal Enhancement to Timeline", variant="primary")
                    vocal_status = gr.Textbox(label="Vocal Status", interactive=False)
            
            # Event handlers
            gen_lyrics_btn.click(
                fn=self.generate_lyrics,
                inputs=[lyrics_prompt, lyrics_lines],
                outputs=lyrics_box
            )
            
            generate_btn.click(
                fn=self.generate_clip,
                inputs=[prompt_box, lyrics_box, bpm, position, context_length],
                outputs=[status_text, library_display, timeline_display, waveform_display]
            )
            
            apply_effects_btn.click(
                fn=self.apply_advanced_effects,
                inputs=[
                    eq_low, eq_mid, eq_high,
                    comp_threshold, comp_ratio,
                    reverb_size, reverb_damp,
                    limiter_threshold, limiter_release
                ],
                outputs=effects_status
            )
            
            apply_vocal_btn.click(
                fn=self.apply_vocal_enhancement,
                inputs=[
                    vocal_denoise, vocal_brightness, vocal_warmth,
                    vocal_clarity, vocal_level
                ],
                outputs=vocal_status
            )
            
            # Playback controls
            play_btn.click(
                fn=self._load_timeline_audio,
                inputs=[],
                outputs=audio_player
            )
            
            export_btn.click(
                fn=self._export_timeline,
                inputs=[],
                outputs=[export_status]
            )
        
        logger.info("Udio-style interface created successfully")
        return interface
    
    def _export_timeline(self) -> tuple[str, str]:
        """Export the timeline as a merged audio file.
        
        Returns:
            Status message and file path for download
        """
        try:
            if not self.timeline:
                return "❌ Timeline is empty", ""
            
            logger.info("Exporting timeline...")
            
            # Merge timeline
            merged_audio, sample_rate = self._merge_timeline()
            
            # Save exported file
            export_dir = self.output_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"timeline_export_{timestamp}.wav"
            
            import soundfile as sf
            sf.write(str(export_path), merged_audio, sample_rate)
            
            logger.info(f"Timeline exported to {export_path}")
            return f"✅ Exported to {export_path.name}", str(export_path)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return f"❌ Export failed: {str(e)}", ""
    
    def _load_timeline_audio(self) -> str:
        """Load the current timeline as audio for playback.
        
        Returns:
            Path to temporary audio file for playback
        """
        try:
            if not self.timeline:
                logger.warning("Timeline empty, cannot load audio")
                return ""
            
            # Merge timeline
            merged_audio, sample_rate = self._merge_timeline()
            
            # Save to temp file for playback
            temp_dir = self.output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / "timeline_playback.wav"
            
            import soundfile as sf
            sf.write(str(temp_path), merged_audio, sample_rate)
            
            logger.info("Timeline loaded for playback")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to load timeline audio: {e}")
            return ""
    
    def launch(self, server_name: str = "127.0.0.1", server_port: int = 7860,
              share: bool = False, debug: bool = False):
        """Launch the Gradio interface.
        
        Args:
            server_name: Server host address
            server_port: Server port
            share: Create public link
            debug: Enable debug mode
        """
        logger.info(f"Launching Udio interface on {server_name}:{server_port}")
        
        interface = self.create_interface()
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug
        )


def is_udio_available() -> bool:
    """Check if Gradio is available for Udio interface.
    
    Returns:
        True if Gradio is installed and available
    """
    return GRADIO_AVAILABLE
