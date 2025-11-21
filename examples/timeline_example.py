#!/usr/bin/env python3
"""
Examples for Timeline and Arrangement System

This file demonstrates the timeline module capabilities:
1. Creating and managing tracks
2. Building timelines with multiple tracks
3. Using the arrangement engine
4. Exporting mixed audio
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mage.timeline import (
    Track,
    TrackType,
    FadeType,
    Timeline,
    ArrangementEngine,
    ArrangementSection,
    TimelineMarker
)
from mage.config import Config
from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


def generate_tone(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a simple sine wave tone."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def example_1_basic_track():
    """Example 1: Create a basic audio track"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Track Creation")
    print("="*60)
    
    # Generate 10 seconds of audio (440 Hz = A4 note)
    audio = generate_tone(440.0, 10.0)
    
    # Create a track
    track = Track(
        name="Lead Vocals",
        track_type=TrackType.VOCALS,
        audio_data=audio,
        sample_rate=44100,
        start_time=2.0,      # Start at 2 seconds
        volume=0.8,          # 80% volume
        fade_in=1.0,         # 1 second fade-in
        fade_out=1.5,        # 1.5 second fade-out
        fade_type=FadeType.S_CURVE
    )
    
    print(f"Created track: {track.name}")
    print(f"  Type: {track.track_type.value}")
    print(f"  Duration: {track.get_audio_duration():.2f}s")
    print(f"  Timeline position: {track.start_time:.2f}s - {track.end_time:.2f}s")
    print(f"  Volume: {track.volume}")
    print(f"  Fades: {track.fade_in}s in, {track.fade_out}s out")


def example_2_stereo_panning():
    """Example 2: Stereo track with panning"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Stereo Track with Panning")
    print("="*60)
    
    # Generate stereo audio
    audio_left = generate_tone(440.0, 8.0)
    audio_right = generate_tone(554.37, 8.0)  # C#5
    audio_stereo = np.stack([audio_left, audio_right])
    
    # Create track with right panning
    track = Track(
        name="Guitar",
        track_type=TrackType.INSTRUMENTAL,
        audio_data=audio_stereo,
        sample_rate=44100,
        pan=0.7,  # Pan 70% to the right
        volume=0.6
    )
    
    print(f"Created stereo track: {track.name}")
    print(f"  Stereo: {track.is_stereo}")
    print(f"  Pan: {track.pan} (0 = center, -1 = left, 1 = right)")
    print(f"  Audio shape: {track.audio_data.shape} (channels, samples)")


def example_3_simple_timeline():
    """Example 3: Create a simple timeline with multiple tracks"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Simple Multi-Track Timeline")
    print("="*60)
    
    # Create timeline
    timeline = Timeline(sample_rate=44100, name="My Song")
    
    # Create tracks with different instruments
    vocals = generate_tone(440.0, 15.0)
    instrumental = generate_tone(330.0, 20.0)
    bass = generate_tone(110.0, 20.0)
    
    track1 = Track("Vocals", TrackType.VOCALS, vocals, 44100, start_time=5.0, volume=0.9)
    track2 = Track("Synth", TrackType.INSTRUMENTAL, instrumental, 44100, start_time=0.0, volume=0.7)
    track3 = Track("Bass", TrackType.BASS, bass, 44100, start_time=0.0, volume=0.6)
    
    # Add tracks to timeline
    timeline.add_track(track1)
    timeline.add_track(track2)
    timeline.add_track(track3)
    
    print(f"Timeline: {timeline.name}")
    print(f"  Tracks: {len(timeline.tracks)}")
    print(f"  Duration: {timeline.get_duration():.2f}s")
    print(f"  Sample rate: {timeline.sample_rate} Hz")
    
    # List all tracks
    for track in timeline.tracks:
        print(f"  - {track.name} ({track.track_type.value}): "
              f"{track.start_time:.1f}s - {track.end_time:.1f}s")


def example_4_render_and_export():
    """Example 4: Render timeline and export to file"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Render and Export Timeline")
    print("="*60)
    
    # Create timeline
    timeline = Timeline(sample_rate=44100, name="Export Demo")
    
    # Add tracks
    audio1 = generate_tone(440.0, 8.0)
    audio2 = generate_tone(523.25, 8.0)
    
    track1 = Track("Track1", TrackType.VOCALS, audio1, 44100, fade_in=0.5, fade_out=0.5)
    track2 = Track("Track2", TrackType.INSTRUMENTAL, audio2, 44100, start_time=4.0)
    
    timeline.add_track(track1)
    timeline.add_track(track2)
    
    # Render to numpy array
    mixed_audio, sample_rate = timeline.render()
    
    print(f"Rendered audio:")
    print(f"  Shape: {mixed_audio.shape} (channels, samples)")
    print(f"  Duration: {mixed_audio.shape[1] / sample_rate:.2f}s")
    print(f"  Peak level: {np.max(np.abs(mixed_audio)):.3f}")
    
    # Export to file
    output_path = Path("output/my_song.wav")
    timeline.export(output_path)
    
    print(f"✓ Exported to: {output_path}")


def example_5_arrangement_engine():
    """Example 5: Use arrangement engine to create song structure"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Arrangement Engine")
    print("="*60)
    
    # Create arrangement engine
    engine = ArrangementEngine(tempo=120.0, time_signature=(4, 4))
    
    print(f"Arrangement Engine:")
    print(f"  Tempo: {engine.tempo} BPM")
    print(f"  Time signature: {engine.time_signature[0]}/{engine.time_signature[1]}")
    print(f"  Bar duration: {engine.get_bar_duration():.2f}s")
    
    # Create song structure
    structure = engine.create_simple_structure(
        include_intro=True,
        num_verses=2,
        num_choruses=3,
        include_bridge=True,
        include_outro=True
    )
    
    print(f"\nSong structure ({len(structure)} sections):")
    total_duration = 0.0
    for section_name, duration in structure:
        total_duration += duration
        print(f"  {section_name:12} - {duration:5.1f}s (total: {total_duration:5.1f}s)")


def example_6_auto_arrange():
    """Example 6: Automatically arrange tracks into a song"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Automatic Track Arrangement")
    print("="*60)
    
    # Create tracks
    vocals = generate_tone(440.0, 30.0)
    instrumental = generate_tone(330.0, 30.0)
    bass = generate_tone(110.0, 30.0)
    drums = generate_tone(220.0, 30.0)
    
    tracks = [
        Track("Vocals", TrackType.VOCALS, vocals, 44100),
        Track("Instrumental", TrackType.INSTRUMENTAL, instrumental, 44100),
        Track("Bass", TrackType.BASS, bass, 44100),
        Track("Drums", TrackType.DRUMS, drums, 44100)
    ]
    
    # Create arrangement engine
    engine = ArrangementEngine(tempo=140.0)
    
    # Automatically arrange tracks
    timeline = engine.arrange_tracks(tracks, timeline_name="Auto-Arranged Song")
    
    print(f"Created arranged timeline: {timeline.name}")
    print(f"  Tracks: {len(timeline.tracks)}")
    print(f"  Sections: {len(timeline.sections)}")
    print(f"  Total duration: {timeline.get_duration():.2f}s")
    
    # Show sections
    print(f"\nArrangement sections:")
    for section in timeline.sections:
        print(f"  {section.name:12} - {section.start_time:5.1f}s to {section.end_time:5.1f}s")


def example_7_advanced_arrangement():
    """Example 7: Advanced arrangement with custom structure"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Advanced Custom Arrangement")
    print("="*60)
    
    # Create timeline
    timeline = Timeline(sample_rate=44100, name="Custom Arrangement")
    
    # Create tracks
    vocals = generate_tone(440.0, 20.0)
    instrumental = generate_tone(330.0, 20.0)
    
    track1 = Track("Vocals", TrackType.VOCALS, vocals, 44100, start_time=8.0)
    track2 = Track("Instrumental", TrackType.INSTRUMENTAL, instrumental, 44100)
    
    timeline.add_track(track1)
    timeline.add_track(track2)
    
    # Add custom sections
    intro = ArrangementSection(
        name="intro",
        start_time=0.0,
        duration=8.0,
        track_config={
            TrackType.VOCALS: 0.0,  # No vocals in intro
            TrackType.INSTRUMENTAL: 0.7
        }
    )
    
    verse = ArrangementSection(
        name="verse1",
        start_time=8.0,
        duration=16.0,
        track_config={
            TrackType.VOCALS: 0.9,
            TrackType.INSTRUMENTAL: 0.6
        }
    )
    
    timeline.add_section(intro)
    timeline.add_section(verse)
    
    # Add markers
    marker1 = TimelineMarker("Verse Start", 8.0, color="#FF0000")
    marker2 = TimelineMarker("Hook", 16.0, color="#00FF00")
    
    timeline.add_marker(marker1)
    timeline.add_marker(marker2)
    
    print(f"Timeline: {timeline.name}")
    print(f"  Sections: {len(timeline.sections)}")
    print(f"  Markers: {len(timeline.markers)}")
    
    for section in timeline.sections:
        print(f"  Section '{section.name}': {section.start_time}s - {section.end_time}s")
    
    for marker in timeline.markers:
        print(f"  Marker '{marker.name}' at {marker.time}s")


def example_8_config_based():
    """Example 8: Create arrangement from configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Configuration-Based Arrangement")
    print("="*60)
    
    # Load configuration
    config = Config.from_file("config/config.yaml")
    
    # Create engine from config
    engine = ArrangementEngine(
        tempo=config.timeline.default_tempo,
        time_signature=(
            config.timeline.time_signature_numerator,
            config.timeline.time_signature_denominator
        )
    )
    
    print(f"Engine from config:")
    print(f"  Tempo: {engine.tempo} BPM")
    print(f"  Time signature: {engine.time_signature[0]}/{engine.time_signature[1]}")
    
    # Use config section durations
    custom_structure = [
        ("intro", config.timeline.intro_duration),
        ("verse1", config.timeline.verse_duration),
        ("chorus1", config.timeline.chorus_duration),
        ("verse2", config.timeline.verse_duration),
        ("bridge", config.timeline.bridge_duration),
        ("chorus2", config.timeline.chorus_duration),
        ("outro", config.timeline.outro_duration)
    ]
    
    sections = engine.create_sections_from_structure(custom_structure)
    
    print(f"\nCreated {len(sections)} sections from config:")
    for section in sections:
        print(f"  {section.name:8} - {section.duration:5.1f}s")


def example_9_crossfade():
    """Example 9: Crossfade between tracks"""
    print("\n" + "="*60)
    print("EXAMPLE 9: Crossfade Transitions")
    print("="*60)
    
    engine = ArrangementEngine(tempo=120.0)
    
    # Create two tracks
    audio1 = generate_tone(440.0, 10.0)
    audio2 = generate_tone(523.25, 10.0)
    
    track1 = Track("Part1", TrackType.INSTRUMENTAL, audio1, 44100)
    track2 = Track("Part2", TrackType.INSTRUMENTAL, audio2, 44100)
    
    # Create crossfade
    crossfaded_tracks = engine.create_crossfade_transition(track1, track2, crossfade_duration=2.0)
    
    print(f"Created crossfade between tracks:")
    for track in crossfaded_tracks:
        print(f"  {track.name}: {track.start_time:.2f}s - {track.end_time:.2f}s")
        print(f"    Fade in: {track.fade_in}s, Fade out: {track.fade_out}s")


def main():
    """Run all examples."""
    print("="*60)
    print("MAGE TIMELINE & ARRANGEMENT EXAMPLES")
    print("="*60)
    
    examples = [
        example_1_basic_track,
        example_2_stereo_panning,
        example_3_simple_timeline,
        example_4_render_and_export,
        example_5_arrangement_engine,
        example_6_auto_arrange,
        example_7_advanced_arrangement,
        example_8_config_based,
        example_9_crossfade
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
            logger.exception(f"Example {i} failed")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
