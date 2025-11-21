#!/usr/bin/env python3
"""
Test script for Phase 4: Timeline & Arrangement System

This script tests the timeline module functionality including:
- Track creation and management
- Timeline mixing and rendering
- Arrangement engine with song structures
- Crossfading and transitions
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mage.timeline import (
    Track,
    TrackType,
    FadeType,
    Timeline,
    ArrangementEngine,
    ArrangementSection
)
from mage.config import Config
from mage.utils import MAGELogger

# Configure logging
logger = MAGELogger.get_logger(__name__)


def create_test_audio(duration: float, frequency: float, sample_rate: int = 44100) -> np.ndarray:
    """Create simple sine wave test audio."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def test_1_track_creation():
    """Test 1: Create tracks with different configurations"""
    print("\n" + "="*60)
    print("TEST 1: Track Creation")
    print("="*60)
    
    try:
        # Create test audio
        audio_mono = create_test_audio(5.0, 440.0)  # A4 note, 5 seconds
        audio_stereo = np.stack([audio_mono, audio_mono * 0.8])
        
        # Create mono track
        track1 = Track(
            name="Vocals",
            track_type=TrackType.VOCALS,
            audio_data=audio_mono,
            sample_rate=44100,
            start_time=2.0,
            volume=0.8,
            fade_in=1.0,
            fade_out=1.0
        )
        
        # Create stereo track
        track2 = Track(
            name="Instrumental",
            track_type=TrackType.INSTRUMENTAL,
            audio_data=audio_stereo,
            sample_rate=44100,
            start_time=0.0,
            pan=0.5
        )
        
        print(f"✓ Created mono track: {track1.name}")
        print(f"  - Type: {track1.track_type.value}")
        print(f"  - Duration: {track1.get_audio_duration():.2f}s")
        print(f"  - Start: {track1.start_time:.2f}s, End: {track1.end_time:.2f}s")
        print(f"  - Mono: {track1.is_mono}, Stereo: {track1.is_stereo}")
        
        print(f"✓ Created stereo track: {track2.name}")
        print(f"  - Type: {track2.track_type.value}")
        print(f"  - Duration: {track2.get_audio_duration():.2f}s")
        print(f"  - Pan: {track2.pan}")
        
        print("✅ Test 1 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        logger.exception("Test 1 failed")
        return False


def test_2_timeline_basic():
    """Test 2: Basic timeline operations"""
    print("\n" + "="*60)
    print("TEST 2: Basic Timeline Operations")
    print("="*60)
    
    try:
        # Create timeline
        timeline = Timeline(sample_rate=44100, name="Test Timeline")
        
        # Create tracks
        vocals = create_test_audio(10.0, 440.0)
        instrumental = create_test_audio(15.0, 330.0)
        
        track1 = Track("Vocals", TrackType.VOCALS, vocals, 44100, start_time=2.0)
        track2 = Track("Instrumental", TrackType.INSTRUMENTAL, instrumental, 44100, start_time=0.0)
        
        # Add tracks
        timeline.add_track(track1)
        timeline.add_track(track2)
        
        print(f"✓ Timeline: {timeline.name}")
        print(f"  - Sample rate: {timeline.sample_rate} Hz")
        print(f"  - Number of tracks: {len(timeline.tracks)}")
        print(f"  - Duration: {timeline.get_duration():.2f}s")
        
        # Get track by name
        found_track = timeline.get_track("Vocals")
        print(f"✓ Found track: {found_track.name if found_track else 'None'}")
        
        # Get tracks by type
        vocal_tracks = timeline.get_tracks_by_type(TrackType.VOCALS)
        print(f"✓ Vocal tracks: {len(vocal_tracks)}")
        
        # Remove track
        removed = timeline.remove_track("Vocals")
        print(f"✓ Removed track: {removed}")
        print(f"  - Tracks remaining: {len(timeline.tracks)}")
        
        print("✅ Test 2 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        logger.exception("Test 2 failed")
        return False


def test_3_timeline_rendering():
    """Test 3: Render timeline to audio"""
    print("\n" + "="*60)
    print("TEST 3: Timeline Rendering")
    print("="*60)
    
    try:
        # Create timeline
        timeline = Timeline(sample_rate=44100, name="Render Test")
        
        # Create tracks with different frequencies
        audio1 = create_test_audio(8.0, 440.0)  # A4
        audio2 = create_test_audio(8.0, 523.25)  # C5
        audio3 = create_test_audio(8.0, 329.63)  # E4
        
        track1 = Track("Track1", TrackType.VOCALS, audio1, 44100, start_time=0.0, volume=0.5)
        track2 = Track("Track2", TrackType.INSTRUMENTAL, audio2, 44100, start_time=2.0, volume=0.3)
        track3 = Track("Track3", TrackType.BASS, audio3, 44100, start_time=4.0, volume=0.4, fade_in=1.0, fade_out=1.0)
        
        timeline.add_track(track1)
        timeline.add_track(track2)
        timeline.add_track(track3)
        
        print(f"✓ Created timeline with {len(timeline.tracks)} tracks")
        
        # Render
        mixed_audio, sample_rate = timeline.render()
        
        print(f"✓ Rendered audio:")
        print(f"  - Shape: {mixed_audio.shape} (channels, samples)")
        print(f"  - Duration: {mixed_audio.shape[1] / sample_rate:.2f}s")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Peak level: {np.max(np.abs(mixed_audio)):.3f}")
        
        # Test mute/solo
        track1.mute = True
        mixed_muted, _ = timeline.render()
        print(f"✓ Rendered with track1 muted (peak: {np.max(np.abs(mixed_muted)):.3f})")
        
        track1.mute = False
        track2.solo = True
        mixed_solo, _ = timeline.render()
        print(f"✓ Rendered with track2 solo (peak: {np.max(np.abs(mixed_solo)):.3f})")
        
        print("✅ Test 3 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        logger.exception("Test 3 failed")
        return False


def test_4_arrangement_engine():
    """Test 4: Arrangement engine basics"""
    print("\n" + "="*60)
    print("TEST 4: Arrangement Engine")
    print("="*60)
    
    try:
        # Create arrangement engine
        engine = ArrangementEngine(tempo=120.0, time_signature=(4, 4))
        
        print(f"✓ Created ArrangementEngine:")
        print(f"  - Tempo: {engine.tempo} BPM")
        print(f"  - Time signature: {engine.time_signature[0]}/{engine.time_signature[1]}")
        print(f"  - Bar duration: {engine.get_bar_duration():.2f}s")
        
        # Test bar conversion
        bars_8 = engine.bars_to_seconds(8)
        print(f"✓ 8 bars = {bars_8:.2f}s")
        
        seconds_16 = engine.seconds_to_bars(16.0)
        print(f"✓ 16s = {seconds_16} bars")
        
        # Test quantization
        time_unquantized = 3.14159
        time_quantized = engine.quantize_to_grid(time_unquantized, grid_division=4)
        print(f"✓ Quantized {time_unquantized:.3f}s to {time_quantized:.3f}s (quarter notes)")
        
        print("✅ Test 4 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        logger.exception("Test 4 failed")
        return False


def test_5_song_structure():
    """Test 5: Create song structure"""
    print("\n" + "="*60)
    print("TEST 5: Song Structure Creation")
    print("="*60)
    
    try:
        engine = ArrangementEngine(tempo=120.0)
        
        # Create simple structure
        structure = engine.create_simple_structure(
            include_intro=True,
            num_verses=2,
            num_choruses=3,
            include_bridge=True,
            include_outro=True
        )
        
        print(f"✓ Created song structure with {len(structure)} sections:")
        total_duration = 0.0
        for section_name, duration in structure:
            total_duration += duration
            print(f"  - {section_name}: {duration:.1f}s")
        print(f"  Total duration: {total_duration:.1f}s")
        
        # Create arrangement sections
        sections = engine.create_sections_from_structure(structure)
        
        print(f"✓ Created {len(sections)} arrangement sections")
        for section in sections[:3]:  # Show first 3
            print(f"  - {section.name}: {section.start_time:.1f}s - {section.end_time:.1f}s")
            print(f"    Volumes: {dict(list(section.track_config.items())[:2])}")
        
        print("✅ Test 5 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        logger.exception("Test 5 failed")
        return False


def test_6_arrange_tracks():
    """Test 6: Automatically arrange tracks"""
    print("\n" + "="*60)
    print("TEST 6: Automatic Track Arrangement")
    print("="*60)
    
    try:
        engine = ArrangementEngine(tempo=140.0)
        
        # Create test tracks
        vocals = create_test_audio(20.0, 440.0)
        instrumental = create_test_audio(20.0, 330.0)
        bass = create_test_audio(20.0, 110.0)
        drums = create_test_audio(20.0, 220.0)
        
        tracks = [
            Track("Vocals", TrackType.VOCALS, vocals, 44100),
            Track("Instrumental", TrackType.INSTRUMENTAL, instrumental, 44100),
            Track("Bass", TrackType.BASS, bass, 44100),
            Track("Drums", TrackType.DRUMS, drums, 44100)
        ]
        
        print(f"✓ Created {len(tracks)} tracks")
        
        # Arrange tracks
        timeline = engine.arrange_tracks(
            tracks,
            timeline_name="Auto-Arranged Song"
        )
        
        print(f"✓ Created arranged timeline:")
        print(f"  - Name: {timeline.name}")
        print(f"  - Tracks: {len(timeline.tracks)}")
        print(f"  - Sections: {len(timeline.sections)}")
        print(f"  - Duration: {timeline.get_duration():.2f}s")
        
        # Show timeline info
        info = timeline.get_info()
        print(f"✓ Timeline info:")
        for key, value in info.items():
            if key != 'tracks':
                print(f"  - {key}: {value}")
        
        print("✅ Test 6 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        logger.exception("Test 6 failed")
        return False


def test_7_export_timeline():
    """Test 7: Export timeline to file"""
    print("\n" + "="*60)
    print("TEST 7: Timeline Export")
    print("="*60)
    
    try:
        # Create timeline
        timeline = Timeline(sample_rate=44100, name="Export Test")
        
        # Create tracks
        audio1 = create_test_audio(5.0, 440.0)
        audio2 = create_test_audio(5.0, 523.25)
        
        track1 = Track("Track1", TrackType.VOCALS, audio1, 44100, fade_in=0.5, fade_out=0.5)
        track2 = Track("Track2", TrackType.INSTRUMENTAL, audio2, 44100, start_time=2.0)
        
        timeline.add_track(track1)
        timeline.add_track(track2)
        
        # Export
        output_path = Path("output/test_timeline.wav")
        timeline.export(output_path)
        
        print(f"✓ Exported timeline to {output_path}")
        
        # Verify file exists
        if output_path.exists():
            info = sf.info(str(output_path))
            print(f"✓ File created successfully:")
            print(f"  - Sample rate: {info.samplerate} Hz")
            print(f"  - Channels: {info.channels}")
            print(f"  - Duration: {info.duration:.2f}s")
            print(f"  - Format: {info.format}")
        else:
            raise FileNotFoundError(f"Output file not created: {output_path}")
        
        print("✅ Test 7 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")
        logger.exception("Test 7 failed")
        return False


def test_8_config_integration():
    """Test 8: Timeline configuration integration"""
    print("\n" + "="*60)
    print("TEST 8: Configuration Integration")
    print("="*60)
    
    try:
        # Load config
        config = Config.from_file("config/config.yaml")
        
        print(f"✓ Loaded configuration")
        print(f"  - Default tempo: {config.timeline.default_tempo} BPM")
        print(f"  - Time signature: {config.timeline.time_signature_numerator}/{config.timeline.time_signature_denominator}")
        print(f"  - Default crossfade: {config.timeline.default_crossfade}s")
        print(f"  - Auto normalize: {config.timeline.auto_normalize}")
        print(f"  - Quantize grid: {config.timeline.quantize_grid}")
        
        # Create engine from config
        engine = ArrangementEngine(
            tempo=config.timeline.default_tempo,
            time_signature=(
                config.timeline.time_signature_numerator,
                config.timeline.time_signature_denominator
            )
        )
        
        print(f"✓ Created ArrangementEngine from config")
        print(f"  - Tempo: {engine.tempo} BPM")
        print(f"  - Bar duration: {engine.get_bar_duration():.2f}s")
        
        # Verify section durations
        print(f"✓ Section durations from config:")
        print(f"  - Intro: {config.timeline.intro_duration}s")
        print(f"  - Verse: {config.timeline.verse_duration}s")
        print(f"  - Chorus: {config.timeline.chorus_duration}s")
        print(f"  - Bridge: {config.timeline.bridge_duration}s")
        print(f"  - Outro: {config.timeline.outro_duration}s")
        
        print("✅ Test 8 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 8 FAILED: {e}")
        logger.exception("Test 8 failed")
        return False


def main():
    """Run all timeline tests."""
    print("\n" + "="*60)
    print("MAGE PHASE 4: TIMELINE & ARRANGEMENT SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_1_track_creation,
        test_2_timeline_basic,
        test_3_timeline_rendering,
        test_4_arrangement_engine,
        test_5_song_structure,
        test_6_arrange_tracks,
        test_7_export_timeline,
        test_8_config_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            logger.exception(f"Test {test.__name__} crashed")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {total - passed} test(s) failed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
