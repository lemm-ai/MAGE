"""Test suite for Udio-style interface.

This module tests the timeline-based workflow, clip library management,
and advanced DAW controls.
"""

import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from mage.gui import UdioInterface, is_udio_available


def test_udio_availability():
    """Test 1: Check Udio interface availability."""
    print("\n" + "=" * 60)
    print("TEST 1: Check Udio Interface Availability")
    print("=" * 60)
    
    available = is_udio_available()
    print(f"Udio interface available: {available}")
    
    assert available, "Udio interface should be available"
    print("✅ Test 1 PASSED\n")


def test_interface_initialization():
    """Test 2: Initialize UdioInterface."""
    print("=" * 60)
    print("TEST 2: Initialize UdioInterface")
    print("=" * 60)
    
    udio = UdioInterface()
    
    print(f"✓ UdioInterface initialized")
    print(f"  - Output dir: {udio.output_dir}")
    print(f"  - Clips dir: {udio.clips_dir}")
    print(f"  - Timeline dir: {udio.timeline_dir}")
    print(f"  - Library file: {udio.library_file}")
    
    assert udio.output_dir.exists(), "Output directory should exist"
    assert udio.clips_dir.exists(), "Clips directory should exist"
    assert udio.timeline_dir.exists(), "Timeline directory should exist"
    
    print("✅ Test 2 PASSED\n")
    return udio


def test_generate_lyrics(udio):
    """Test 3: Generate lyrics."""
    print("=" * 60)
    print("TEST 3: Generate Lyrics")
    print("=" * 60)
    
    lyrics = udio.generate_lyrics(
        prompt="Write upbeat rock lyrics about freedom and adventure",
        lines=8
    )
    
    print(f"✓ Lyrics generated")
    print(f"  - Length: {len(lyrics)} characters")
    print(f"  - Preview: {lyrics[:100]}...")
    
    assert len(lyrics) > 0, "Lyrics should not be empty"
    
    print("✅ Test 3 PASSED\n")
    return lyrics


def test_generate_clip(udio):
    """Test 4: Generate a clip and add to library."""
    print("=" * 60)
    print("TEST 4: Generate Clip")
    print("=" * 60)
    
    status, library_html, timeline_html, waveform_html = udio.generate_clip(
        prompt="Epic rock anthem with powerful drums and energetic guitars",
        lyrics="We are the champions",
        bpm=120,
        position="Next",
        context_length=10
    )
    
    print(f"✓ Clip generated")
    print(f"  - Status: {status}")
    print(f"  - Library clips: {len(udio.clip_library)}")
    print(f"  - Timeline clips: {len(udio.timeline_clips)}")
    
    assert len(udio.clip_library) > 0, "Clip library should have clips"
    assert len(udio.timeline_clips) > 0, "Timeline should have clips"
    assert "successfully" in status.lower(), "Status should indicate success"
    
    print("✅ Test 4 PASSED\n")


def test_generate_multiple_clips(udio):
    """Test 5: Generate multiple clips with different positions."""
    print("=" * 60)
    print("TEST 5: Generate Multiple Clips")
    print("=" * 60)
    
    positions = ["Intro", "Next", "Outro"]
    prompts = [
        "Calm ambient electronic intro",
        "Upbeat electronic dance section",
        "Slow atmospheric electronic outro"
    ]
    initial_count = len(udio.clip_library)
    
    for i, (position, prompt) in enumerate(zip(positions, prompts)):
        status, _, _, _ = udio.generate_clip(
            prompt=prompt,
            lyrics="",
            bpm=120 + i*10,
            position=position,
            context_length=5
        )
        print(f"  - Generated clip at {position}: {status[:50]}...")
    
    print(f"✓ Multiple clips generated")
    print(f"  - Initial count: {initial_count}")
    print(f"  - Final count: {len(udio.clip_library)}")
    print(f"  - Timeline length: {len(udio.timeline_clips)}")
    
    assert len(udio.clip_library) > initial_count, "Should have more clips"
    
    print("✅ Test 5 PASSED\n")


def test_library_rendering(udio):
    """Test 6: Render clip library HTML."""
    print("=" * 60)
    print("TEST 6: Render Clip Library")
    print("=" * 60)
    
    library_html = udio._render_library()
    
    print(f"✓ Library rendered")
    print(f"  - HTML length: {len(library_html)} characters")
    print(f"  - Contains clips: {'clip' in library_html.lower()}")
    
    assert len(library_html) > 0, "Library HTML should not be empty"
    
    print("✅ Test 6 PASSED\n")


def test_timeline_rendering(udio):
    """Test 7: Render timeline with waveforms."""
    print("=" * 60)
    print("TEST 7: Render Timeline & Waveform")
    print("=" * 60)
    
    timeline_html, waveform_html = udio._render_timeline()
    
    print(f"✓ Timeline and waveform rendered")
    print(f"  - Timeline HTML length: {len(timeline_html)} characters")
    print(f"  - Waveform HTML length: {len(waveform_html)} characters")
    print(f"  - Timeline contains divs: {'div' in timeline_html.lower()}")
    print(f"  - Waveform contains image: {'img' in waveform_html.lower()}")
    
    assert len(timeline_html) > 0, "Timeline HTML should not be empty"
    assert len(waveform_html) > 0, "Waveform HTML should not be empty"
    
    print("✅ Test 7 PASSED\n")


def test_timeline_merging(udio):
    """Test 8: Merge timeline clips."""
    print("=" * 60)
    print("TEST 8: Merge Timeline")
    print("=" * 60)
    
    initial_timeline_count = len(udio.timeline_clips)
    
    if initial_timeline_count > 1:
        merged_path = udio._merge_timeline()
        
        print(f"✓ Timeline merged")
        print(f"  - Initial clips: {initial_timeline_count}")
        print(f"  - Final clips: {len(udio.timeline_clips)}")
        print(f"  - Merged file: {merged_path}")
        
        assert merged_path is not None, "Merge should produce a file"
        assert os.path.exists(merged_path), "Merged file should exist"
        assert len(udio.timeline_clips) == 1, "Timeline should have 1 merged clip"
    else:
        print(f"  - Not enough clips to merge (need >1, have {initial_timeline_count})")
    
    print("✅ Test 8 PASSED\n")


def test_clip_deletion(udio):
    """Test 9: Delete a clip."""
    print("=" * 60)
    print("TEST 9: Delete Clip")
    print("=" * 60)
    
    if udio.clip_library:
        # Get first clip ID
        clip_id = list(udio.clip_library.keys())[0]
        clip_name = udio.clip_library[clip_id].name
        initial_count = len(udio.clip_library)
        
        # Delete clip
        library_html, timeline_html, waveform_html = udio.delete_clip(clip_id)
        
        print(f"✓ Clip deleted")
        print(f"  - Deleted: {clip_name}")
        print(f"  - Initial count: {initial_count}")
        print(f"  - Final count: {len(udio.clip_library)}")
        
        assert len(udio.clip_library) == initial_count - 1, "Clip should be deleted"
        assert clip_id not in udio.clip_library, "Clip ID should not exist"
    else:
        print("  - No clips to delete")
    
    print("✅ Test 9 PASSED\n")


def test_extend_clip(udio):
    """Test 10: Extend from a clip."""
    print("=" * 60)
    print("TEST 10: Extend Clip")
    print("=" * 60)
    
    if udio.clip_library:
        # Get last clip ID
        clip_id = list(udio.clip_library.keys())[-1]
        clip_name = udio.clip_library[clip_id].name
        
        # Extend from clip
        library_html, timeline_html, waveform_html = udio.extend_clip(clip_id)
        
        print(f"✓ Extended from clip")
        print(f"  - Extended: {clip_name}")
        print(f"  - Timeline length: {len(udio.timeline_clips)}")
        print(f"  - Timeline contains clip: {clip_id in udio.timeline_clips}")
        
        assert len(udio.timeline_clips) == 1, "Timeline should have only extended clip"
        assert udio.timeline_clips[0] == clip_id, "Timeline should contain the clip"
    else:
        print("  - No clips to extend from")
    
    print("✅ Test 10 PASSED\n")


def test_create_interface(udio):
    """Test 11: Create Gradio interface."""
    print("=" * 60)
    print("TEST 11: Create Gradio Interface")
    print("=" * 60)
    
    interface = udio.create_interface()
    
    print(f"✓ Gradio interface created")
    print(f"  - Type: {type(interface).__name__}")
    
    assert interface is not None, "Interface should be created"
    
    print("✅ Test 11 PASSED\n")


def test_advanced_effects(udio):
    """Test 12: Apply advanced effects."""
    print("=" * 60)
    print("TEST 12: Apply Advanced Effects")
    print("=" * 60)
    
    if udio.timeline_clips:
        status = udio.apply_advanced_effects(
            eq_low=2.0,
            eq_mid=0.0,
            eq_high=1.5,
            compressor_threshold=-20,
            compressor_ratio=4.0,
            reverb_room_size=0.5,
            reverb_damping=0.5,
            limiter_threshold=-1.0,
            limiter_release=100
        )
        
        print(f"✓ Advanced effects applied")
        print(f"  - Status: {status}")
        
        assert "error" not in status.lower() or "no clips" in status.lower(), \
            "Effects should apply or indicate no clips"
    else:
        print("  - No clips on timeline to process")
    
    print("✅ Test 12 PASSED\n")


def test_vocal_enhancement(udio):
    """Test 13: Apply vocal enhancement."""
    print("=" * 60)
    print("TEST 13: Apply Vocal Enhancement")
    print("=" * 60)
    
    if udio.timeline_clips:
        status = udio.apply_vocal_enhancement(
            denoise_amount=0.7,
            brightness=1.2,
            warmth=1.1,
            clarity=1.3,
            target_level=-18
        )
        
        print(f"✓ Vocal enhancement applied")
        print(f"  - Status: {status}")
        
        assert "error" not in status.lower() or "no clips" in status.lower(), \
            "Enhancement should apply or indicate no clips"
    else:
        print("  - No clips on timeline to process")
    
    print("✅ Test 13 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MAGE UDIO INTERFACE TESTS")
    print("=" * 60)
    
    try:
        # Test availability
        test_udio_availability()
        
        # Initialize interface
        udio = test_interface_initialization()
        
        # Test lyrics generation
        test_generate_lyrics(udio)
        
        # Test clip generation
        test_generate_clip(udio)
        test_generate_multiple_clips(udio)
        
        # Test rendering
        test_library_rendering(udio)
        test_timeline_rendering(udio)
        
        # Test timeline operations
        test_timeline_merging(udio)
        
        # Test clip operations
        test_clip_deletion(udio)
        test_extend_clip(udio)
        
        # Test interface creation
        test_create_interface(udio)
        
        # Test advanced features
        test_advanced_effects(udio)
        test_vocal_enhancement(udio)
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("Tests passed: 13/13")
        print("✅ ALL TESTS PASSED!")
        print("\nNote: To launch the Udio interface, run:")
        print("  python examples/udio_example.py")
        print("=" * 60)
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
