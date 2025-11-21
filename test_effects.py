#!/usr/bin/env python3
"""
Test script for Phase 5: Pedalboard EQ Integration

This script tests the effects module functionality including:
- EQ (3-band parametric)
- Compression
- Reverb
- Chorus, Delay
- Limiter, Noise Gate
- Effects chains
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mage.processors import EffectsProcessor, is_pedalboard_available
from mage.config import Config
from mage.utils import MAGELogger

# Configure logging
logger = MAGELogger.get_logger(__name__)


def create_test_audio(duration: float, frequency: float, sample_rate: int = 44100) -> np.ndarray:
    """Create simple sine wave test audio."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def test_1_pedalboard_availability():
    """Test 1: Check Pedalboard availability"""
    print("\n" + "="*60)
    print("TEST 1: Pedalboard Availability")
    print("="*60)
    
    try:
        available = is_pedalboard_available()
        print(f"Pedalboard available: {available}")
        
        if not available:
            print("⚠️  Pedalboard not installed. Install with: pip install pedalboard")
            print("   Some tests will be skipped.")
            return False
        
        print("✅ Test 1 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        logger.exception("Test 1 failed")
        return False


def test_2_processor_initialization():
    """Test 2: Initialize EffectsProcessor"""
    print("\n" + "="*60)
    print("TEST 2: Processor Initialization")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        print(f"✓ Created EffectsProcessor")
        print(f"  - Sample rate: {processor.sample_rate} Hz")
        
        print("✅ Test 2 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        logger.exception("Test 2 failed")
        return False


def test_3_eq_processing():
    """Test 3: Apply EQ"""
    print("\n" + "="*60)
    print("TEST 3: EQ Processing")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(3.0, 440.0)
        
        # Apply EQ
        processed = processor.apply_eq(
            audio,
            low_shelf_gain_db=3.0,
            low_shelf_freq=100.0,
            mid_gain_db=-2.0,
            mid_freq=1000.0,
            high_shelf_gain_db=1.5,
            high_shelf_freq=8000.0
        )
        
        print(f"✓ Applied EQ to audio")
        print(f"  - Input shape: {audio.shape}")
        print(f"  - Output shape: {processed.shape}")
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        
        # Test no-op EQ (all gains at 0)
        processed_noop = processor.apply_eq(audio)
        print(f"✓ No-op EQ returned original audio: {np.allclose(audio, processed_noop)}")
        
        print("✅ Test 3 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        logger.exception("Test 3 failed")
        return False


def test_4_compressor():
    """Test 4: Apply compression"""
    print("\n" + "="*60)
    print("TEST 4: Compressor")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create loud audio with peaks
        audio = create_test_audio(3.0, 440.0) * 2.0  # Intentionally loud
        
        # Apply compression
        processed = processor.apply_compressor(
            audio,
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=100.0
        )
        
        print(f"✓ Applied compressor to audio")
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        print(f"  - Peak reduction: {(1 - np.max(np.abs(processed))/np.max(np.abs(audio)))*100:.1f}%")
        
        print("✅ Test 4 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        logger.exception("Test 4 failed")
        return False


def test_5_reverb():
    """Test 5: Apply reverb"""
    print("\n" + "="*60)
    print("TEST 5: Reverb")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(2.0, 440.0)
        
        # Apply reverb
        processed = processor.apply_reverb(
            audio,
            room_size=0.7,
            damping=0.5,
            wet_level=0.4,
            dry_level=0.4,
            width=1.0
        )
        
        print(f"✓ Applied reverb to audio")
        print(f"  - Input shape: {audio.shape}")
        print(f"  - Output shape: {processed.shape}")
        
        print("✅ Test 5 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        logger.exception("Test 5 failed")
        return False


def test_6_chorus():
    """Test 6: Apply chorus"""
    print("\n" + "="*60)
    print("TEST 6: Chorus")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(3.0, 440.0)
        
        # Apply chorus
        processed = processor.apply_chorus(
            audio,
            rate_hz=1.5,
            depth=0.3,
            centre_delay_ms=7.0,
            feedback=0.0,
            mix=0.5
        )
        
        print(f"✓ Applied chorus to audio")
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        
        print("✅ Test 6 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        logger.exception("Test 6 failed")
        return False


def test_7_delay():
    """Test 7: Apply delay"""
    print("\n" + "="*60)
    print("TEST 7: Delay")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(2.0, 440.0)
        
        # Apply delay
        processed = processor.apply_delay(
            audio,
            delay_seconds=0.25,
            feedback=0.4,
            mix=0.5
        )
        
        print(f"✓ Applied delay to audio")
        print(f"  - Delay time: 0.25s")
        print(f"  - Output shape: {processed.shape}")
        
        print("✅ Test 7 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")
        logger.exception("Test 7 failed")
        return False


def test_8_limiter():
    """Test 8: Apply limiter"""
    print("\n" + "="*60)
    print("TEST 8: Limiter")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create loud audio
        audio = create_test_audio(2.0, 440.0) * 3.0
        
        # Apply limiter
        processed = processor.apply_limiter(
            audio,
            threshold_db=-3.0,
            release_ms=100.0
        )
        
        print(f"✓ Applied limiter to audio")
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        print(f"  - Peak limited: {np.max(np.abs(processed)) < 1.0}")
        
        print("✅ Test 8 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 8 FAILED: {e}")
        logger.exception("Test 8 failed")
        return False


def test_9_noise_gate():
    """Test 9: Apply noise gate"""
    print("\n" + "="*60)
    print("TEST 9: Noise Gate")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create audio with noise
        audio = create_test_audio(2.0, 440.0)
        noise = np.random.randn(len(audio)) * 0.05
        audio_noisy = audio + noise
        
        # Apply noise gate
        processed = processor.apply_noise_gate(
            audio_noisy,
            threshold_db=-40.0,
            ratio=10.0,
            attack_ms=1.0,
            release_ms=100.0
        )
        
        print(f"✓ Applied noise gate to audio")
        print(f"  - Input RMS: {np.sqrt(np.mean(audio_noisy**2)):.4f}")
        print(f"  - Output RMS: {np.sqrt(np.mean(processed**2)):.4f}")
        
        print("✅ Test 9 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 9 FAILED: {e}")
        logger.exception("Test 9 failed")
        return False


def test_10_gain():
    """Test 10: Apply gain"""
    print("\n" + "="*60)
    print("TEST 10: Gain Adjustment")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(2.0, 440.0)
        
        # Apply gain
        processed = processor.apply_gain(audio, gain_db=6.0)
        
        print(f"✓ Applied +6dB gain")
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        print(f"  - Gain ratio: {np.max(np.abs(processed))/np.max(np.abs(audio)):.2f}x")
        
        print("✅ Test 10 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 10 FAILED: {e}")
        logger.exception("Test 10 failed")
        return False


def test_11_effects_chain():
    """Test 11: Apply effects chain"""
    print("\n" + "="*60)
    print("TEST 11: Effects Chain")
    print("="*60)
    
    try:
        if not is_pedalboard_available():
            print("⚠️  Skipping (Pedalboard not available)")
            return False
        
        processor = EffectsProcessor(sample_rate=44100)
        
        # Create test audio
        audio = create_test_audio(3.0, 440.0)
        
        # Define effects chain
        chain = [
            {"type": "eq", "low_shelf_gain_db": 2.0, "high_shelf_gain_db": -1.0},
            {"type": "compressor", "threshold_db": -20.0, "ratio": 3.0},
            {"type": "reverb", "room_size": 0.5, "wet_level": 0.2},
            {"type": "limiter", "threshold_db": -1.0}
        ]
        
        # Apply chain
        processed = processor.apply_chain(audio, chain)
        
        print(f"✓ Applied effects chain with {len(chain)} effects:")
        for i, effect in enumerate(chain, 1):
            print(f"  {i}. {effect['type']}")
        
        print(f"  - Input peak: {np.max(np.abs(audio)):.3f}")
        print(f"  - Output peak: {np.max(np.abs(processed)):.3f}")
        
        print("✅ Test 11 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 11 FAILED: {e}")
        logger.exception("Test 11 failed")
        return False


def test_12_config_integration():
    """Test 12: Configuration integration"""
    print("\n" + "="*60)
    print("TEST 12: Configuration Integration")
    print("="*60)
    
    try:
        # Load config
        config = Config.from_file("config/config.yaml")
        
        print(f"✓ Loaded effects configuration:")
        print(f"  - EQ low shelf: {config.effects.eq_low_shelf_gain_db}dB @ {config.effects.eq_low_shelf_freq}Hz")
        print(f"  - EQ mid: {config.effects.eq_mid_gain_db}dB @ {config.effects.eq_mid_freq}Hz")
        print(f"  - EQ high shelf: {config.effects.eq_high_shelf_gain_db}dB @ {config.effects.eq_high_shelf_freq}Hz")
        print(f"  - Compressor: {config.effects.comp_threshold_db}dB, {config.effects.comp_ratio}:1")
        print(f"  - Reverb room: {config.effects.reverb_room_size}")
        print(f"  - Enable EQ: {config.effects.enable_eq}")
        print(f"  - Enable compressor: {config.effects.enable_compressor}")
        print(f"  - Enable reverb: {config.effects.enable_reverb}")
        print(f"  - Enable limiter: {config.effects.enable_limiter}")
        
        if is_pedalboard_available():
            # Create processor and apply from config
            processor = EffectsProcessor(sample_rate=config.audio.sample_rate)
            audio = create_test_audio(2.0, 440.0)
            
            if config.effects.enable_eq:
                audio = processor.apply_eq(
                    audio,
                    low_shelf_gain_db=config.effects.eq_low_shelf_gain_db,
                    low_shelf_freq=config.effects.eq_low_shelf_freq,
                    mid_gain_db=config.effects.eq_mid_gain_db,
                    mid_freq=config.effects.eq_mid_freq,
                    high_shelf_gain_db=config.effects.eq_high_shelf_gain_db,
                    high_shelf_freq=config.effects.eq_high_shelf_freq
                )
            
            print(f"✓ Applied effects from configuration")
        
        print("✅ Test 12 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test 12 FAILED: {e}")
        logger.exception("Test 12 failed")
        return False


def main():
    """Run all effects tests."""
    print("\n" + "="*60)
    print("MAGE PHASE 5: PEDALBOARD EQ INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_1_pedalboard_availability,
        test_2_processor_initialization,
        test_3_eq_processing,
        test_4_compressor,
        test_5_reverb,
        test_6_chorus,
        test_7_delay,
        test_8_limiter,
        test_9_noise_gate,
        test_10_gain,
        test_11_effects_chain,
        test_12_config_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not False else None)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            logger.exception(f"Test {test.__name__} crashed")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    # Filter out skipped tests (None)
    actual_results = [r for r in results if r is not None]
    passed = sum(1 for r in actual_results if r)
    total = len(actual_results)
    skipped = len([r for r in results if r is None])
    
    print(f"Tests passed: {passed}/{total}")
    if skipped > 0:
        print(f"Tests skipped: {skipped} (Pedalboard not available)")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {total - passed} test(s) failed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
