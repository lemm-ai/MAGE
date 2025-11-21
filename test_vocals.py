"""Test script for vocal enhancement module.

This script tests the VocalEnhancer class with comprehensive examples.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from mage.vocals import VocalEnhancer, is_enhancement_available
from mage.config import Config

print("="*60)
print("MAGE PHASE 6: VOCAL ENHANCEMENT TESTS")
print("="*60)

# Test 1: Check availability
print("\n" + "="*60)
print("TEST 1: Check Vocal Enhancement Availability")
print("="*60)
available = is_enhancement_available()
print(f"Vocal enhancement available: {available}")
if not available:
    print("⚠️  Required dependencies not installed")
    print("   Install: pip install torch torchaudio librosa")
    exit(1)
print("✅ Test 1 PASSED")

# Create test audio
sample_rate = 44100
duration = 3.0
num_samples = int(sample_rate * duration)

# Generate synthetic vocal-like audio with noise
np.random.seed(42)
t = np.linspace(0, duration, num_samples)

# Fundamental frequency (vocal formant)
f0 = 200  # Hz
vocal = 0.3 * np.sin(2 * np.pi * f0 * t)

# Add harmonics
vocal += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
vocal += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)

# Add formant resonance
f1 = 800  # First formant
vocal += 0.08 * np.sin(2 * np.pi * f1 * t)

# Add noise (simulated background)
noise = 0.05 * np.random.randn(num_samples)
noisy_vocal = vocal + noise

# Test 2: Initialize enhancer
print("\n" + "="*60)
print("TEST 2: Initialize VocalEnhancer")
print("="*60)
try:
    enhancer = VocalEnhancer(sample_rate=sample_rate, device="cpu")
    print("✓ VocalEnhancer initialized")
    print(f"  - Sample rate: {enhancer.sample_rate} Hz")
    print(f"  - Device: {enhancer._device}")
    print(f"  - Cache dir: {enhancer.cache_dir}")
    print("✅ Test 2 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 3: Denoise vocal
print("\n" + "="*60)
print("TEST 3: Denoise Vocal Audio")
print("="*60)
try:
    denoised = enhancer.denoise(noisy_vocal, noise_reduction=0.8)
    print("✓ Denoising completed")
    print(f"  - Input shape: {noisy_vocal.shape}")
    print(f"  - Output shape: {denoised.shape}")
    print(f"  - Input RMS: {np.sqrt(np.mean(noisy_vocal**2)):.4f}")
    print(f"  - Output RMS: {np.sqrt(np.mean(denoised**2)):.4f}")
    
    # Calculate noise reduction
    input_noise_est = np.std(noisy_vocal - vocal)
    output_noise_est = np.std(denoised - vocal)
    noise_reduction_pct = (1 - output_noise_est / input_noise_est) * 100
    print(f"  - Noise reduction: {noise_reduction_pct:.1f}%")
    print("✅ Test 3 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 4: Spectral enhancement
print("\n" + "="*60)
print("TEST 4: Spectral Enhancement")
print("="*60)
try:
    enhanced = enhancer.enhance_spectral(
        vocal,
        brightness=0.3,
        warmth=0.2,
        clarity=0.4
    )
    print("✓ Spectral enhancement completed")
    print(f"  - Input shape: {vocal.shape}")
    print(f"  - Output shape: {enhanced.shape}")
    print(f"  - Input peak: {np.max(np.abs(vocal)):.3f}")
    print(f"  - Output peak: {np.max(np.abs(enhanced)):.3f}")
    print("✅ Test 4 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 5: Dynamic optimization
print("\n" + "="*60)
print("TEST 5: Dynamic Optimization")
print("="*60)
try:
    # Create audio with varying dynamics
    dynamic_audio = vocal.copy()
    dynamic_audio[:len(dynamic_audio)//3] *= 0.2  # Quiet section
    dynamic_audio[len(dynamic_audio)//3:2*len(dynamic_audio)//3] *= 0.8  # Medium
    dynamic_audio[2*len(dynamic_audio)//3:] *= 1.2  # Loud section (will clip)
    
    optimized = enhancer.optimize_dynamics(
        dynamic_audio,
        target_level=-20.0,
        compression_ratio=4.0
    )
    print("✓ Dynamic optimization completed")
    print(f"  - Input RMS: {20*np.log10(np.sqrt(np.mean(dynamic_audio**2)) + 1e-10):.2f} dB")
    print(f"  - Output RMS: {20*np.log10(np.sqrt(np.mean(optimized**2)) + 1e-10):.2f} dB")
    print(f"  - Input peak: {np.max(np.abs(dynamic_audio)):.3f}")
    print(f"  - Output peak: {np.max(np.abs(optimized)):.3f}")
    print("✅ Test 5 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 6: Full enhancement pipeline
print("\n" + "="*60)
print("TEST 6: Full Enhancement Pipeline")
print("="*60)
try:
    fully_enhanced = enhancer.enhance(
        noisy_vocal,
        denoise_strength=0.7,
        brightness=0.2,
        warmth=0.1,
        clarity=0.3,
        target_level=-18.0
    )
    print("✓ Full enhancement pipeline completed")
    print(f"  - Input shape: {noisy_vocal.shape}")
    print(f"  - Output shape: {fully_enhanced.shape}")
    print(f"  - Input RMS: {20*np.log10(np.sqrt(np.mean(noisy_vocal**2)) + 1e-10):.2f} dB")
    print(f"  - Output RMS: {20*np.log10(np.sqrt(np.mean(fully_enhanced**2)) + 1e-10):.2f} dB")
    print("✅ Test 6 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 7: Stereo processing
print("\n" + "="*60)
print("TEST 7: Stereo Audio Processing")
print("="*60)
try:
    # Create stereo audio
    stereo_vocal = np.stack([noisy_vocal, noisy_vocal * 0.9], axis=0)
    
    stereo_enhanced = enhancer.enhance(
        stereo_vocal,
        denoise_strength=0.6,
        brightness=0.15,
        warmth=0.1,
        clarity=0.25,
        target_level=-20.0
    )
    print("✓ Stereo enhancement completed")
    print(f"  - Input shape: {stereo_vocal.shape}")
    print(f"  - Output shape: {stereo_enhanced.shape}")
    print(f"  - Left channel RMS: {np.sqrt(np.mean(stereo_enhanced[0]**2)):.4f}")
    print(f"  - Right channel RMS: {np.sqrt(np.mean(stereo_enhanced[1]**2)):.4f}")
    print("✅ Test 7 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 8: Configuration integration
print("\n" + "="*60)
print("TEST 8: Configuration Integration")
print("="*60)
try:
    config = Config.from_file("config/config.yaml")
    print("✓ Loaded vocal configuration:")
    print(f"  - Denoise strength: {config.vocals.denoise_strength}")
    print(f"  - Brightness: {config.vocals.brightness}")
    print(f"  - Warmth: {config.vocals.warmth}")
    print(f"  - Clarity: {config.vocals.clarity}")
    print(f"  - Target level: {config.vocals.target_level_db} dB")
    print(f"  - Compression ratio: {config.vocals.compression_ratio}")
    print(f"  - Device: {config.vocals.device}")
    print(f"  - Cache dir: {config.vocals.cache_dir}")
    
    # Apply config-based enhancement
    config_enhanced = enhancer.enhance(
        noisy_vocal,
        denoise_strength=config.vocals.denoise_strength,
        brightness=config.vocals.brightness,
        warmth=config.vocals.warmth,
        clarity=config.vocals.clarity,
        target_level=config.vocals.target_level_db
    )
    print("✓ Config-based enhancement completed")
    print("✅ Test 8 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 9: Parameter validation
print("\n" + "="*60)
print("TEST 9: Parameter Validation")
print("="*60)
try:
    # Test invalid noise reduction
    try:
        enhancer.denoise(vocal, noise_reduction=1.5)
        print("✗ Should have raised InvalidParameterError")
        exit(1)
    except Exception as e:
        print(f"✓ Caught invalid noise_reduction: {type(e).__name__}")
    
    # Test invalid brightness
    try:
        enhancer.enhance_spectral(vocal, brightness=2.0)
        print("✗ Should have raised InvalidParameterError")
        exit(1)
    except Exception as e:
        print(f"✓ Caught invalid brightness: {type(e).__name__}")
    
    # Test invalid target level
    try:
        enhancer.optimize_dynamics(vocal, target_level=10.0)
        print("✗ Should have raised InvalidParameterError")
        exit(1)
    except Exception as e:
        print(f"✓ Caught invalid target_level: {type(e).__name__}")
    
    print("✅ Test 9 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 10: Save enhanced audio
print("\n" + "="*60)
print("TEST 10: Save Enhanced Audio")
print("="*60)
try:
    output_dir = Path("output/vocals")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mono enhanced
    mono_path = output_dir / "enhanced_mono.wav"
    sf.write(mono_path, fully_enhanced, sample_rate)
    print(f"✓ Saved mono enhanced audio: {mono_path}")
    
    # Save stereo enhanced
    stereo_path = output_dir / "enhanced_stereo.wav"
    sf.write(stereo_path, stereo_enhanced.T, sample_rate)
    print(f"✓ Saved stereo enhanced audio: {stereo_path}")
    
    # Save comparison (original vs enhanced)
    comparison_path = output_dir / "comparison.wav"
    comparison = np.stack([noisy_vocal, fully_enhanced], axis=0)
    sf.write(comparison_path, comparison.T, sample_rate)
    print(f"✓ Saved comparison audio: {comparison_path}")
    
    print("✅ Test 10 PASSED")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
tests_passed = 10
tests_total = 10
print(f"Tests passed: {tests_passed}/{tests_total}")
if tests_passed == tests_total:
    print("✅ ALL TESTS PASSED!")
else:
    print(f"⚠️  {tests_total - tests_passed} test(s) failed")
print("="*60)
