"""Test script for stem separation functionality."""

import numpy as np
import soundfile as sf
from pathlib import Path

from mage.stems import DemucsSeparator, StemManager, StemType
from mage.config import Config

print("=" * 70)
print("Testing MAGE Stem Separation (Phase 3)")
print("=" * 70)

# Create a test audio file
print("\n0. Creating test audio file...")
test_audio_path = Path("output/test_audio.wav")
test_audio_path.parent.mkdir(parents=True, exist_ok=True)

# Generate simple test audio (2 seconds, stereo, 44.1kHz)
duration = 2.0
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration))

# Create stereo test signal (sine waves)
left = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4)
right = np.sin(2 * np.pi * 554 * t)  # 554 Hz (C#5)
test_audio = np.stack([left, right], axis=1)

sf.write(test_audio_path, test_audio, sample_rate)
print(f"   Created test audio: {test_audio_path}")

# Test 1: Initialize DemucsSeparator
print("\n1. Initializing DemucsSeparator...")
try:
    separator = DemucsSeparator(model_name="htdemucs", device="cpu")
    print("   ✓ DemucsSeparator initialized successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Separate stems
print("\n2. Separating stems...")
try:
    stems = separator.separate(
        test_audio_path,
        output_dir="output/stems/test"
    )
    print("   ✓ Stems separated successfully")
    print(f"   Available stems: {[s.value for s in stems.available_stems()]}")
    
    for stem_type in stems.available_stems():
        stem_audio = stems.get_stem(stem_type)
        print(f"   - {stem_type.value}: shape={stem_audio.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Test StemManager
print("\n3. Testing StemManager...")
try:
    manager = StemManager(cache_dir="output/stems/cache")
    print("   ✓ StemManager initialized")
    
    # Get cache stats
    stats = manager.get_cache_stats()
    print(f"   Cache stats: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test caching
print("\n4. Testing stem caching...")
try:
    # First separation (not cached)
    print("   First separation (no cache)...")
    stems1 = manager.separate_with_cache(test_audio_path, separator)
    print("   ✓ First separation complete")
    
    # Second separation (should use cache)
    print("   Second separation (using cache)...")
    stems2 = manager.separate_with_cache(test_audio_path, separator)
    print("   ✓ Second separation complete (from cache)")
    
    # Verify same data
    if stems1.vocals is not None and stems2.vocals is not None:
        if np.allclose(stems1.vocals, stems2.vocals):
            print("   ✓ Cached data matches original")
        else:
            print("   ✗ Cached data doesn't match!")
    
    # Check cache stats again
    stats = manager.get_cache_stats()
    print(f"   Cache stats: {stats['total_files']} files, {stats['total_size_mb']:.2f} MB")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Mix stems
print("\n5. Testing stem mixing...")
try:
    # Mix all stems
    mixed_all = stems.mix_stems()
    print(f"   ✓ Mixed all stems: shape={mixed_all.shape}")
    
    # Mix specific stems
    mixed_partial = stems.mix_stems([StemType.VOCALS, StemType.DRUMS])
    print(f"   ✓ Mixed vocals+drums: shape={mixed_partial.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Config integration
print("\n6. Testing config integration...")
try:
    config = Config()
    print(f"   Stems model: {config.stems.model_name}")
    print(f"   Stems device: {config.stems.device}")
    print(f"   Resolved device: {config.stems.get_device()}")
    print(f"   Cache dir: {config.stems.cache_dir}")
    print(f"   Output dir: {config.stems.output_dir}")
    print(f"   Use cache: {config.stems.use_cache}")
    print("   ✓ Config integration working")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 7: Clear cache
print("\n7. Testing cache management...")
try:
    initial_stats = manager.get_cache_stats()
    print(f"   Initial cache: {initial_stats['total_files']} files")
    
    manager.clear_cache(test_audio_path)
    
    cleared_stats = manager.get_cache_stats()
    print(f"   After clear: {cleared_stats['total_files']} files")
    print("   ✓ Cache cleared successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("Stem Separation Test Complete")
print("=" * 70)
