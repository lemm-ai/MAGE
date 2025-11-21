"""Example: Stem separation with MAGE.

This example demonstrates how to use the DemucsSeparator and StemManager
to separate audio into individual stems.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from mage.stems import DemucsSeparator, StemManager, StemType
from mage.config import Config


def create_demo_audio():
    """Create a demo audio file for testing."""
    output_path = Path("output/demo_song.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 seconds of demo audio
    duration = 5.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate different instruments
    vocals = np.sin(2 * np.pi * 440 * t) * 0.3  # A4
    bass = np.sin(2 * np.pi * 110 * t) * 0.4    # A2
    drums = np.random.randn(len(t)) * 0.2        # Noise
    other = np.sin(2 * np.pi * 880 * t) * 0.25   # A5
    
    # Mix together
    mixed = vocals + bass + drums + other
    
    # Create stereo
    stereo = np.stack([mixed, mixed], axis=1)
    
    sf.write(output_path, stereo, sample_rate)
    print(f"Created demo audio: {output_path}")
    
    return output_path


def main():
    print("MAGE Stem Separation Example")
    print("=" * 60)
    
    # Create demo audio
    print("\n1. Creating Demo Audio")
    print("-" * 60)
    demo_path = create_demo_audio()
    
    # Example 1: Basic stem separation
    print("\n2. Basic Stem Separation")
    print("-" * 60)
    
    separator = DemucsSeparator(model_name="htdemucs", device="cpu")
    
    stems = separator.separate(
        demo_path,
        output_dir="output/stems/demo"
    )
    
    print(f"\nSeparated {len(stems.available_stems())} stems:")
    for stem_type in stems.available_stems():
        stem_audio = stems.get_stem(stem_type)
        print(f"  {stem_type.value}: {stem_audio.shape}")
    
    print(f"\nStems saved to: output/stems/demo/")
    
    # Example 2: Using StemManager with caching
    print("\n3. Stem Separation with Caching")
    print("-" * 60)
    
    manager = StemManager(cache_dir="output/stems/cache")
    
    print("First separation (not cached)...")
    stems1 = manager.separate_with_cache(demo_path, separator)
    print(f"Separated {len(stems1.available_stems())} stems")
    
    print("\nSecond separation (using cache)...")
    stems2 = manager.separate_with_cache(demo_path, separator)
    print(f"Retrieved {len(stems2.available_stems())} stems from cache")
    
    # Show cache stats
    stats = manager.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Files cached: {stats['total_files']}")
    print(f"  Cache size: {stats['total_size_mb']:.2f} MB")
    print(f"  Cache location: {stats['cache_dir']}")
    
    # Example 3: Working with individual stems
    print("\n4. Working with Individual Stems")
    print("-" * 60)
    
    # Get vocals only
    vocals = stems.get_stem(StemType.VOCALS)
    if vocals is not None:
        print(f"Vocals stem: {vocals.shape}")
        print(f"  Duration: {vocals.shape[-1] / stems.sample_rate:.2f} seconds")
        print(f"  RMS level: {np.sqrt(np.mean(vocals**2)):.4f}")
    
    # Get drums only
    drums = stems.get_stem(StemType.DRUMS)
    if drums is not None:
        print(f"\nDrums stem: {drums.shape}")
        print(f"  Duration: {drums.shape[-1] / stems.sample_rate:.2f} seconds")
        print(f"  RMS level: {np.sqrt(np.mean(drums**2)):.4f}")
    
    # Example 4: Mixing stems
    print("\n5. Mixing Stems")
    print("-" * 60)
    
    # Mix all stems back together
    full_mix = stems.mix_stems()
    print(f"Full mix: {full_mix.shape}")
    
    # Create custom mix (vocals + drums only)
    vocal_drum_mix = stems.mix_stems([StemType.VOCALS, StemType.DRUMS])
    print(f"Vocals + Drums mix: {vocal_drum_mix.shape}")
    
    # Save custom mix
    custom_mix_path = Path("output/stems/custom_mix.wav")
    custom_mix_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct shape for saving
    if len(vocal_drum_mix.shape) == 2:
        vocal_drum_mix = vocal_drum_mix.T
    
    sf.write(custom_mix_path, vocal_drum_mix, stems.sample_rate)
    print(f"Saved custom mix to: {custom_mix_path}")
    
    # Example 5: Configuration-based separation
    print("\n6. Using Configuration")
    print("-" * 60)
    
    config = Config.from_file("config/config.yaml")
    
    print(f"Stems configuration:")
    print(f"  Model: {config.stems.model_name}")
    print(f"  Device: {config.stems.get_device()}")
    print(f"  Cache enabled: {config.stems.use_cache}")
    print(f"  Output directory: {config.stems.output_dir}")
    
    # Create separator from config
    config_separator = DemucsSeparator(
        model_name=config.stems.model_name,
        device=config.stems.get_device(),
        cache_dir=config.stems.cache_dir
    )
    
    config_stems = config_separator.separate(
        demo_path,
        output_dir=config.stems.output_dir
    )
    
    print(f"\nSeparated using config: {len(config_stems.available_stems())} stems")
    
    # Example 6: Cache management
    print("\n7. Cache Management")
    print("-" * 60)
    
    stats_before = manager.get_cache_stats()
    print(f"Cache before: {stats_before['total_files']} files, {stats_before['total_size_mb']:.2f} MB")
    
    # Clear cache for specific file
    manager.clear_cache(demo_path)
    
    stats_after = manager.get_cache_stats()
    print(f"Cache after clear: {stats_after['total_files']} files, {stats_after['total_size_mb']:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
    
    print("\nOutput files:")
    print(f"  Demo audio: {demo_path}")
    print(f"  Separated stems: output/stems/demo/")
    print(f"  Custom mix: {custom_mix_path}")
    print(f"  Cached stems: {stats['cache_dir']}")


if __name__ == "__main__":
    main()
