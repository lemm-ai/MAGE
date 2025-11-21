"""Example: Vocal enhancement with MAGE.

This example demonstrates how to use the VocalEnhancer class
for improving vocal audio quality.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from mage.vocals import VocalEnhancer, is_enhancement_available
from mage.config import Config


def create_demo_vocal():
    """Create a demo vocal audio file for testing."""
    sample_rate = 44100
    duration = 5.0
    num_samples = int(sample_rate * duration)
    
    # Generate synthetic vocal
    t = np.linspace(0, duration, num_samples)
    
    # Fundamental frequency (A4 = 440 Hz)
    f0 = 440
    vocal = 0.4 * np.sin(2 * np.pi * f0 * t)
    
    # Add harmonics for richness
    vocal += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
    vocal += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    vocal += 0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    
    # Add formants
    f1 = 800  # First formant
    f2 = 1200  # Second formant
    vocal += 0.1 * np.sin(2 * np.pi * f1 * t)
    vocal += 0.05 * np.sin(2 * np.pi * f2 * t)
    
    # Add background noise
    noise = 0.08 * np.random.randn(num_samples)
    noisy_vocal = vocal + noise
    
    # Add some variations
    envelope = np.sin(2 * np.pi * 0.5 * t) * 0.3 + 0.7
    noisy_vocal *= envelope
    
    return noisy_vocal, sample_rate


def main():
    """Run vocal enhancement examples."""
    
    print("\n" + "=" * 60)
    print("MAGE Vocal Enhancement Examples")
    print("=" * 60)
    
    # Check availability
    if not is_enhancement_available():
        print("\n⚠️  Vocal enhancement not available")
        print("Install: pip install torch torchaudio librosa")
        return
    
    # Create demo audio
    print("\n1. Creating Demo Vocal Audio")
    print("-" * 60)
    demo_vocal, sample_rate = create_demo_vocal()
    print(f"Created demo vocal: {len(demo_vocal)} samples @ {sample_rate} Hz")
    print(f"Duration: {len(demo_vocal) / sample_rate:.2f} seconds")
    print(f"RMS level: {20 * np.log10(np.sqrt(np.mean(demo_vocal**2)) + 1e-10):.2f} dB")
    
    # Initialize enhancer
    print("\n2. Initialize VocalEnhancer")
    print("-" * 60)
    enhancer = VocalEnhancer(sample_rate=sample_rate)
    print(f"Sample rate: {enhancer.sample_rate} Hz")
    print(f"Device: {enhancer._device}")
    
    # Example 1: Basic denoising
    print("\n3. Example 1: Basic Denoising")
    print("-" * 60)
    denoised = enhancer.denoise(demo_vocal, noise_reduction=0.8)
    print(f"Denoised vocal")
    print(f"  Input RMS: {20 * np.log10(np.sqrt(np.mean(demo_vocal**2)) + 1e-10):.2f} dB")
    print(f"  Output RMS: {20 * np.log10(np.sqrt(np.mean(denoised**2)) + 1e-10):.2f} dB")
    
    # Save
    output_dir = Path("output/vocals/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "example1_denoised.wav", denoised, sample_rate)
    print(f"  Saved: {output_dir / 'example1_denoised.wav'}")
    
    # Example 2: Spectral enhancement for brightness
    print("\n4. Example 2: Brighten Vocals")
    print("-" * 60)
    brightened = enhancer.enhance_spectral(
        demo_vocal,
        brightness=0.5,  # Boost high frequencies
        warmth=0.0,
        clarity=0.3  # Boost presence
    )
    print(f"Brightened vocal")
    print(f"  Brightness: +0.5 (high freq boost)")
    print(f"  Clarity: +0.3 (presence boost)")
    sf.write(output_dir / "example2_brightened.wav", brightened, sample_rate)
    print(f"  Saved: {output_dir / 'example2_brightened.wav'}")
    
    # Example 3: Warm and smooth vocals
    print("\n5. Example 3: Warm & Smooth Vocals")
    print("-" * 60)
    warmed = enhancer.enhance_spectral(
        demo_vocal,
        brightness=-0.2,  # Reduce harshness
        warmth=0.4,  # Boost low-mids
        clarity=0.1
    )
    print(f"Warmed vocal")
    print(f"  Brightness: -0.2 (reduce harshness)")
    print(f"  Warmth: +0.4 (boost low-mids)")
    sf.write(output_dir / "example3_warmed.wav", warmed, sample_rate)
    print(f"  Saved: {output_dir / 'example3_warmed.wav'}")
    
    # Example 4: Dynamic optimization
    print("\n6. Example 4: Optimize Dynamics")
    print("-" * 60)
    # Create dynamic vocal with varying levels
    quiet_part = demo_vocal[:len(demo_vocal)//2] * 0.3
    loud_part = demo_vocal[len(demo_vocal)//2:] * 1.5
    dynamic_vocal = np.concatenate([quiet_part, loud_part])
    
    optimized = enhancer.optimize_dynamics(
        dynamic_vocal,
        target_level=-18.0,
        compression_ratio=4.0
    )
    print(f"Optimized dynamics")
    print(f"  Input RMS: {20 * np.log10(np.sqrt(np.mean(dynamic_vocal**2)) + 1e-10):.2f} dB")
    print(f"  Output RMS: {20 * np.log10(np.sqrt(np.mean(optimized**2)) + 1e-10):.2f} dB")
    print(f"  Compression: 4:1 ratio")
    sf.write(output_dir / "example4_optimized.wav", optimized, sample_rate)
    print(f"  Saved: {output_dir / 'example4_optimized.wav'}")
    
    # Example 5: Full vocal enhancement pipeline
    print("\n7. Example 5: Full Enhancement Pipeline")
    print("-" * 60)
    fully_enhanced = enhancer.enhance(
        demo_vocal,
        denoise_strength=0.7,
        brightness=0.25,
        warmth=0.15,
        clarity=0.35,
        target_level=-16.0
    )
    print(f"Full enhancement applied:")
    print(f"  1. Denoise (70% strength)")
    print(f"  2. Spectral enhance (bright +0.25, warm +0.15, clear +0.35)")
    print(f"  3. Dynamic optimize (-16 dB target)")
    print(f"  Input RMS: {20 * np.log10(np.sqrt(np.mean(demo_vocal**2)) + 1e-10):.2f} dB")
    print(f"  Output RMS: {20 * np.log10(np.sqrt(np.mean(fully_enhanced**2)) + 1e-10):.2f} dB")
    sf.write(output_dir / "example5_full_enhancement.wav", fully_enhanced, sample_rate)
    print(f"  Saved: {output_dir / 'example5_full_enhancement.wav'}")
    
    # Example 6: Podcast vocal enhancement
    print("\n8. Example 6: Podcast Vocal Enhancement")
    print("-" * 60)
    podcast_enhanced = enhancer.enhance(
        demo_vocal,
        denoise_strength=0.9,  # Strong noise reduction
        brightness=0.1,  # Slight brightness
        warmth=0.2,  # Warm tone
        clarity=0.5,  # Strong clarity for intelligibility
        target_level=-20.0  # Comfortable listening level
    )
    print(f"Podcast-optimized vocal:")
    print(f"  - Strong noise reduction (90%)")
    print(f"  - Enhanced clarity for speech intelligibility")
    print(f"  - Warm, natural tone")
    print(f"  - Consistent level (-20 dB)")
    sf.write(output_dir / "example6_podcast.wav", podcast_enhanced, sample_rate)
    print(f"  Saved: {output_dir / 'example6_podcast.wav'}")
    
    # Example 7: Singing vocal enhancement
    print("\n9. Example 7: Singing Vocal Enhancement")
    print("-" * 60)
    singing_enhanced = enhancer.enhance(
        demo_vocal,
        denoise_strength=0.6,  # Moderate noise reduction
        brightness=0.3,  # More brightness for presence
        warmth=0.1,  # Slight warmth
        clarity=0.4,  # Good clarity
        target_level=-14.0  # Louder for music production
    )
    print(f"Singing-optimized vocal:")
    print(f"  - Moderate noise reduction (60%)")
    print(f"  - Enhanced brightness and presence")
    print(f"  - Balanced spectral enhancement")
    print(f"  - Production-ready level (-14 dB)")
    sf.write(output_dir / "example7_singing.wav", singing_enhanced, sample_rate)
    print(f"  Saved: {output_dir / 'example7_singing.wav'}")
    
    # Example 8: Configuration-based enhancement
    print("\n10. Example 8: Configuration-Based Enhancement")
    print("-" * 60)
    config = Config.from_file("config/config.yaml")
    config_enhanced = enhancer.enhance(
        demo_vocal,
        denoise_strength=config.vocals.denoise_strength,
        brightness=config.vocals.brightness,
        warmth=config.vocals.warmth,
        clarity=config.vocals.clarity,
        target_level=config.vocals.target_level_db
    )
    print(f"Enhancement from config:")
    print(f"  - Denoise: {config.vocals.denoise_strength}")
    print(f"  - Brightness: {config.vocals.brightness}")
    print(f"  - Warmth: {config.vocals.warmth}")
    print(f"  - Clarity: {config.vocals.clarity}")
    print(f"  - Target level: {config.vocals.target_level_db} dB")
    sf.write(output_dir / "example8_config_based.wav", config_enhanced, sample_rate)
    print(f"  Saved: {output_dir / 'example8_config_based.wav'}")
    
    # Example 9: Stereo vocal enhancement
    print("\n11. Example 9: Stereo Vocal Enhancement")
    print("-" * 60)
    # Create stereo vocal
    stereo_vocal = np.stack([demo_vocal, demo_vocal * 0.95], axis=0)
    stereo_enhanced = enhancer.enhance(
        stereo_vocal,
        denoise_strength=0.7,
        brightness=0.2,
        warmth=0.15,
        clarity=0.3,
        target_level=-18.0
    )
    print(f"Stereo vocal enhanced:")
    print(f"  Input shape: {stereo_vocal.shape}")
    print(f"  Output shape: {stereo_enhanced.shape}")
    print(f"  Left RMS: {np.sqrt(np.mean(stereo_enhanced[0]**2)):.4f}")
    print(f"  Right RMS: {np.sqrt(np.mean(stereo_enhanced[1]**2)):.4f}")
    sf.write(output_dir / "example9_stereo.wav", stereo_enhanced.T, sample_rate)
    print(f"  Saved: {output_dir / 'example9_stereo.wav'}")
    
    # Example 10: Comparison (before/after)
    print("\n12. Example 10: Before/After Comparison")
    print("-" * 60)
    # Create side-by-side comparison
    comparison = np.stack([demo_vocal, fully_enhanced], axis=0)
    sf.write(output_dir / "example10_comparison.wav", comparison.T, sample_rate)
    print(f"Comparison file created:")
    print(f"  Left channel: Original")
    print(f"  Right channel: Enhanced")
    print(f"  Saved: {output_dir / 'example10_comparison.wav'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"All examples saved to: {output_dir}")
    print("\nEnhancement techniques demonstrated:")
    print("  1. Denoising - Remove background noise")
    print("  2. Spectral enhancement - Adjust frequency response")
    print("  3. Dynamic optimization - Even out levels")
    print("  4. Full pipeline - Combined processing")
    print("  5. Use cases - Podcast, singing, general")
    print("  6. Configuration - Config-based enhancement")
    print("  7. Stereo processing - Multi-channel support")
    print("\nNext steps:")
    print("  - Try with your own vocal recordings")
    print("  - Adjust parameters in config/config.yaml")
    print("  - Integrate with stems separation for isolated vocals")
    print("  - Combine with effects processing for production")
    print("=" * 60)


if __name__ == "__main__":
    main()
