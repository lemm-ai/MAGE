#!/usr/bin/env python3
"""
Examples for Audio Effects Processing with Pedalboard

This file demonstrates professional audio effects capabilities:
1. Basic EQ
2. Compression
3. Reverb
4. Modulation effects (chorus, delay)
5. Dynamics (limiter, noise gate)
6. Effects chains
7. Configuration-based processing
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mage.processors import EffectsProcessor, is_pedalboard_available
from mage.config import Config
from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


def generate_tone(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a simple sine wave tone."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def example_1_basic_eq():
    """Example 1: Apply basic 3-band EQ"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic EQ")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available. Install with: pip install pedalboard")
        return
    
    # Create processor
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(440.0, 5.0)
    
    # Apply EQ boost bass, cut highs
    processed = processor.apply_eq(
        audio,
        low_shelf_gain_db=3.0,      # Boost bass by 3dB
        low_shelf_freq=100.0,
        mid_gain_db=0.0,            # No mid change
        mid_freq=1000.0,
        high_shelf_gain_db=-2.0,    # Cut highs by 2dB
        high_shelf_freq=8000.0
    )
    
    print(f"Applied EQ:")
    print(f"  Low shelf: +3.0dB @ 100Hz")
    print(f"  Mid: 0.0dB @ 1kHz")
    print(f"  High shelf: -2.0dB @ 8kHz")
    print(f"  Input peak: {np.max(np.abs(audio)):.3f}")
    print(f"  Output peak: {np.max(np.abs(processed)):.3f}")


def example_2_vocal_compression():
    """Example 2: Compress vocals for consistency"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Vocal Compression")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio with varying levels
    audio = generate_tone(440.0, 5.0)
    audio = audio * (1.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, len(audio))))  # Add amplitude variation
    
    # Apply gentle vocal compression
    processed = processor.apply_compressor(
        audio,
        threshold_db=-18.0,  # Catch peaks above -18dB
        ratio=3.0,           # 3:1 compression ratio
        attack_ms=10.0,      # Relatively slow attack
        release_ms=200.0     # Longer release for natural sound
    )
    
    print(f"Vocal compression settings:")
    print(f"  Threshold: -18dB")
    print(f"  Ratio: 3:1")
    print(f"  Attack: 10ms (preserves transients)")
    print(f"  Release: 200ms (smooth)")
    print(f"  Dynamic range reduced: {(1 - (np.std(processed)/np.std(audio)))*100:.1f}%")


def example_3_hall_reverb():
    """Example 3: Add concert hall reverb"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Hall Reverb")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(440.0, 3.0)
    
    # Apply large hall reverb
    processed = processor.apply_reverb(
        audio,
        room_size=0.9,       # Large hall
        damping=0.3,         # Less damping for brighter sound
        wet_level=0.4,       # 40% reverb
        dry_level=0.6,       # 60% dry signal
        width=1.0            # Full stereo width
    )
    
    print(f"Hall reverb settings:")
    print(f"  Room size: 0.9 (large hall)")
    print(f"  Damping: 0.3 (bright)")
    print(f"  Wet/Dry: 40%/60%")
    print(f"  Width: 1.0 (stereo)")


def example_4_chorus_effect():
    """Example 4: Thicken sound with chorus"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Chorus Effect")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(330.0, 4.0)
    
    # Apply chorus
    processed = processor.apply_chorus(
        audio,
        rate_hz=1.2,             # LFO rate
        depth=0.35,              # Modulation depth
        centre_delay_ms=7.0,     # Center delay
        feedback=0.1,            # Slight feedback
        mix=0.5                  # 50/50 mix
    )
    
    print(f"Chorus settings:")
    print(f"  Rate: 1.2Hz")
    print(f"  Depth: 0.35")
    print(f"  Delay: 7ms")
    print(f"  Mix: 50%")


def example_5_delay_effect():
    """Example 5: Add rhythmic delay"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Delay Effect")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(440.0, 3.0)
    
    # Calculate delay time for 120 BPM eighth notes
    bpm = 120
    eighth_note_seconds = 60.0 / bpm / 2  # 0.25s
    
    # Apply rhythmic delay
    processed = processor.apply_delay(
        audio,
        delay_seconds=eighth_note_seconds,
        feedback=0.4,        # 40% feedback (a few repeats)
        mix=0.3              # 30% wet signal
    )
    
    print(f"Rhythmic delay (120 BPM eighth notes):")
    print(f"  Delay time: {eighth_note_seconds}s")
    print(f"  Feedback: 0.4 (multiple repeats)")
    print(f"  Mix: 30%")


def example_6_mastering_limiter():
    """Example 6: Prevent clipping with limiter"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Mastering Limiter")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate loud audio that might clip
    audio = generate_tone(440.0, 3.0) * 3.5
    
    # Apply limiter
    processed = processor.apply_limiter(
        audio,
        threshold_db=-0.5,   # Limit at -0.5dB
        release_ms=100.0     # Fast release
    )
    
    print(f"Mastering limiter:")
    print(f"  Threshold: -0.5dB")
    print(f"  Release: 100ms")
    print(f"  Input peak: {np.max(np.abs(audio)):.3f} (would clip!)")
    print(f"  Output peak: {np.max(np.abs(processed)):.3f} (safe)")


def example_7_noise_gate():
    """Example 7: Remove background noise"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Noise Gate")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio with noise
    audio = generate_tone(440.0, 3.0)
    noise = np.random.randn(len(audio)) * 0.05
    audio_noisy = audio + noise
    
    # Apply noise gate
    processed = processor.apply_noise_gate(
        audio_noisy,
        threshold_db=-35.0,  # Gate opens above -35dB
        ratio=8.0,           # 8:1 ratio
        attack_ms=1.0,       # Fast attack
        release_ms=150.0     # Medium release
    )
    
    print(f"Noise gate settings:")
    print(f"  Threshold: -35dB")
    print(f"  Ratio: 8:1")
    print(f"  Attack: 1ms (fast)")
    print(f"  Release: 150ms")
    print(f"  Noise reduced: {(1 - np.std(processed)/np.std(audio_noisy))*100:.1f}%")


def example_8_vocal_chain():
    """Example 8: Complete vocal processing chain"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Vocal Processing Chain")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(440.0, 5.0)
    
    # Build vocal processing chain
    chain = [
        # 1. High-pass filter (remove rumble)
        {"type": "eq", "low_shelf_gain_db": -12.0, "low_shelf_freq": 80.0},
        
        # 2. De-esser (reduce harsh highs)
        {"type": "eq", "high_shelf_gain_db": -3.0, "high_shelf_freq": 7000.0},
        
        # 3. Compression (even out dynamics)
        {"type": "compressor", "threshold_db": -18.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 200.0},
        
        # 4. EQ boost (add presence)
        {"type": "eq", "mid_gain_db": 2.0, "mid_freq": 3000.0, "mid_q": 1.5},
        
        # 5. Reverb (add space)
        {"type": "reverb", "room_size": 0.6, "wet_level": 0.25, "dry_level": 0.75},
        
        # 6. Limiter (prevent clipping)
        {"type": "limiter", "threshold_db": -1.0, "release_ms": 100.0}
    ]
    
    # Apply chain
    processed = processor.apply_chain(audio, chain)
    
    print(f"Vocal chain ({len(chain)} stages):")
    for i, effect in enumerate(chain, 1):
        print(f"  {i}. {effect['type'].upper()}")
    
    print(f"\nResults:")
    print(f"  Input peak: {np.max(np.abs(audio)):.3f}")
    print(f"  Output peak: {np.max(np.abs(processed)):.3f}")


def example_9_mastering_chain():
    """Example 9: Mastering effects chain"""
    print("\n" + "="*60)
    print("EXAMPLE 9: Mastering Chain")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    processor = EffectsProcessor(sample_rate=44100)
    
    # Generate audio
    audio = generate_tone(440.0, 5.0) * 0.5
    
    # Build mastering chain
    chain = [
        # 1. Gentle EQ shaping
        {"type": "eq", "low_shelf_gain_db": 1.0, "low_shelf_freq": 100.0,
         "high_shelf_gain_db": 0.5, "high_shelf_freq": 10000.0},
        
        # 2. Multiband compression (simplified with single compressor)
        {"type": "compressor", "threshold_db": -15.0, "ratio": 2.0, 
         "attack_ms": 5.0, "release_ms": 150.0},
        
        # 3. Final limiter for loudness
        {"type": "limiter", "threshold_db": -0.3, "release_ms": 100.0}
    ]
    
    # Apply chain
    processed = processor.apply_chain(audio, chain)
    
    print(f"Mastering chain:")
    print(f"  1. EQ - Gentle low/high enhancement")
    print(f"  2. Compressor - Glue the mix")
    print(f"  3. Limiter - Maximize loudness")
    print(f"\nLoudness increase: {20*np.log10(np.sqrt(np.mean(processed**2))/np.sqrt(np.mean(audio**2))):.1f} dB")


def example_10_config_based():
    """Example 10: Apply effects from configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 10: Configuration-Based Processing")
    print("="*60)
    
    if not is_pedalboard_available():
        print("⚠️  Pedalboard not available")
        return
    
    # Load config
    config = Config.from_file("config/config.yaml")
    
    # Create processor from config
    processor = EffectsProcessor(sample_rate=config.audio.sample_rate)
    
    # Generate audio
    audio = generate_tone(440.0, 3.0)
    
    # Apply enabled effects from config
    processed = audio.copy()
    
    if config.effects.enable_eq:
        processed = processor.apply_eq(
            processed,
            low_shelf_gain_db=config.effects.eq_low_shelf_gain_db,
            low_shelf_freq=config.effects.eq_low_shelf_freq,
            mid_gain_db=config.effects.eq_mid_gain_db,
            mid_freq=config.effects.eq_mid_freq,
            high_shelf_gain_db=config.effects.eq_high_shelf_gain_db,
            high_shelf_freq=config.effects.eq_high_shelf_freq
        )
        print("✓ Applied EQ from config")
    
    if config.effects.enable_compressor:
        processed = processor.apply_compressor(
            processed,
            threshold_db=config.effects.comp_threshold_db,
            ratio=config.effects.comp_ratio,
            attack_ms=config.effects.comp_attack_ms,
            release_ms=config.effects.comp_release_ms
        )
        print("✓ Applied compressor from config")
    
    if config.effects.enable_reverb:
        processed = processor.apply_reverb(
            processed,
            room_size=config.effects.reverb_room_size,
            damping=config.effects.reverb_damping,
            wet_level=config.effects.reverb_wet_level,
            dry_level=config.effects.reverb_dry_level,
            width=config.effects.reverb_width
        )
        print("✓ Applied reverb from config")
    
    if config.effects.enable_limiter:
        processed = processor.apply_limiter(
            processed,
            threshold_db=config.effects.limiter_threshold_db,
            release_ms=config.effects.limiter_release_ms
        )
        print("✓ Applied limiter from config")
    
    print(f"\nProcessing complete!")
    print(f"  Effects enabled: EQ={config.effects.enable_eq}, "
          f"Comp={config.effects.enable_compressor}, "
          f"Reverb={config.effects.enable_reverb}, "
          f"Limiter={config.effects.enable_limiter}")


def main():
    """Run all examples."""
    print("="*60)
    print("MAGE AUDIO EFFECTS EXAMPLES")
    print("="*60)
    
    examples = [
        example_1_basic_eq,
        example_2_vocal_compression,
        example_3_hall_reverb,
        example_4_chorus_effect,
        example_5_delay_effect,
        example_6_mastering_limiter,
        example_7_noise_gate,
        example_8_vocal_chain,
        example_9_mastering_chain,
        example_10_config_based
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
