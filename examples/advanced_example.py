"""Example: Advanced audio generation with custom configuration and effects.

This example demonstrates advanced features including:
- Custom configuration
- Multiple generation styles
- Audio effects
- Batch generation
"""

from pathlib import Path
from mage import MAGE, Config

def main():
    """Generate audio with advanced features."""
    # Create custom configuration
    config = Config()
    config.audio.sample_rate = 48000
    config.audio.default_duration = 20.0
    config.generation.complexity = 0.7
    config.logging.level = "DEBUG"
    
    # Initialize engine with custom config
    print("Initializing MAGE engine with custom configuration...")
    engine = MAGE(config=config)
    
    # Get available styles
    styles = engine.get_available_styles()
    print(f"\nAvailable styles: {', '.join(styles)}")
    
    # Generate multiple variations
    output_dir = Path("examples/output/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variations = [
        {"style": "ambient", "tempo": 60, "complexity": 0.3},
        {"style": "electronic", "tempo": 128, "complexity": 0.7},
        {"style": "orchestral", "tempo": 90, "complexity": 0.8},
    ]
    
    for i, params in enumerate(variations, 1):
        print(f"\n[{i}/{len(variations)}] Generating {params['style']} variation...")
        
        # Generate audio
        audio = engine.generate(
            duration=15.0,
            style=params["style"],
            tempo=params["tempo"],
            complexity=params["complexity"],
            key="C_major",
            seed=42  # For reproducibility
        )
        
        # Apply effects
        print("  Applying effects...")
        audio.apply_effects(
            reverb=0.3,
            compression=0.4
        )
        
        # Normalize
        audio.normalize(target_level=-6.0)
        
        # Export
        filename = f"{params['style']}_t{params['tempo']}.wav"
        output_path = output_dir / filename
        audio.export(output_path)
        
        print(f"  ✓ Saved to {output_path}")
    
    print("\n✓ All variations generated successfully!")


if __name__ == "__main__":
    main()
