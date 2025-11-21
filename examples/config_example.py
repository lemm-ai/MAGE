"""Example: Using custom configuration files.

This example shows how to use YAML configuration files for MAGE.
"""

from pathlib import Path
from mage import MAGE, Config

def main():
    """Generate audio using a custom configuration file."""
    # Load configuration from file
    config_path = Path("examples/configs/custom_config.yaml")
    
    if not config_path.exists():
        print("Creating example configuration file...")
        create_example_config(config_path)
    
    print(f"Loading configuration from {config_path}...")
    config = Config.from_file(config_path)
    
    # Initialize engine
    print("Initializing MAGE engine...")
    engine = MAGE(config=config)
    
    # Generate audio
    print("Generating audio with custom settings...")
    audio = engine.generate(
        style="jazz",
        tempo=120
    )
    
    # Apply effects and export
    audio.apply_effects(reverb=0.2, compression=0.3)
    audio.normalize()
    
    output_file = "examples/output/config_example.wav"
    audio.export(output_file)
    
    print(f"✓ Audio generated and saved to {output_file}")


def create_example_config(config_path: Path):
    """Create an example configuration file.
    
    Args:
        config_path: Path where to save the config
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create custom configuration
    config = Config()
    config.audio.sample_rate = 48000
    config.audio.default_duration = 25.0
    config.generation.default_style = "jazz"
    config.generation.default_tempo = 120
    config.generation.complexity = 0.6
    config.logging.level = "INFO"
    
    # Save to file
    config.to_file(config_path)
    print(f"Created example config at {config_path}")


if __name__ == "__main__":
    main()
