"""Example: Basic audio generation with MAGE.

This example demonstrates the simplest way to generate audio using MAGE.
"""

from mage import MAGE

def main():
    """Generate a simple audio file."""
    # Initialize the engine
    print("Initializing MAGE engine...")
    engine = MAGE()
    
    # Generate audio
    print("Generating audio...")
    audio = engine.generate(
        duration=10.0,
        style="ambient",
        tempo=80
    )
    
    # Export the result
    output_file = "examples/output/basic_example.wav"
    print(f"Exporting to {output_file}...")
    audio.export(output_file)
    
    print("✓ Done!")
    print(f"Generated {audio.duration:.2f} seconds of audio")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
