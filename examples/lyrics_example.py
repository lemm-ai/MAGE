"""Example: Generate lyrics with MAGE.

This example demonstrates how to use the LyricGenerator to create
song lyrics with different themes, genres, and structures.
"""

from pathlib import Path
from mage.lyrics import LyricGenerator, LyricConfig
from mage.config import Config

def main():
    print("MAGE Lyrics Generation Example")
    print("=" * 60)
    
    # Example 1: Simple lyrics generation
    print("\n1. Simple Lyrics Generation")
    print("-" * 60)
    
    generator = LyricGenerator()
    
    lyrics = generator.generate(
        theme="summer nights and memories",
        genre="pop",
        max_lines=16
    )
    
    print(f"Genre: {lyrics.genre}")
    print(f"Theme: {lyrics.theme}")
    print(f"\nLyrics ({len(lyrics.get_lines())} lines):")
    print(lyrics.text)
    
    # Example 2: Structured song
    print("\n\n2. Structured Song Generation")
    print("-" * 60)
    
    structure = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    
    song_lyrics = generator.generate(
        theme="finding strength in difficult times",
        genre="rock",
        structure=structure,
        max_lines=32
    )
    
    print(f"Structure: {' -> '.join(structure)}")
    print(f"\nSong:")
    print(song_lyrics.text)
    
    # Parse sections
    sections = song_lyrics.get_sections()
    print(f"\n\nParsed Sections ({len(sections)}):")
    for section_name in sections:
        print(f"  - {section_name}")
    
    # Example 3: Multiple genres
    print("\n\n3. Genre Comparison")
    print("-" * 60)
    
    test_genres = ["pop", "rock", "country", "hip-hop"]
    
    for genre in test_genres:
        test_lyrics = generator.generate(
            theme="freedom",
            genre=genre,
            max_lines=8
        )
        print(f"\n{genre.upper()}:")
        for line in test_lyrics.get_lines()[:4]:
            print(f"  {line}")
        print("  ...")
    
    # Example 4: Using custom configuration
    print("\n\n4. Custom Configuration")
    print("-" * 60)
    
    custom_config = LyricConfig(
        cache_dir="models/custom_lyrics",
        temperature=0.9,  # More creative
        max_length=1024,
        device="cpu"
    )
    
    custom_generator = LyricGenerator(config=custom_config)
    
    creative_lyrics = custom_generator.generate(
        theme="digital dreams in a neon city",
        genre="electronic",
        max_lines=12
    )
    
    print("Custom generation settings:")
    print(f"  Temperature: {custom_config.temperature}")
    print(f"  Max length: {custom_config.max_length}")
    print(f"\nCreative lyrics:")
    print(creative_lyrics.text)
    
    # Example 5: Save lyrics to file
    print("\n\n5. Saving Lyrics")
    print("-" * 60)
    
    output_dir = Path("output/lyrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "my_song_lyrics.txt"
    generator.save_lyrics(song_lyrics, str(output_path))
    
    print(f"✓ Lyrics saved to: {output_path}")
    
    # Example 6: Integration with MAGE Config
    print("\n\n6. Config Integration")
    print("-" * 60)
    
    config = Config.from_file("config/config.yaml")
    
    print(f"Lyrics configuration from config.yaml:")
    print(f"  Cache directory: {config.lyrics.cache_dir}")
    print(f"  Device: {config.lyrics.device}")
    print(f"  Resolved device: {config.lyrics.get_device()}")
    print(f"  Temperature: {config.lyrics.temperature}")
    print(f"  Top-k: {config.lyrics.top_k}")
    print(f"  Top-p: {config.lyrics.top_p}")
    
    # Create generator from config
    config_generator = LyricGenerator(config=config.lyrics)
    
    config_lyrics = config_generator.generate(
        theme="peaceful morning",
        genre="folk",
        max_lines=10
    )
    
    print(f"\nGenerated using config:")
    print(config_lyrics.text)
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
