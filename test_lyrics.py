"""Test script for lyrics generation functionality."""

from mage.lyrics import LyricGenerator
from mage.config import Config

print("=" * 70)
print("Testing MAGE Lyrics Generation (Phase 2)")
print("=" * 70)

# Test 1: Initialize LyricGenerator
print("\n1. Initializing LyricGenerator...")
try:
    generator = LyricGenerator()
    print("   ✓ LyricGenerator initialized successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Generate simple lyrics
print("\n2. Generating simple lyrics...")
try:
    lyrics = generator.generate(
        theme="love and dreams",
        genre="pop",
        max_lines=12
    )
    print("   ✓ Lyrics generated successfully")
    print(f"   Generated {len(lyrics.get_lines())} lines")
    print("\n   Preview:")
    for i, line in enumerate(lyrics.get_lines()[:6], 1):
        print(f"   {i}. {line}")
    if len(lyrics.get_lines()) > 6:
        print(f"   ... ({len(lyrics.get_lines()) - 6} more lines)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Generate structured lyrics
print("\n3. Generating structured lyrics...")
try:
    structured_lyrics = generator.generate(
        theme="journey and hope",
        genre="rock",
        structure=["verse", "chorus", "verse", "chorus", "bridge", "chorus"],
        max_lines=24
    )
    print("   ✓ Structured lyrics generated")
    sections = structured_lyrics.get_sections()
    print(f"   Sections: {', '.join(sections.keys())}")
    
    # Show first section
    first_section = list(sections.items())[0]
    print(f"\n   {first_section[0].upper()}:")
    for line in first_section[1].split('\n')[:4]:
        print(f"   {line}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test different genres
print("\n4. Testing different genres...")
genres = ["pop", "rock", "hip-hop", "country"]
for genre in genres:
    try:
        test_lyrics = generator.generate(genre=genre, max_lines=4)
        print(f"   ✓ {genre.upper()}: {len(test_lyrics.get_lines())} lines generated")
    except Exception as e:
        print(f"   ✗ {genre.upper()}: {e}")

# Test 5: Save lyrics
print("\n5. Saving lyrics to file...")
try:
    output_path = "output/test_lyrics.txt"
    generator.save_lyrics(lyrics, output_path)
    print(f"   ✓ Lyrics saved to {output_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Config integration
print("\n6. Testing config integration...")
try:
    config = Config()
    print(f"   Lyrics cache dir: {config.lyrics.cache_dir}")
    print(f"   Lyrics device: {config.lyrics.device}")
    print(f"   Resolved device: {config.lyrics.get_device()}")
    print(f"   Temperature: {config.lyrics.temperature}")
    print(f"   Max length: {config.lyrics.max_length}")
    print("   ✓ Config integration working")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("Lyrics Generation Test Complete")
print("=" * 70)
print("\nFull lyrics output:")
print("-" * 70)
print(f"Genre: {lyrics.genre}")
print(f"Theme: {lyrics.theme}")
print("-" * 70)
print(lyrics.text)
print("-" * 70)
