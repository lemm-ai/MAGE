# MAGE Lyrics Module

## Overview

The lyrics module provides AI-powered lyrics generation using LyricMind-AI architecture. It supports multiple music genres, themes, and song structures with comprehensive error handling and logging throughout.

## Features

- **Multi-Genre Support**: Pop, rock, hip-hop, country, R&B, electronic, folk, jazz, metal
- **Themed Generation**: Create lyrics based on specific themes or topics
- **Structured Songs**: Define song structure (verse, chorus, bridge, etc.)
- **Device Support**: Automatic device detection (CUDA, ROCm, MPS, CPU)
- **Flexible Configuration**: Customizable temperature, top-k, top-p parameters
- **Model Caching**: Local model storage in `models/lyricmind/`

## Quick Start

```python
from mage.lyrics import LyricGenerator

# Create generator
generator = LyricGenerator()

# Generate simple lyrics
lyrics = generator.generate(
    theme="love and dreams",
    genre="pop",
    max_lines=16
)

print(lyrics.text)
```

## Structured Songs

```python
# Define song structure
structure = ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]

lyrics = generator.generate(
    theme="finding strength",
    genre="rock",
    structure=structure,
    max_lines=32
)

# Parse sections
sections = lyrics.get_sections()
for section_name, section_text in sections.items():
    print(f"[{section_name.upper()}]")
    print(section_text)
    print()
```

## Configuration

### Using Config File (config/config.yaml)

```yaml
lyrics:
  model_path: null
  cache_dir: "models/lyricmind"
  device: "auto"
  max_length: 512
  temperature: 0.8
  top_k: 50
  top_p: 0.9
  num_return_sequences: 1
```

### Programmatic Configuration

```python
from mage.lyrics import LyricGenerator, LyricConfig

config = LyricConfig(
    cache_dir="models/custom",
    temperature=0.9,  # More creative
    device="cuda",
    max_length=1024
)

generator = LyricGenerator(config=config)
```

## API Reference

### LyricGenerator

#### `__init__(config=None, device=None)`
Initialize the lyrics generator.

**Parameters:**
- `config` (LyricConfig, optional): Configuration object
- `device` (str, optional): Device override ("cuda", "cpu", "mps")

#### `generate(theme=None, genre="pop", structure=None, prompt=None, max_lines=16)`
Generate lyrics based on parameters.

**Parameters:**
- `theme` (str, optional): Theme or topic for lyrics
- `genre` (str): Music genre
- `structure` (list, optional): Song structure sections
- `prompt` (str, optional): Starting prompt
- `max_lines` (int): Maximum number of lines

**Returns:**
- `GeneratedLyrics`: Generated lyrics object

**Raises:**
- `AudioGenerationError`: If generation fails
- `InvalidParameterError`: If parameters are invalid

#### `save_lyrics(lyrics, output_path)`
Save lyrics to text file.

**Parameters:**
- `lyrics` (GeneratedLyrics): Lyrics to save
- `output_path` (str): Output file path

#### `clone_lyricmind_repo()`
Clone LyricMind-AI repository to local cache directory.

**Returns:**
- `bool`: True if successful

### GeneratedLyrics

Container for generated lyrics.

**Attributes:**
- `text` (str): Full lyrics text
- `genre` (str): Music genre
- `theme` (str, optional): Theme/topic
- `structure` (list, optional): Song structure
- `metadata` (dict, optional): Additional metadata

**Methods:**
- `get_lines()`: Get lyrics as list of lines
- `get_sections()`: Parse lyrics into sections dictionary

### LyricConfig

Configuration for lyrics generation.

**Attributes:**
- `model_path` (str, optional): Path to model file
- `cache_dir` (str): Model cache directory
- `device` (str): Device to use ("auto", "cuda", "cpu", "mps")
- `max_length` (int): Maximum sequence length (1-2048)
- `temperature` (float): Generation temperature (0.1-2.0)
- `top_k` (int): Top-k sampling (1-1000)
- `top_p` (float): Nucleus sampling (0.0-1.0)
- `num_return_sequences` (int): Number of sequences to generate

## Supported Genres

- **Pop**: Dance, shine, bright, party themes
- **Rock**: Thunder, fire, power, rebel themes
- **Country**: Road, home, fields, sunset themes
- **Hip-Hop**: Street, flow, rhythm, hustle themes
- **R&B**: Groove, smooth, vibe, passion themes
- **Electronic**: Digital, neon, synthetic themes
- **Folk**: Nature, tradition, story themes
- **Jazz**: Swing, smooth, improvisation themes
- **Metal**: Dark, intense, power themes

## Error Handling

All operations include comprehensive error handling with custom exceptions:

- `ModelLoadError`: Model fails to load
- `AudioGenerationError`: Generation fails
- `InvalidParameterError`: Invalid parameters provided
- `ConfigurationError`: Invalid configuration

Example:
```python
try:
    lyrics = generator.generate(genre="pop", max_lines=16)
except ModelLoadError as e:
    print(f"Model loading failed: {e.message}")
    print(f"Details: {e.details}")
except AudioGenerationError as e:
    print(f"Generation failed: {e.message}")
```

## Logging

All operations are logged with appropriate levels:

```python
2025-11-20 13:32:16 - mage.lyrics.generator - INFO - Initializing LyricGenerator
2025-11-20 13:32:16 - mage.lyrics.generator - INFO - Generating lyrics - Genre: pop, Theme: love
2025-11-20 13:32:16 - mage.lyrics.generator - INFO - Loading LyricMind-AI model...
2025-11-20 13:32:16 - mage.lyrics.generator - INFO - Using device: cpu
2025-11-20 13:32:17 - mage.lyrics.generator - INFO - Generated 16 lines of lyrics
```

## Examples

See `examples/lyrics_example.py` for comprehensive examples including:
1. Simple lyrics generation
2. Structured song generation
3. Genre comparison
4. Custom configuration
5. Saving lyrics to file
6. Config file integration

## Model Information

### Placeholder Model

Currently uses a placeholder LSTM model for testing. The architecture mimics LyricMind-AI:
- Embedding layer (vocab_size → 256)
- LSTM layers (256 → 512, 2 layers)
- Output layer (512 → vocab_size)

### LyricMind-AI Integration

To use the actual LyricMind-AI model:

1. Clone the repository:
```python
generator.clone_lyricmind_repo()
```

2. Place trained model in `models/lyricmind/lyricmind_model.pt`

3. Model file should contain:
- `model_state_dict`: Model weights
- `vocab`: Vocabulary dictionary
- `vocab_size`: Size of vocabulary

## Performance

- **Placeholder Model**: ~100ms per generation (CPU)
- **GPU Acceleration**: Automatic when CUDA/ROCm available
- **Memory Usage**: ~100MB (placeholder), ~500MB (full model)

## Future Enhancements

- Fine-tuned LyricMind-AI models
- Rhyme scheme enforcement
- Syllable counting for meter
- Multi-language support
- Style transfer between genres
- Collaborative filtering
