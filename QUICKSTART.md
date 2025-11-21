# MAGE Quick Start Guide

Welcome to MAGE (Mixed Audio Generation Engine)! This guide will help you get started quickly.

## Installation

1. **Clone or download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install MAGE:**
   ```bash
   pip install -e .
   ```

## Your First Generation

### Option 1: Python API

```python
from mage import MAGE

# Initialize the engine
engine = MAGE()

# Generate 10 seconds of ambient music
audio = engine.generate(
    duration=10.0,
    style="ambient",
    tempo=80
)

# Export the result
audio.export("my_first_track.wav")
```

### Option 2: Command Line

```bash
mage generate --duration 10 --style ambient --output my_first_track.wav
```

## Common Tasks

### List Available Styles

```bash
mage list-styles
```

Or in Python:
```python
engine = MAGE()
styles = engine.get_available_styles()
print(styles)
```

### Generate with Effects

```python
from mage import MAGE

engine = MAGE()

audio = engine.generate(
    duration=20.0,
    style="electronic",
    tempo=128
)

# Add reverb and compression
audio.apply_effects(reverb=0.3, compression=0.4)

# Normalize
audio.normalize(target_level=-3.0)

# Export
audio.export("electronic_track.wav")
```

### Use Custom Configuration

Create `my_config.yaml`:
```yaml
audio:
  sample_rate: 48000
  default_duration: 30.0

generation:
  default_style: "orchestral"
  default_tempo: 90
  complexity: 0.8
```

Use it:
```python
from mage import MAGE, Config

config = Config.from_file("my_config.yaml")
engine = MAGE(config=config)

audio = engine.generate()
audio.export("orchestral_track.wav")
```

## Running Examples

```bash
# Basic example
python examples/basic_example.py

# Advanced features
python examples/advanced_example.py

# Configuration example
python examples/config_example.py
```

## Running Tests

```bash
pytest tests/
```

## Next Steps

- Read the [API Documentation](docs/API.md)
- Check out the [examples](examples/)
- Experiment with different styles and parameters
- Customize the configuration to your needs

## Getting Help

- Check the README.md for detailed documentation
- Review the API.md for complete API reference
- Look at examples for usage patterns
- Open an issue on GitHub for bugs or questions

## Tips

- Start with low complexity (0.3-0.5) for simpler, cleaner sounds
- Higher complexity (0.7-0.9) creates more intricate compositions
- Use seeds for reproducible results
- Apply effects subtly for natural sound
- Normalize audio before export for consistent levels
