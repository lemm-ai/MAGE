# API Reference

## Core Classes

### MAGE

Main engine class for audio generation.

```python
from mage import MAGE, Config

engine = MAGE(config=None)
```

**Parameters:**
- `config` (Config, optional): Configuration object. Uses defaults if not provided.

**Methods:**

#### `generate()`

Generate audio with specified parameters.

```python
audio = engine.generate(
    duration=30.0,
    style="ambient",
    tempo=120,
    key="C_major",
    complexity=0.5,
    seed=None
)
```

**Parameters:**
- `duration` (float): Duration in seconds
- `style` (str): Music style (ambient, electronic, orchestral, jazz, rock, classical)
- `tempo` (int): Tempo in BPM (20-300)
- `key` (str): Musical key
- `complexity` (float): Complexity level (0.0-1.0)
- `seed` (int, optional): Random seed for reproducibility

**Returns:** `GeneratedAudio` object

#### `get_available_styles()`

Get list of available music styles.

**Returns:** `list[str]`

#### `get_available_keys()`

Get list of available musical keys.

**Returns:** `list[str]`

---

### GeneratedAudio

Container for generated audio with processing capabilities.

**Attributes:**
- `data` (np.ndarray): Audio data array
- `sample_rate` (int): Sample rate in Hz
- `duration` (float): Duration in seconds
- `metadata` (dict): Generation metadata

**Methods:**

#### `apply_effects()`

Apply audio effects.

```python
audio.apply_effects(
    reverb=0.3,
    compression=0.4,
    eq={"bass": 0.5, "treble": -0.2}
)
```

**Parameters:**
- `reverb` (float): Reverb amount (0.0-1.0)
- `compression` (float): Compression amount (0.0-1.0)
- `eq` (dict, optional): EQ parameters

**Returns:** Self for method chaining

#### `normalize()`

Normalize audio to target level.

```python
audio.normalize(target_level=-3.0)
```

**Parameters:**
- `target_level` (float): Target level in dB

**Returns:** Self for method chaining

#### `export()`

Export audio to file.

```python
audio.export("output.wav", format=None, bitrate=None)
```

**Parameters:**
- `output_path` (str | Path): Output file path
- `format` (str, optional): Audio format (inferred from extension)
- `bitrate` (str, optional): Bitrate for compressed formats

---

### Config

Configuration management class.

```python
from mage import Config

config = Config()
config = Config.from_file("config.yaml")
```

**Attributes:**
- `audio` (AudioConfig): Audio settings
- `model` (ModelConfig): Model settings
- `generation` (GenerationConfig): Generation settings
- `logging` (LoggingConfig): Logging settings

**Methods:**

#### `from_file()`

Load configuration from YAML file.

```python
config = Config.from_file("custom_config.yaml")
```

#### `to_file()`

Save configuration to YAML file.

```python
config.to_file("my_config.yaml")
```

#### `update()`

Update configuration values.

```python
config.update({
    "audio": {"sample_rate": 48000},
    "generation": {"default_tempo": 140}
})
```

---

## Configuration Options

### AudioConfig

- `sample_rate` (int): Sample rate in Hz (22050, 44100, 48000, 96000)
- `bit_depth` (int): Bit depth (16, 24, 32)
- `channels` (int): Number of channels (1, 2)
- `max_duration` (float): Maximum duration in seconds
- `default_duration` (float): Default generation duration

### ModelConfig

- `model_type` (str): Model type to use
- `model_path` (str): Path to custom model
- `cache_dir` (str): Model cache directory
- `device` (str): Device (auto, cpu, cuda, mps)
- `precision` (str): Model precision (float16, float32)

### GenerationConfig

- `default_style` (str): Default music style
- `default_tempo` (int): Default tempo in BPM
- `default_key` (str): Default musical key
- `complexity` (float): Generation complexity (0.0-1.0)
- `seed` (int): Random seed

### LoggingConfig

- `level` (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_dir` (str): Log file directory
- `console_output` (bool): Enable console logging
- `file_output` (bool): Enable file logging

---

## Exceptions

All exceptions inherit from `MAGEException`.

- `AudioGenerationError`: Audio generation failed
- `ModelLoadError`: Model loading failed
- `ConfigurationError`: Invalid configuration
- `AudioProcessingError`: Audio processing failed
- `InvalidParameterError`: Invalid parameter value
- `ResourceNotFoundError`: Resource not found
- `ExportError`: Audio export failed

---

## CLI Usage

```bash
# Generate audio
mage generate --duration 30 --style ambient --output ambient.wav

# List available styles
mage list-styles

# List available keys
mage list-keys

# Use custom configuration
mage generate --config custom_config.yaml --output track.wav
```
