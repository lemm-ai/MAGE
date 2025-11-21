# MAGE Gradio GUI Module

The MAGE Gradio GUI provides a comprehensive web-based interface for music generation and audio processing.

## Features

### 📝 Lyrics Generation
- AI-powered lyrics generation using LyricMind-AI
- Customizable genre, theme, length, and creativity
- Supports multiple music genres

### 🎼 Stem Separation
- High-quality stem separation using Demucs
- Extracts vocals, drums, bass, and other stems
- Supports various audio formats
- Built-in caching for faster processing

### 🎙️ Vocal Enhancement
- AI-based vocal quality improvement
- Denoising with adjustable strength
- Spectral enhancement (brightness, warmth, clarity)
- Dynamic range optimization
- Real-time preview

### 🎚️ Audio Effects
- Professional audio effects using Pedalboard
- EQ, Compressor, Reverb, Limiter
- Adjustable parameters for each effect
- High-quality processing

## Installation

```bash
pip install gradio>=4.0.0
```

## Quick Start

### Launch Web Interface

```python
from mage.gui import GradioInterface

# Create interface
interface = GradioInterface(config_path="config/config.yaml")

# Launch web server
interface.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False  # Set to True for public sharing
)
```

### Command Line

```bash
python examples/gradio_example.py
```

Then open your browser to: http://127.0.0.1:7860

## Usage Examples

### Generate Lyrics

1. Navigate to "📝 Lyrics Generation" tab
2. Select genre (Pop, Rock, Hip-Hop, etc.)
3. Enter theme/topic (love, adventure, freedom, etc.)
4. Adjust number of lines (4-32)
5. Set temperature for creativity (0.1-2.0)
6. Click "🎤 Generate Lyrics"

### Separate Stems

1. Navigate to "🎼 Stem Separation" tab
2. Upload audio file (WAV, MP3, etc.)
3. Click "🔀 Separate Stems"
4. Download individual stems:
   - Vocals
   - Drums
   - Bass
   - Other

### Enhance Vocals

1. Navigate to "🎙️ Vocal Enhancement" tab
2. Upload vocal audio file
3. Adjust enhancement settings:
   - **Denoise Strength** (0-1): Amount of noise reduction
   - **Brightness** (-1 to 1): High frequency boost/cut
   - **Warmth** (-1 to 1): Low-mid frequency boost/cut
   - **Clarity** (-1 to 1): Mid-high frequency boost/cut
   - **Target Level** (-40 to 0 dB): Final RMS level
4. Click "✨ Enhance Vocals"
5. Download enhanced audio

### Apply Effects

1. Navigate to "🎚️ Audio Effects" tab
2. Upload audio file
3. Select effect type:
   - **EQ**: Param1=Low Shelf, Param2=High Shelf
   - **Compressor**: Param1=Threshold, Param2=Ratio
   - **Reverb**: Param1=Room Size, Param2=Wet Level
   - **Limiter**: Param1=Threshold
4. Adjust parameters
5. Click "🎛️ Apply Effect"
6. Download processed audio

## Configuration

The interface uses settings from `config/config.yaml`:

```yaml
audio:
  sample_rate: 44100
  
lyrics:
  device: "auto"
  temperature: 0.8
  
stems:
  model_name: "htdemucs"
  device: "auto"
  
vocals:
  denoise_strength: 0.7
  brightness: 0.2
  warmth: 0.1
  clarity: 0.3
  target_level_db: -18.0
  
effects:
  enable_limiter: true
```

## Advanced Usage

### Custom Configuration

```python
from mage.gui import GradioInterface
from mage.config import Config

# Load and modify config
config = Config.from_file("config/config.yaml")
config.audio.sample_rate = 48000
config.vocals.denoise_strength = 0.9

# Save modified config
config.to_file("config/custom_config.yaml")

# Create interface with custom config
interface = GradioInterface(config_path="config/custom_config.yaml")
```

### Programmatic Access

```python
from mage.gui import GradioInterface

interface = GradioInterface()

# Generate lyrics programmatically
lyrics = interface.generate_lyrics(
    genre="Rock",
    theme="rebellion",
    num_lines=16,
    temperature=0.9
)

# Enhance vocals
enhanced_path, status = interface.enhance_vocals(
    audio_file="input.wav",
    denoise_strength=0.8,
    brightness=0.3,
    warmth=0.1,
    clarity=0.4,
    target_level=-16.0
)
```

### Public Sharing

To create a public share link:

```python
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Creates public Gradio share link
)
```

## Interface Components

### Lyrics Tab
- Genre dropdown (Pop, Rock, Hip-Hop, Country, Jazz, Classical, Electronic)
- Theme text input
- Number of lines slider (4-32)
- Temperature slider (0.1-2.0)
- Generate button
- Output text area

### Stems Tab
- Audio file upload
- Separate stems button
- Individual stem players (vocals, drums, bass, other)
- Status display

### Vocals Tab
- Audio file upload
- Enhancement sliders:
  - Denoise strength (0-1)
  - Brightness (-1 to 1)
  - Warmth (-1 to 1)
  - Clarity (-1 to 1)
  - Target level (-40 to 0 dB)
- Enhance button
- Output audio player
- Status display

### Effects Tab
- Audio file upload
- Effect type dropdown (EQ, Compressor, Reverb, Limiter)
- Parameter sliders (2 per effect)
- Apply effect button
- Output audio player
- Parameter guide
- Status display

### About Tab
- Feature overview
- System information
- Version info
- Credits

## Error Handling

The interface includes comprehensive error handling:

```python
try:
    enhanced_path, status = interface.enhance_vocals(...)
    if enhanced_path:
        print(f"Success: {status}")
    else:
        print(f"Error: {status}")
except Exception as e:
    print(f"Failed: {e}")
```

All errors are logged and displayed in the UI status field.

## Performance Tips

1. **Model Loading**: Models are lazy-loaded on first use
2. **Caching**: Enable stem caching in config for faster repeat processing
3. **Sample Rate**: Use 44.1kHz for most applications, 48kHz for professional
4. **Device**: Set `device: "cuda"` in config for GPU acceleration

## Troubleshooting

### Interface Won't Launch

```bash
# Check Gradio installation
pip install --upgrade gradio

# Verify dependencies
pip install -r requirements.txt
```

### Models Not Loading

```bash
# Check PyTorch installation
pip install torch torchaudio

# Check other dependencies
pip install demucs pedalboard librosa
```

### Port Already in Use

```python
# Use different port
interface.launch(server_port=7861)
```

### Audio Processing Fails

- Ensure audio files are valid (WAV, MP3, FLAC, etc.)
- Check file size (large files may take longer)
- Verify sufficient RAM/VRAM for model loading
- Check logs in `logs/` directory

## API Reference

### GradioInterface

```python
class GradioInterface:
    def __init__(self, config_path: str | Path = "config/config.yaml")
    def generate_lyrics(self, genre, theme, num_lines, temperature) -> str
    def separate_stems(self, audio_file, progress) -> Tuple[...]
    def enhance_vocals(self, audio_file, ..., progress) -> Tuple[str, str]
    def apply_effects(self, audio_file, effect_type, ..., progress) -> Tuple[str, str]
    def create_interface(self) -> gr.Blocks
    def launch(self, server_name, server_port, share, debug) -> None
```

### is_gradio_available

```python
def is_gradio_available() -> bool
```

Check if Gradio is installed and available.

## Integration

The Gradio interface integrates with all MAGE modules:

- **mage.lyrics**: LyricGenerator for AI lyrics
- **mage.stems**: DemucsSeparator for stem separation
- **mage.vocals**: VocalEnhancer for vocal improvement
- **mage.processors**: EffectsProcessor for audio effects
- **mage.config**: Config for settings management

## Future Enhancements

Planned features:
- Real-time audio preview
- Batch processing
- Custom effect chains
- Waveform visualization
- Audio comparison (before/after)
- Export presets
- User accounts and saved settings
- Timeline arrangement interface
- Multi-file projects
