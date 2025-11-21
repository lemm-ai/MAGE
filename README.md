# MAGE - Mixed Audio Generation Engine

Professional AI-based music generation system with a DAW-style web interface for timeline-based composition.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

## ✨ Features

### 🎹 Timeline Studio Interface (Phase 7 ✅)
- **DAW-Style Workflow**: Professional timeline-based composition
- **Visual Timeline**: Color-coded clip blocks with waveform display
- **Clip Library**: Manage and organize all your generated clips
- **Playback Controls**: Play, pause, export your timeline
- **AI Generation**: Prompt-based music and lyrics generation

### 🎛️ Advanced Audio Processing
- **3-Band EQ**: Precise frequency control (Low/Mid/High)
- **Dynamic Compression**: Professional-grade compressor
- **Reverb**: Spatial enhancement with room size and damping
- **Limiter**: Peak control for mastering
- **Vocal Enhancement**: Denoise, brightness, warmth, clarity controls

### 🎵 AI-Powered Features
- **Music Generation**: MusicControlNet with context awareness
- **Lyrics Generation**: AI-powered lyric creation
- **Stem Separation**: Demucs-based stem extraction
- **Vocal Processing**: Advanced vocal enhancement

### 🖥️ Platform Support
- **GPU Acceleration**: CUDA, ROCm, Metal detection
- **CPU Fallback**: Automatic fallback for systems without GPU
- **Cross-Platform**: Windows, macOS, Linux support

### 🎛️ Enhancement Popup (Phase 8 ✅)
- **Desktop GUI**: Tkinter-based parameter adjustment popup
- **17 Parameters**: EQ, Compressor, Reverb, Limiter, Vocal Enhancement
- **Real-Time Preview**: Hear changes before applying
- **Integration**: Works alongside Timeline Studio interface

## 🏗️ Architecture

```
mage/
├── core/               # Core audio generation engine
├── models/             # AI model implementations
├── processors/         # Audio effects and processing
│   ├── effects.py      # EQ, compressor, reverb, limiter
│   └── audio_processor.py
├── stems/              # Demucs stem separation
├── vocals/             # Vocal enhancement
├── timeline/           # Timeline arrangement and merging
├── lyrics/             # AI lyrics generation
├── gui/                # Web interface
│   └── udio_interface.py  # Timeline Studio interface
├── platform/           # GPU/CPU detection
├── config/             # Configuration management
├── utils/              # Logging and utilities
└── exceptions/         # Custom exception classes
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
```bash
# Edit config/config.yaml with your settings (optional)
```

## 🚀 Quick Start

### Launch Timeline Studio Interface

```bash
python examples/timeline_studio_example.py
```

Then open your browser to `http://127.0.0.1:7861`

### Launch Enhancement Popup

```bash
# With audio file
python examples/enhancement_popup_example.py my_audio.wav

# Without audio (load later)
python examples/enhancement_popup_example.py
```

### Basic Python Usage

```python
from mage.gui.udio_interface import UdioInterface

# Initialize Timeline Studio
studio = UdioInterface()

# Launch web interface
studio.launch(
    server_name="127.0.0.1",
    server_port=7861,
    share=False
)
```

### Enhancement Popup Usage

```python
from mage.gui import EnhancementPopup

# Create popup with audio file
popup = EnhancementPopup(audio_path="song.wav")

# With callback for parameter changes
def on_parameter_change(param_name, value):
    print(f"{param_name} = {value}")

popup = EnhancementPopup(
    audio_path="song.wav",
    callback=on_parameter_change
)

# Show popup (blocks until closed)
popup.show()

# Get final parameters
params = popup.get_parameters()
```

### Generate Music with Timeline

```python
from mage.core.engine import MAGE
from mage.timeline import Timeline, TimelineClip

# Initialize engine
engine = MAGE()

# Create timeline
timeline = Timeline(bpm=120)

# Generate clips
clip1 = engine.generate(
    prompt="Energetic electronic intro with heavy bass",
    duration=8.0,
    bpm=120
)

clip2 = engine.generate(
    prompt="Melodic synth lead over driving beat",
    duration=16.0,
    bpm=120
)

# Add to timeline
timeline.add_clip(TimelineClip(audio=clip1, position=0.0))
timeline.add_clip(TimelineClip(audio=clip2, position=8.0))

# Export merged result
timeline.export("my_song.wav")
```

### Advanced Effects Processing

```python
from mage.processors.effects import EffectsProcessor
import soundfile as sf

# Load audio
audio, sr = sf.read("input.wav")

# Initialize effects processor
effects = EffectsProcessor()

# Apply EQ
audio = effects.apply_eq(
    audio,
    low_shelf_gain_db=3.0,
    mid_gain_db=0.0,
    high_shelf_gain_db=2.0
)

# Add compression
audio = effects.apply_compressor(
    audio,
    threshold_db=-20.0,
    ratio=4.0,
    attack_ms=5.0,
    release_ms=100.0
)

# Add reverb
audio = effects.apply_reverb(
    audio,
    room_size=0.7,
    damping=0.5,
    wet_level=0.3
)

# Save processed audio
sf.write("output.wav", audio, sr)
```

### Stem Separation

```python
from mage.stems import DemucsSeparator
import soundfile as sf

# Initialize separator
separator = DemucsSeparator()

# Separate stems
stems = separator.separate("song.wav")

# Access individual stems
drums = stems['drums']
bass = stems['bass']
vocals = stems['vocals']
other = stems['other']

# Save stems
sf.write("drums.wav", drums, separator.sample_rate)
sf.write("bass.wav", bass, separator.sample_rate)
```

### AI Lyrics Generation

```python
from mage.lyrics import LyricGenerator

# Initialize generator
generator = LyricGenerator()

# Generate lyrics
lyrics = generator.generate(
    prompt="Upbeat pop song about summer adventures",
    num_lines=8
)

print(lyrics)
```

## 📖 Examples

See the `examples/` directory for comprehensive examples:

- `timeline_studio_example.py` - Launch the web interface
- `enhancement_popup_example.py` - Launch desktop GUI popup
- `basic_example.py` - Simple music generation
- `effects_example.py` - Audio effects processing
- `stems_example.py` - Stem separation
- `lyrics_example.py` - Lyrics generation
- `vocals_example.py` - Vocal enhancement
- `timeline_example.py` - Timeline management

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
audio:
  sample_rate: 44100
  bit_depth: 16
  channels: 2

generation:
  default_duration: 10.0
  default_bpm: 120
  
effects:
  enable_gpu: true
  
models:
  cache_dir: "./models"
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest test_timeline_studio.py

# Run with coverage
pytest --cov=mage tests/
```

### Running with Debug Logging

```bash
python examples/timeline_studio_example.py --debug
```

## 📋 Requirements

### Core Dependencies
- Python 3.10+
- NumPy >= 1.21.0
- SoundFile >= 0.10.0
- librosa >= 0.9.0

### AI/ML Dependencies
- torch >= 2.0.0
- transformers >= 4.30.0
- demucs >= 4.0.0

### Audio Processing
- pedalboard >= 0.7.0
- matplotlib >= 3.5.0

### Web Interface
- gradio >= 4.0.0
- Pillow >= 9.0.0

### Desktop GUI
- tkinter (built-in, may need system package on Linux)

### Optional
- colorlog (enhanced logging)

See `requirements.txt` for complete list.

## 🎯 Roadmap

### Phase 7: Timeline Studio ✅ COMPLETE
- [x] DAW-style web interface
- [x] Timeline-based workflow
- [x] Clip library management
- [x] Advanced audio effects
- [x] Playback controls
- [x] Export functionality

### Phase 8: Enhancement Popup ✅ COMPLETE
- [x] Desktop Tkinter GUI
- [x] 17 parameter sliders (EQ, Compressor, Reverb, Limiter, Vocal)
- [x] Real-time preview functionality
- [x] Apply and save enhanced audio
- [x] Integration with Timeline Studio
- [x] Comprehensive error handling

### Phase 9: Advanced Features (Planned)
- [ ] Parameter preset system
- [ ] Real-time audio playback in popup
- [ ] Waveform visualizations
- [ ] Batch processing
- [ ] VST plugin support

### Future Phases
- Multi-track layering (drums, bass, melody, vocals)
- Collaborative features
- Cloud model hosting
- Mobile app support

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📚 Documentation

- `API.md` - Complete API reference
- `PHASE7_COMPLETE.md` - Timeline Studio implementation details
- `PHASE8_COMPLETE.md` - Enhancement Popup implementation details
- `GENERATION_PIPELINE_FIXES.md` - Generation pipeline fixes and improvements
- `DESIGN_ENHANCEMENTS.md` - Design decisions and enhancements
- `PROJECT_SUMMARY.md` - Project overview

## 🙏 Acknowledgments

- Demucs for stem separation
- Gradio for web interface framework
- Pedalboard for audio effects processing

---

**Made with ❤️ for music creators and AI enthusiasts**
