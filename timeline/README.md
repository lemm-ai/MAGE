# Timeline & Arrangement System

The MAGE Timeline module provides a comprehensive system for multi-track audio composition and automatic song arrangement.

## Features

- **Multi-track Timeline**: Manage multiple audio tracks with precise timing control
- **Track Effects**: Volume, panning, fades (linear, exponential, logarithmic, S-curve)
- **Mute/Solo**: Individual track control for mixing
- **Arrangement Engine**: Automatic song structure generation
- **Section Management**: Define song sections (intro, verse, chorus, etc.)
- **Export**: Render mixed audio to WAV, FLAC, OGG formats
- **Markers**: Add timeline markers for navigation and organization

## Quick Start

### Basic Track Creation

```python
from mage.timeline import Track, TrackType
import numpy as np

# Create audio data (mono or stereo)
audio = np.random.randn(44100 * 5)  # 5 seconds of audio

# Create a track
track = Track(
    name="Lead Vocals",
    track_type=TrackType.VOCALS,
    audio_data=audio,
    sample_rate=44100,
    start_time=2.0,      # Start at 2 seconds
    volume=0.8,          # 80% volume
    fade_in=1.0,         # 1 second fade-in
    fade_out=1.5,        # 1.5 second fade-out
)
```

### Creating a Timeline

```python
from mage.timeline import Timeline

# Create timeline
timeline = Timeline(sample_rate=44100, name="My Song")

# Add tracks
timeline.add_track(track1)
timeline.add_track(track2)
timeline.add_track(track3)

# Render mixed audio
mixed_audio, sample_rate = timeline.render()

# Export to file
timeline.export("output/my_song.wav")
```

### Automatic Arrangement

```python
from mage.timeline import ArrangementEngine

# Create arrangement engine
engine = ArrangementEngine(tempo=120.0, time_signature=(4, 4))

# Create song structure
structure = engine.create_simple_structure(
    include_intro=True,
    num_verses=2,
    num_choruses=3,
    include_bridge=True,
    include_outro=True
)

# Arrange tracks automatically
timeline = engine.arrange_tracks(tracks, timeline_name="Auto-Arranged Song")
```

## Track Types

```python
class TrackType(Enum):
    VOCALS = "vocals"
    INSTRUMENTAL = "instrumental"
    BASS = "bass"
    DRUMS = "drums"
    OTHER = "other"
    MASTER = "master"
```

## Fade Types

```python
class FadeType(Enum):
    LINEAR = "linear"           # Linear fade
    EXPONENTIAL = "exponential" # Exponential curve (fast start, slow end)
    LOGARITHMIC = "logarithmic" # Logarithmic curve (slow start, fast end)
    S_CURVE = "s_curve"         # S-curve (smooth acceleration/deceleration)
```

## Track Class

### Attributes

- `name`: Human-readable track name
- `track_type`: Type of audio content (TrackType enum)
- `audio_data`: Audio samples as numpy array (mono or stereo)
- `sample_rate`: Sample rate in Hz
- `start_time`: Start position in timeline (seconds)
- `duration`: Duration override (None = use full audio)
- `volume`: Volume multiplier (0.0 to 1.0)
- `pan`: Stereo pan (-1.0 = left, 0.0 = center, 1.0 = right)
- `fade_in`: Fade-in duration (seconds)
- `fade_out`: Fade-out duration (seconds)
- `fade_type`: Type of fade curve
- `mute`: Mute track
- `solo`: Solo track (only solo tracks play)
- `metadata`: Additional metadata dictionary

### Properties

```python
track.end_time           # Calculate end time
track.is_stereo          # Check if stereo
track.is_mono            # Check if mono
track.get_audio_duration()  # Get audio duration in seconds
```

## Timeline Class

### Methods

#### Track Management

```python
# Add track
timeline.add_track(track, validate=True)

# Remove track
timeline.remove_track("track_name")

# Get track
track = timeline.get_track("track_name")

# Get tracks by type
vocal_tracks = timeline.get_tracks_by_type(TrackType.VOCALS)
```

#### Section Management

```python
from mage.timeline import ArrangementSection

# Create section
section = ArrangementSection(
    name="verse1",
    start_time=8.0,
    duration=16.0,
    track_config={
        TrackType.VOCALS: 0.9,
        TrackType.INSTRUMENTAL: 0.6
    }
)

# Add to timeline
timeline.add_section(section)
```

#### Markers

```python
from mage.timeline import TimelineMarker

# Add marker
marker = TimelineMarker("Chorus", 32.0, color="#FF0000")
timeline.add_marker(marker)
```

#### Rendering

```python
# Render entire timeline
audio, sample_rate = timeline.render()

# Render specific time range
audio, sr = timeline.render(start_time=10.0, end_time=30.0)

# Render without sections
audio, sr = timeline.render(apply_sections=False)
```

#### Export

```python
# Export to WAV
timeline.export("output/song.wav")

# Export to FLAC
timeline.export("output/song.flac", format="flac")

# Export specific range
timeline.export("output/intro.wav", start_time=0.0, end_time=8.0)
```

#### Information

```python
# Get timeline info
info = timeline.get_info()
# Returns:
# {
#     "name": "My Song",
#     "sample_rate": 44100,
#     "duration": 180.5,
#     "num_tracks": 4,
#     "num_sections": 7,
#     "num_markers": 3,
#     "tracks": [...]
# }

# Get duration
duration = timeline.get_duration()
```

## ArrangementEngine Class

### Initialization

```python
engine = ArrangementEngine(
    tempo=120.0,                  # BPM
    time_signature=(4, 4)         # (beats_per_bar, note_value)
)
```

### Methods

#### Time Conversion

```python
# Convert bars to seconds
seconds = engine.bars_to_seconds(8)  # 8 bars

# Convert seconds to bars
bars = engine.seconds_to_bars(16.0)  # 16 seconds

# Get bar duration
duration = engine.get_bar_duration()
```

#### Quantization

```python
# Quantize time to grid
quantized = engine.quantize_to_grid(
    time=3.14159,
    grid_division=4  # Quarter notes
)
# grid_division: 1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth, etc.
```

#### Song Structure

```python
# Create simple structure
structure = engine.create_simple_structure(
    include_intro=True,
    num_verses=2,
    num_choruses=3,
    include_bridge=True,
    include_outro=True
)
# Returns: [(section_name, duration), ...]

# Create sections from structure
sections = engine.create_sections_from_structure(structure)
```

#### Custom Structure

```python
# Define custom structure
custom_structure = [
    ("intro", 8.0),
    ("verse1", 16.0),
    ("prechorus", 8.0),
    ("chorus1", 16.0),
    ("verse2", 16.0),
    ("chorus2", 16.0),
    ("bridge", 12.0),
    ("chorus3", 16.0),
    ("outro", 8.0)
]

sections = engine.create_sections_from_structure(custom_structure)
```

#### Automatic Arrangement

```python
# Arrange tracks with default structure
timeline = engine.arrange_tracks(
    tracks=[track1, track2, track3],
    timeline_name="Arranged Song"
)

# Arrange with custom structure
timeline = engine.arrange_tracks(
    tracks=[track1, track2, track3],
    structure=custom_structure,
    timeline_name="Custom Arrangement"
)
```

#### Crossfades

```python
# Create crossfade transition
crossfaded = engine.create_crossfade_transition(
    track1=first_track,
    track2=second_track,
    crossfade_duration=2.0  # 2 second crossfade
)
# Returns list of modified tracks
```

## Default Section Volumes

The arrangement engine applies default volume configurations for each section type:

### Intro
- Vocals: 0% (silent)
- Instrumental: 70%
- Bass: 50%
- Drums: 60%

### Verse
- Vocals: 90%
- Instrumental: 60%
- Bass: 70%
- Drums: 70%

### Chorus
- Vocals: 100%
- Instrumental: 80%
- Bass: 90%
- Drums: 90%

### Bridge
- Vocals: 85%
- Instrumental: 75%
- Bass: 60%
- Drums: 70%

### Outro
- Vocals: 70%
- Instrumental: 60%
- Bass: 50%
- Drums: 40%

## Configuration

Timeline settings in `config/config.yaml`:

```yaml
timeline:
  default_tempo: 120.0             # Default tempo in BPM
  time_signature_numerator: 4      # Beats per bar
  time_signature_denominator: 4    # Note value
  default_crossfade: 2.0           # Default crossfade duration (seconds)
  quantize_grid: 4                 # Quantization grid (quarter notes)
  auto_normalize: true             # Auto-normalize to prevent clipping
  prevent_clipping: true           # Prevent clipping during mix
  
  # Section durations
  intro_duration: 8.0
  verse_duration: 16.0
  chorus_duration: 16.0
  bridge_duration: 12.0
  outro_duration: 8.0
```

### Loading Configuration

```python
from mage.config import Config

# Load config
config = Config.from_file("config/config.yaml")

# Create engine from config
engine = ArrangementEngine(
    tempo=config.timeline.default_tempo,
    time_signature=(
        config.timeline.time_signature_numerator,
        config.timeline.time_signature_denominator
    )
)
```

## Examples

See `examples/timeline_example.py` for comprehensive examples including:

1. Basic track creation
2. Stereo tracks with panning
3. Simple multi-track timeline
4. Rendering and exporting
5. Using the arrangement engine
6. Automatic track arrangement
7. Advanced custom arrangements
8. Configuration-based setup
9. Crossfade transitions

## Error Handling

All timeline operations include comprehensive error handling:

```python
from mage.exceptions import ValidationError, AudioProcessingError

try:
    timeline.render()
except ValidationError as e:
    print(f"Validation error: {e}")
except AudioProcessingError as e:
    print(f"Processing error: {e}")
```

## Logging

The module uses MAGELogger for detailed logging:

```python
import logging
from mage.utils import MAGELogger

# Set logging level
MAGELogger.configure(level=logging.DEBUG)

# All timeline operations are logged automatically
```

## Best Practices

### Track Management

1. **Use consistent sample rates**: All tracks should use the same sample rate for best results
2. **Normalize audio**: Keep audio levels below 1.0 to prevent clipping
3. **Use fades**: Apply fade-in/out to prevent clicks and pops
4. **Test solo/mute**: Use solo to isolate tracks during mixing

### Timeline Organization

1. **Use sections**: Define song sections for better organization
2. **Add markers**: Mark important positions for navigation
3. **Name tracks clearly**: Use descriptive names for easy identification
4. **Group by type**: Organize tracks by instrument type

### Arrangement

1. **Start simple**: Use `create_simple_structure()` as a starting point
2. **Customize gradually**: Modify default structure as needed
3. **Use quantization**: Align tracks to musical grid for tighter timing
4. **Test different tempos**: Experiment with tempo settings

### Performance

1. **Cache renders**: Store rendered audio if rendering multiple times
2. **Render ranges**: Only render needed time ranges
3. **Use appropriate sample rates**: Higher rates = more processing
4. **Monitor peak levels**: Check for clipping before export

## Integration

The timeline module integrates with other MAGE components:

- **Lyrics**: Generate vocals and arrange with instrumental tracks
- **Stems**: Separate stems and arrange individually
- **Audio Processing**: Apply effects before adding to timeline
- **Configuration**: Use centralized config for all settings

## Future Enhancements

Planned features:
- Real-time preview
- Time stretching and pitch shifting
- Automation curves for volume/pan
- Plugin effects integration
- MIDI support
- Multi-format export (MP3, AAC)
