# Audio Effects Processing with Pedalboard

The MAGE Effects module provides professional-quality audio effects processing using Spotify's Pedalboard library.

## Features

- **Parametric EQ**: 3-band EQ with low shelf, parametric mid, and high shelf
- **Compression**: Dynamic range compression with configurable threshold, ratio, attack, and release
- **Reverb**: High-quality reverb with room size, damping, and wet/dry control
- **Chorus**: Modulation effect for thickening sounds
- **Delay**: Configurable delay with feedback
- **Limiter**: Prevent clipping and maximize loudness
- **Noise Gate**: Remove background noise
- **Gain**: Simple gain adjustment
- **Effects Chains**: Apply multiple effects in sequence

## Installation

```bash
pip install pedalboard>=0.7.0
```

## Quick Start

### Basic Usage

```python
from mage.processors import EffectsProcessor
import numpy as np

# Create processor
processor = EffectsProcessor(sample_rate=44100)

# Apply EQ
processed = processor.apply_eq(
    audio,
    low_shelf_gain_db=3.0,      # Boost bass
    mid_gain_db=-2.0,           # Cut mids
    high_shelf_gain_db=1.0      # Boost highs
)
```

### Check Availability

```python
from mage.processors import is_pedalboard_available

if is_pedalboard_available():
    # Use effects
    processor = EffectsProcessor(sample_rate=44100)
else:
    print("Install pedalboard: pip install pedalboard")
```

## Effects Reference

### Parametric EQ

3-band equalizer with shelving filters and parametric mid.

```python
processed = processor.apply_eq(
    audio,
    low_shelf_gain_db=3.0,      # Low shelf gain (-24 to +24 dB)
    low_shelf_freq=100.0,       # Low shelf frequency (Hz)
    mid_gain_db=-2.0,           # Mid peak gain (-24 to +24 dB)
    mid_freq=1000.0,            # Mid frequency (Hz)
    mid_q=1.0,                  # Mid Q factor (0.1 to 10.0)
    high_shelf_gain_db=1.5,     # High shelf gain (-24 to +24 dB)
    high_shelf_freq=8000.0      # High shelf frequency (Hz)
)
```

**Parameters:**
- `low_shelf_gain_db`: Boost/cut low frequencies (-24 to +24 dB)
- `low_shelf_freq`: Crossover frequency for low shelf (Hz)
- `mid_gain_db`: Boost/cut mid frequencies (-24 to +24 dB)
- `mid_freq`: Center frequency for mid band (Hz)
- `mid_q`: Q factor (bandwidth) for mid band (0.1 to 10.0)
- `high_shelf_gain_db`: Boost/cut high frequencies (-24 to +24 dB)
- `high_shelf_freq`: Crossover frequency for high shelf (Hz)

### Compressor

Dynamic range compression for evening out volume levels.

```python
processed = processor.apply_compressor(
    audio,
    threshold_db=-20.0,  # Compression threshold (-60 to 0 dB)
    ratio=4.0,           # Compression ratio (1.0 to 20.0)
    attack_ms=5.0,       # Attack time (0.1 to 100 ms)
    release_ms=100.0     # Release time (10 to 1000 ms)
)
```

**Parameters:**
- `threshold_db`: Level above which compression occurs (-60 to 0 dB)
- `ratio`: Compression ratio (1.0 = no compression, 20.0 = limiting)
- `attack_ms`: How quickly compression engages (0.1 to 100 ms)
- `release_ms`: How quickly compression releases (10 to 1000 ms)

**Common Settings:**
- **Vocals**: threshold=-18dB, ratio=3:1, attack=10ms, release=200ms
- **Drums**: threshold=-15dB, ratio=4:1, attack=1ms, release=50ms
- **Master**: threshold=-10dB, ratio=2:1, attack=5ms, release=150ms

### Reverb

High-quality reverb effect for adding space and depth.

```python
processed = processor.apply_reverb(
    audio,
    room_size=0.5,       # Room size (0.0 to 1.0)
    damping=0.5,         # High frequency damping (0.0 to 1.0)
    wet_level=0.33,      # Wet signal level (0.0 to 1.0)
    dry_level=0.4,       # Dry signal level (0.0 to 1.0)
    width=1.0            # Stereo width (0.0 to 1.0)
)
```

**Parameters:**
- `room_size`: Size of the reverberant space (0.0 = small, 1.0 = large)
- `damping`: High frequency damping (0.0 = bright, 1.0 = dark)
- `wet_level`: Reverb signal level (0.0 to 1.0)
- `dry_level`: Original signal level (0.0 to 1.0)
- `width`: Stereo width of reverb (0.0 = mono, 1.0 = full stereo)

**Common Settings:**
- **Small Room**: room_size=0.3, damping=0.6, wet=0.2, dry=0.8
- **Hall**: room_size=0.8, damping=0.3, wet=0.4, dry=0.6
- **Plate**: room_size=0.5, damping=0.5, wet=0.35, dry=0.65

### Chorus

Modulation effect that thickens and widens sound.

```python
processed = processor.apply_chorus(
    audio,
    rate_hz=1.0,            # LFO rate (0.1 to 10 Hz)
    depth=0.25,             # Modulation depth (0.0 to 1.0)
    centre_delay_ms=7.0,    # Center delay (1 to 20 ms)
    feedback=0.0,           # Feedback amount (0.0 to 1.0)
    mix=0.5                 # Dry/wet mix (0.0 to 1.0)
)
```

**Parameters:**
- `rate_hz`: LFO modulation rate (0.1 to 10 Hz)
- `depth`: Modulation depth (0.0 to 1.0)
- `centre_delay_ms`: Base delay time (1 to 20 ms)
- `feedback`: Feedback amount (0.0 to 1.0)
- `mix`: Dry/wet mix (0.0 = dry only, 1.0 = wet only)

### Delay

Echo/delay effect with feedback.

```python
processed = processor.apply_delay(
    audio,
    delay_seconds=0.5,   # Delay time (0.001 to 2.0 s)
    feedback=0.3,        # Feedback amount (0.0 to 0.95)
    mix=0.5              # Dry/wet mix (0.0 to 1.0)
)
```

**Parameters:**
- `delay_seconds`: Delay time in seconds (0.001 to 2.0)
- `feedback`: Feedback amount for repeats (0.0 to 0.95)
- `mix`: Dry/wet mix (0.0 to 1.0)

**Tip:** Calculate delay time from tempo:
```python
bpm = 120
quarter_note = 60.0 / bpm        # 0.5s
eighth_note = quarter_note / 2   # 0.25s
```

### Limiter

Prevent clipping and maximize loudness.

```python
processed = processor.apply_limiter(
    audio,
    threshold_db=-1.0,   # Limiting threshold (-20 to 0 dB)
    release_ms=100.0     # Release time (10 to 1000 ms)
)
```

**Parameters:**
- `threshold_db`: Level at which limiting occurs (-20 to 0 dB)
- `release_ms`: How quickly limiter releases (10 to 1000 ms)

**Common Settings:**
- **Mastering**: threshold=-0.3dB, release=100ms
- **Broadcast**: threshold=-1.0dB, release=50ms
- **Safety**: threshold=-3.0dB, release=150ms

### Noise Gate

Remove background noise during quiet passages.

```python
processed = processor.apply_noise_gate(
    audio,
    threshold_db=-40.0,  # Gate threshold (-100 to 0 dB)
    ratio=10.0,          # Gate ratio (1.0 to 20.0)
    attack_ms=1.0,       # Attack time (0.1 to 100 ms)
    release_ms=100.0     # Release time (10 to 1000 ms)
)
```

**Parameters:**
- `threshold_db`: Level below which gating occurs (-100 to 0 dB)
- `ratio`: Gate ratio (higher = more aggressive)
- `attack_ms`: How quickly gate opens (0.1 to 100 ms)
- `release_ms`: How quickly gate closes (10 to 1000 ms)

### Gain

Simple gain adjustment.

```python
processed = processor.apply_gain(
    audio,
    gain_db=6.0  # Gain in dB (-60 to +60)
)
```

**Parameters:**
- `gain_db`: Gain adjustment in decibels (-60 to +60 dB)

## Effects Chains

Apply multiple effects in sequence.

### Basic Chain

```python
chain = [
    {"type": "eq", "low_shelf_gain_db": 3.0},
    {"type": "compressor", "threshold_db": -20.0, "ratio": 4.0},
    {"type": "reverb", "room_size": 0.7, "wet_level": 0.3}
]

processed = processor.apply_chain(audio, chain)
```

### Vocal Processing Chain

```python
vocal_chain = [
    # 1. High-pass filter (remove rumble)
    {"type": "eq", "low_shelf_gain_db": -12.0, "low_shelf_freq": 80.0},
    
    # 2. De-esser (reduce harsh highs)
    {"type": "eq", "high_shelf_gain_db": -3.0, "high_shelf_freq": 7000.0},
    
    # 3. Compression
    {"type": "compressor", "threshold_db": -18.0, "ratio": 3.0, 
     "attack_ms": 10.0, "release_ms": 200.0},
    
    # 4. Presence boost
    {"type": "eq", "mid_gain_db": 2.0, "mid_freq": 3000.0, "mid_q": 1.5},
    
    # 5. Reverb
    {"type": "reverb", "room_size": 0.6, "wet_level": 0.25, "dry_level": 0.75},
    
    # 6. Safety limiter
    {"type": "limiter", "threshold_db": -1.0}
]

processed = processor.apply_chain(audio, vocal_chain)
```

### Mastering Chain

```python
mastering_chain = [
    # 1. Gentle EQ
    {"type": "eq", "low_shelf_gain_db": 1.0, "low_shelf_freq": 100.0,
     "high_shelf_gain_db": 0.5, "high_shelf_freq": 10000.0},
    
    # 2. Multiband compression (simplified)
    {"type": "compressor", "threshold_db": -15.0, "ratio": 2.0, 
     "attack_ms": 5.0, "release_ms": 150.0},
    
    # 3. Final limiter
    {"type": "limiter", "threshold_db": -0.3, "release_ms": 100.0}
]

processed = processor.apply_chain(audio, mastering_chain)
```

## Configuration

Effects settings in `config/config.yaml`:

```yaml
effects:
  # EQ settings
  eq_low_shelf_gain_db: 0.0
  eq_low_shelf_freq: 100.0
  eq_mid_gain_db: 0.0
  eq_mid_freq: 1000.0
  eq_mid_q: 1.0
  eq_high_shelf_gain_db: 0.0
  eq_high_shelf_freq: 8000.0
  
  # Compressor settings
  comp_threshold_db: -20.0
  comp_ratio: 4.0
  comp_attack_ms: 5.0
  comp_release_ms: 100.0
  
  # Reverb settings
  reverb_room_size: 0.5
  reverb_damping: 0.5
  reverb_wet_level: 0.33
  reverb_dry_level: 0.4
  reverb_width: 1.0
  
  # Limiter settings
  limiter_threshold_db: -1.0
  limiter_release_ms: 100.0
  
  # Enable/disable
  enable_eq: false
  enable_compressor: false
  enable_reverb: false
  enable_limiter: true
```

### Using Configuration

```python
from mage.config import Config
from mage.processors import EffectsProcessor

# Load config
config = Config.from_file("config/config.yaml")

# Create processor
processor = EffectsProcessor(sample_rate=config.audio.sample_rate)

# Apply enabled effects
if config.effects.enable_compressor:
    audio = processor.apply_compressor(
        audio,
        threshold_db=config.effects.comp_threshold_db,
        ratio=config.effects.comp_ratio,
        attack_ms=config.effects.comp_attack_ms,
        release_ms=config.effects.comp_release_ms
    )
```

## Error Handling

All effects include comprehensive error handling:

```python
from mage.exceptions import AudioProcessingError, InvalidParameterError

try:
    processed = processor.apply_compressor(
        audio,
        threshold_db=-20.0,
        ratio=4.0
    )
except InvalidParameterError as e:
    print(f"Invalid parameter: {e}")
except AudioProcessingError as e:
    print(f"Processing failed: {e}")
```

## Examples

See `examples/effects_example.py` for comprehensive examples including:

1. Basic EQ
2. Vocal compression
3. Hall reverb
4. Chorus effect
5. Delay effect
6. Mastering limiter
7. Noise gate
8. Vocal processing chain
9. Mastering chain
10. Configuration-based processing

## Best Practices

### EQ
- **Cut before boost**: Removing unwanted frequencies is often better than boosting
- **Use high-pass filters**: Remove rumble below 80Hz on most sources
- **Be gentle**: Small adjustments (±3dB) are usually sufficient
- **Use narrow Q for cuts, wide Q for boosts**

### Compression
- **Slower attack preserves transients**: Use 10-30ms for vocals, 1-5ms for drums
- **Longer release sounds natural**: 100-300ms for most sources
- **Don't over-compress**: 3-6dB of gain reduction is typical
- **Use makeup gain** to compensate for level loss

### Reverb
- **Less is more**: Start with 20-30% wet and adjust
- **Match room size to source**: Small for intimate, large for epic
- **Pre-delay**: Add 10-30ms pre-delay to separate dry and wet signals
- **EQ the reverb**: Roll off lows to prevent muddiness

### Effects Chains
- **Order matters**: Typical order is EQ → Compression → Modulation → Reverb → Limiter
- **Use limiters last**: Always place limiters at the end of the chain
- **A/B compare**: Toggle effects on/off to ensure improvements
- **Save presets**: Document successful chains in config files

### Performance
- **Process in batches**: Apply chains rather than individual effects when possible
- **Use appropriate sample rates**: 44.1kHz for most applications
- **Monitor CPU usage**: Complex chains can be CPU-intensive
- **Cache processed audio**: Store results if applying same settings multiple times

## Integration

The effects module integrates with other MAGE components:

- **Timeline**: Apply effects to individual tracks or mixed output
- **Stems**: Process separated stems independently
- **Audio Processor**: Enhanced version of basic processing
- **Configuration**: Centralized settings management

## Troubleshooting

### Pedalboard Not Available

```python
from mage.processors import is_pedalboard_available

if not is_pedalboard_available():
    print("Install pedalboard:")
    print("  pip install pedalboard>=0.7.0")
```

### Audio Clipping

```python
# Always use a limiter at the end of your chain
chain.append({"type": "limiter", "threshold_db": -0.5})
```

### Distortion

- Reduce input levels before heavy processing
- Lower compression ratio or raise threshold
- Check for cascading gain from multiple effects

### Noise Issues

- Use noise gate before other effects
- Apply EQ to remove problem frequencies
- Ensure proper threshold settings

## Future Enhancements

Planned features:
- Parametric EQ with more bands
- Multiband compression
- De-esser
- Transient shaper
- Saturation/distortion
- Stereo imaging
- Mid/side processing
- Plugin support (VST/AU)
