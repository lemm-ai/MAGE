"""Audio processing utilities."""

from mage.processors.audio_processor import AudioProcessor
from mage.processors.effects import EffectsProcessor, is_pedalboard_available

__all__ = [
    "AudioProcessor",
    "EffectsProcessor",
    "is_pedalboard_available"
]
