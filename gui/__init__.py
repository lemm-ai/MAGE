"""MAGE GUI Module.

This module provides graphical user interfaces for MAGE including
Gradio web interface, Udio-style interface, and Tkinter enhancement popup.
"""

from mage.gui.gradio_interface import GradioInterface, is_gradio_available
from mage.gui.udio_interface import UdioInterface, is_udio_available
from mage.gui.enhancement_popup import EnhancementPopup, is_enhancement_popup_available

__all__ = [
    "GradioInterface", 
    "is_gradio_available",
    "UdioInterface",
    "is_udio_available",
    "EnhancementPopup",
    "is_enhancement_popup_available"
]
