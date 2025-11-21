"""Example launcher for MAGE Enhancement Popup.

This script demonstrates how to use the Tkinter enhancement popup
for real-time audio parameter adjustment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mage.gui.enhancement_popup import EnhancementPopup, is_enhancement_popup_available
from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


def parameter_changed_callback(param_name: str, value: float):
    """Callback function for when parameters change.
    
    Args:
        param_name: Name of the parameter that changed
        value: New value
    """
    logger.info(f"Parameter changed: {param_name} = {value}")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("MAGE Enhancement Popup")
    print("=" * 60)
    
    # Check availability
    if not is_enhancement_popup_available():
        print("\nError: Enhancement popup not available")
        print("Please install required dependencies:")
        print("  pip install numpy soundfile")
        return 1
    
    print("\nInitializing enhancement popup...")
    
    try:
        # Example: Load an audio file for enhancement
        # You can specify an audio path or leave it None to load later
        audio_path = None
        
        # Check if user provided an audio file
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
            print(f"Loading audio: {audio_path}")
        else:
            print("No audio file specified. Launch with: python enhancement_popup_example.py <audio_file>")
            print("You can still adjust parameters and preview later.")
        
        # Create popup with callback
        popup = EnhancementPopup(
            audio_path=audio_path,
            callback=parameter_changed_callback
        )
        
        print("\n" + "=" * 60)
        print("Enhancement Popup Controls")
        print("=" * 60)
        print("  🎛️  Adjust sliders to change parameters")
        print("  🎧  Click 'Preview' to hear changes")
        print("  ✅  Click 'Apply' to save enhanced audio")
        print("  🔄  Click 'Reset' to restore defaults")
        print("  ❌  Click 'Close' to exit")
        print("=" * 60 + "\n")
        
        # Show popup (blocks until window is closed)
        popup.show()
        
        print("\nEnhancement popup closed")
        
        # Get final parameters
        final_params = popup.get_parameters()
        print("\nFinal Parameters:")
        for param, value in final_params.items():
            print(f"  {param}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        logger.error(f"Enhancement popup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
