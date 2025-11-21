"""Example: Launch MAGE Gradio web interface.

This example demonstrates how to launch the full MAGE web GUI.

Usage:
    Run from the MAGE root directory:
    python examples/gradio_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from mage.gui import GradioInterface, is_gradio_available


def main():
    """Launch the Gradio interface."""
    
    print("\n" + "=" * 60)
    print("MAGE Gradio Web Interface")
    print("=" * 60)
    
    # Check availability
    if not is_gradio_available():
        print("\n⚠️  Gradio not available")
        print("Install: pip install gradio>=4.0.0")
        return
    
    print("\nInitializing MAGE web interface...")
    
    try:
        # Create interface
        interface = GradioInterface(config_path="config/config.yaml")
        
        print("\n" + "=" * 60)
        print("Starting Web Server")
        print("=" * 60)
        print("\nFeatures available:")
        print("  📝 Lyrics Generation - AI-powered lyric generation")
        print("  🎼 Stem Separation - Extract vocals, drums, bass, other")
        print("  🎙️ Vocal Enhancement - AI-based vocal quality improvement")
        print("  🎚️ Audio Effects - Professional audio processing")
        print("  ℹ️  About - System information and credits")
        
        print("\n" + "=" * 60)
        print("Server Configuration")
        print("=" * 60)
        print("  Address: http://127.0.0.1:7860")
        print("  Share: No (local only)")
        print("  Debug: No")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        # Launch interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,  # Set to True for public sharing
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed:")
        print("     pip install -r requirements.txt")
        print("  2. Check config/config.yaml exists")
        print("  3. Verify output directory is writable")


if __name__ == "__main__":
    main()
