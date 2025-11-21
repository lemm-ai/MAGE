"""Example: Launch MAGE Udio-style interface.

This example demonstrates the professional DAW-like timeline workflow
with clip library management and advanced controls.

Usage:
    Run from the MAGE root directory:
    python examples/udio_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from mage.gui import UdioInterface, is_udio_available


def main():
    """Launch the Udio-style interface."""
    
    print("\n" + "=" * 60)
    print("MAGE Udio-Style Interface")
    print("=" * 60)
    
    # Check availability
    if not is_udio_available():
        print("\n⚠️  Gradio not available")
        print("Install: pip install gradio>=4.0.0")
        return
    
    print("\nInitializing MAGE Udio-style interface...")
    
    # Create interface
    udio = UdioInterface()
    
    print("\n" + "=" * 60)
    print("Starting Web Server")
    print("=" * 60)
    
    print("\nFeatures available:")
    print("  🎵 AI Music Generation - Timeline-based workflow")
    print("  📝 Lyrics Generation - AI-powered lyrics")
    print("  📚 Clip Library - Manage all your clips")
    print("  🎹 Song Timeline - Visual waveform display")
    print("  🎛️ DAW Effects - Professional EQ, compression, reverb, limiting")
    print("  🎙️ Vocal Enhancement - Advanced vocal processing")
    
    print("\n" + "=" * 60)
    print("Server Configuration")
    print("=" * 60)
    print("  Address: http://127.0.0.1:7861")
    print("  Share: No (local only)")
    print("  Debug: No")
    
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Launch interface
    try:
        udio.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\n\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
