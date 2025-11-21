"""Command-line interface for MAGE.

This module provides a CLI for interacting with the MAGE system.
"""

import argparse
import sys
from pathlib import Path

from mage import MAGE, Config
from mage.utils import MAGELogger
from mage.exceptions import MAGEException
from mage.platform import print_device_info

logger = MAGELogger.get_logger(__name__)


def main():
    """Main entry point for the MAGE CLI."""
    parser = argparse.ArgumentParser(
        description="MAGE - Mixed Audio Generation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30 seconds of ambient music
  mage generate --duration 30 --style ambient --output ambient.wav
  
  # Generate with custom tempo and key
  mage generate --tempo 140 --key D_minor --output track.wav
  
  # Use custom configuration
  mage generate --config custom_config.yaml --output track.wav
  
  # List available styles
  mage list-styles
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate audio")
    gen_parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Duration in seconds"
    )
    gen_parser.add_argument(
        "--style", "-s",
        type=str,
        help="Music style"
    )
    gen_parser.add_argument(
        "--tempo", "-t",
        type=int,
        help="Tempo in BPM"
    )
    gen_parser.add_argument(
        "--key", "-k",
        type=str,
        help="Musical key"
    )
    gen_parser.add_argument(
        "--complexity", "-c",
        type=float,
        help="Complexity (0.0 to 1.0)"
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    gen_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path"
    )
    gen_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    gen_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    gen_parser.add_argument(
        "--reverb",
        type=float,
        default=0.0,
        help="Reverb amount (0.0 to 1.0)"
    )
    gen_parser.add_argument(
        "--compression",
        type=float,
        default=0.0,
        help="Compression amount (0.0 to 1.0)"
    )
    
    # List styles command
    subparsers.add_parser("list-styles", help="List available music styles")
    
    # List keys command
    subparsers.add_parser("list-keys", help="List available musical keys")
    
    # Info command
    subparsers.add_parser("info", help="Show system and device information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "generate":
            handle_generate(args)
        elif args.command == "list-styles":
            handle_list_styles(args)
        elif args.command == "list-keys":
            handle_list_keys(args)
        elif args.command == "info":
            handle_info(args)
    
    except MAGEException as e:
        logger.error(f"MAGE error: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def handle_generate(args):
    """Handle the generate command.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    # Override logging level if specified
    if args.log_level:
        config.logging.level = args.log_level
    
    # Initialize MAGE
    logger.info("Initializing MAGE engine...")
    engine = MAGE(config=config)
    
    # Generate audio
    logger.info("Generating audio...")
    audio = engine.generate(
        duration=args.duration,
        style=args.style,
        tempo=args.tempo,
        key=args.key,
        complexity=args.complexity,
        seed=args.seed
    )
    
    # Apply effects
    if args.reverb > 0 or args.compression > 0:
        logger.info("Applying effects...")
        audio.apply_effects(reverb=args.reverb, compression=args.compression)
    
    # Normalize
    audio.normalize()
    
    # Export
    logger.info(f"Exporting to {args.output}...")
    audio.export(args.output)
    
    logger.info("✓ Audio generation completed successfully!")
    print(f"\nGenerated audio saved to: {args.output}")
    print(f"Duration: {audio.duration:.2f} seconds")
    print(f"Sample rate: {audio.sample_rate} Hz")


def handle_list_styles(args):
    """Handle the list-styles command.
    
    Args:
        args: Command-line arguments
    """
    config = Config()
    engine = MAGE(config=config)
    
    styles = engine.get_available_styles()
    
    print("\nAvailable music styles:")
    print("-" * 40)
    for style in sorted(styles):
        print(f"  • {style}")
    print()


def handle_list_keys(args):
    """Handle the list-keys command.
    
    Args:
        args: Command-line arguments
    """
    config = Config()
    engine = MAGE(config=config)
    
    keys = engine.get_available_keys()
    
    print("\nAvailable musical keys:")
    print("-" * 40)
    for key in keys:
        print(f"  • {key}")
    print()


def handle_info(args):
    """Handle the info command.
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "=" * 60)
    print("MAGE System Information")
    print("=" * 60)
    
    # Display device information
    print_device_info()
    
    # Display configuration defaults
    config = Config()
    print("\nDefault Configuration:")
    print("-" * 60)
    print(f"Sample Rate:     {config.audio.sample_rate} Hz")
    print(f"Bit Depth:       {config.audio.bit_depth} bit")
    print(f"Channels:        {config.audio.channels}")
    print(f"Default Style:   {config.generation.default_style}")
    print(f"Default Tempo:   {config.generation.default_tempo} BPM")
    print(f"Default Key:     {config.generation.default_key}")
    print(f"Model Device:    {config.model.device}")
    print(f"Resolved Device: {config.model.get_device()}")
    print(f"Model Precision: {config.model.precision}")
    print(f"Log Level:       {config.logging.level}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
