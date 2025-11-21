"""Lyrics generation using LyricMind-AI.

This module provides the LyricGenerator class that integrates with LyricMind-AI
for generating song lyrics based on themes, genres, and user input.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn

from mage.utils import MAGELogger
from mage.exceptions import ModelLoadError, AudioGenerationError, InvalidParameterError

logger = MAGELogger.get_logger(__name__)


@dataclass
class LyricConfig:
    """Configuration for lyrics generation."""
    
    model_path: Optional[str] = None
    cache_dir: str = "models/lyricmind"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 1


@dataclass
class GeneratedLyrics:
    """Container for generated lyrics."""
    
    text: str
    genre: str
    theme: Optional[str] = None
    structure: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_lines(self) -> List[str]:
        """Get lyrics as list of lines.
        
        Returns:
            List of lyric lines
        """
        return [line.strip() for line in self.text.split('\n') if line.strip()]
    
    def get_sections(self) -> Dict[str, str]:
        """Parse lyrics into sections (verse, chorus, etc.).
        
        Returns:
            Dictionary mapping section names to text
        """
        sections = {}
        current_section = "verse_1"
        current_lines = []
        
        for line in self.get_lines():
            line_lower = line.lower()
            
            # Check for section markers
            if any(marker in line_lower for marker in ['[verse', '[chorus', '[bridge', '[pre-chorus', '[outro', '[intro']):
                # Save previous section
                if current_lines:
                    sections[current_section] = '\n'.join(current_lines)
                    current_lines = []
                
                # Extract new section name
                current_section = line_lower.strip('[]').replace(' ', '_')
            else:
                current_lines.append(line)
        
        # Save final section
        if current_lines:
            sections[current_section] = '\n'.join(current_lines)
        
        return sections


class SimpleLSTMLyricModel(nn.Module):
    """Simple LSTM model for lyrics generation.
    
    This is a placeholder that mimics LyricMind-AI's architecture.
    In production, this would load the actual LyricMind-AI model.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2):
        """Initialize the LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, x, hidden=None):
        """Forward pass.
        
        Args:
            x: Input tensor
            hidden: Hidden state
            
        Returns:
            Tuple of (output, hidden)
        """
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden


class LyricGenerator:
    """Generator for AI-powered lyrics using LyricMind-AI."""
    
    def __init__(self, config: Optional[LyricConfig] = None, device: Optional[str] = None):
        """Initialize the lyric generator.
        
        Args:
            config: Lyrics generation configuration
            device: Device to use ("cuda", "cpu", "mps")
            
        Raises:
            ModelLoadError: If model fails to load
        """
        self.config = config or LyricConfig()
        self._device = device or self.config.device
        self._model = None
        self._tokenizer = None
        self._vocab = None
        self._model_loaded = False
        
        logger.info("Initializing LyricGenerator")
        
        # Create cache directory
        try:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise ModelLoadError(
                f"Failed to create cache directory: {e}",
                details={"cache_dir": self.config.cache_dir}
            )
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded.
        
        Raises:
            ModelLoadError: If model fails to load
        """
        if self._model_loaded:
            return
        
        logger.info("Loading LyricMind-AI model...")
        
        try:
            # Resolve device
            if self._device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            
            logger.info(f"Using device: {self._device}")
            
            # Check if model exists
            model_path = self._get_model_path()
            
            if model_path and Path(model_path).exists():
                logger.info(f"Loading model from {model_path}")
                self._load_pretrained_model(model_path)
            else:
                logger.warning("Pretrained model not found, using placeholder model")
                self._load_placeholder_model()
            
            self._model_loaded = True
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(
                f"Failed to load LyricMind-AI model: {e}",
                details={"device": self._device, "error": str(e)}
            )
    
    def _get_model_path(self) -> Optional[str]:
        """Get the model path.
        
        Returns:
            Model path or None
        """
        if self.config.model_path:
            return self.config.model_path
        
        # Check cache directory
        cache_path = Path(self.config.cache_dir)
        model_file = cache_path / "lyricmind_model.pt"
        
        if model_file.exists():
            return str(model_file)
        
        return None
    
    def _load_pretrained_model(self, model_path: str) -> None:
        """Load a pretrained model.
        
        Args:
            model_path: Path to model file
            
        Raises:
            ModelLoadError: If loading fails
        """
        try:
            checkpoint = torch.load(model_path, map_location=self._device)
            
            # Extract model parameters
            vocab_size = checkpoint.get('vocab_size', 10000)
            
            self._model = SimpleLSTMLyricModel(vocab_size=vocab_size)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._device)
            self._model.eval()
            
            # Load vocabulary
            self._vocab = checkpoint.get('vocab', self._create_default_vocab())
            
            logger.info(f"Loaded model with vocab size: {vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise ModelLoadError(
                f"Failed to load pretrained model: {e}",
                details={"model_path": model_path}
            )
    
    def _load_placeholder_model(self) -> None:
        """Load a placeholder model for testing."""
        logger.info("Loading placeholder model")
        
        self._model = SimpleLSTMLyricModel(vocab_size=10000)
        self._model.to(self._device)
        self._model.eval()
        
        self._vocab = self._create_default_vocab()
        
        logger.debug("Placeholder model loaded")
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """Create a default vocabulary.
        
        Returns:
            Vocabulary dictionary
        """
        # Basic vocabulary for placeholder
        common_words = [
            '<PAD>', '<START>', '<END>', '<UNK>',
            'love', 'heart', 'night', 'day', 'dream', 'soul', 'life', 'time',
            'feel', 'know', 'take', 'make', 'give', 'fall', 'rise', 'shine',
            'the', 'and', 'you', 'me', 'we', 'in', 'on', 'at', 'to', 'for',
            'verse', 'chorus', 'bridge', 'i', 'a', 'is', 'was', 'are', 'were'
        ]
        return {word: idx for idx, word in enumerate(common_words)}
    
    def generate(self, 
                 theme: Optional[str] = None,
                 genre: str = "pop",
                 structure: Optional[List[str]] = None,
                 prompt: Optional[str] = None,
                 max_lines: int = 16) -> GeneratedLyrics:
        """Generate lyrics based on theme and genre.
        
        Args:
            theme: Theme or topic for lyrics
            genre: Music genre (pop, rock, hip-hop, country, etc.)
            structure: Song structure (e.g., ["verse", "chorus", "verse", "chorus"])
            prompt: Optional starting prompt
            max_lines: Maximum number of lines to generate
            
        Returns:
            GeneratedLyrics object
            
        Raises:
            AudioGenerationError: If generation fails
            InvalidParameterError: If parameters are invalid
        """
        logger.info(f"Generating lyrics - Genre: {genre}, Theme: {theme}")
        
        # Validate parameters
        if max_lines < 1 or max_lines > 100:
            raise InvalidParameterError(
                f"max_lines must be between 1 and 100: {max_lines}"
            )
        
        valid_genres = ['pop', 'rock', 'hip-hop', 'country', 'r&b', 'electronic', 'folk', 'jazz', 'metal']
        if genre.lower() not in valid_genres:
            logger.warning(f"Unknown genre '{genre}', using 'pop'")
            genre = 'pop'
        
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Generate lyrics
            if structure:
                lyrics_text = self._generate_structured(theme, genre, structure, prompt, max_lines)
            else:
                lyrics_text = self._generate_freeform(theme, genre, prompt, max_lines)
            
            # Create result
            result = GeneratedLyrics(
                text=lyrics_text,
                genre=genre,
                theme=theme,
                structure=structure,
                metadata={
                    'prompt': prompt,
                    'max_lines': max_lines,
                    'device': self._device,
                }
            )
            
            logger.info(f"✓ Generated {len(result.get_lines())} lines of lyrics")
            
            return result
            
        except Exception as e:
            logger.error(f"Lyrics generation failed: {e}", exc_info=True)
            raise AudioGenerationError(
                f"Failed to generate lyrics: {e}",
                details={'theme': theme, 'genre': genre, 'error': str(e)}
            )
    
    def _generate_structured(self, theme: Optional[str], genre: str, 
                            structure: List[str], prompt: Optional[str], 
                            max_lines: int) -> str:
        """Generate lyrics with specific structure.
        
        Args:
            theme: Theme for lyrics
            genre: Music genre
            structure: Song structure
            prompt: Starting prompt
            max_lines: Maximum lines
            
        Returns:
            Generated lyrics text
        """
        logger.debug(f"Generating structured lyrics with {len(structure)} sections")
        
        sections = []
        lines_per_section = max(4, max_lines // len(structure))
        
        for section_type in structure:
            section_lines = self._generate_section(section_type, theme, genre, lines_per_section)
            sections.append(f"[{section_type.upper()}]\n{section_lines}")
        
        return "\n\n".join(sections)
    
    def _generate_freeform(self, theme: Optional[str], genre: str, 
                          prompt: Optional[str], max_lines: int) -> str:
        """Generate freeform lyrics.
        
        Args:
            theme: Theme for lyrics
            genre: Music genre
            prompt: Starting prompt
            max_lines: Maximum lines
            
        Returns:
            Generated lyrics text
        """
        logger.debug(f"Generating freeform lyrics ({max_lines} lines max)")
        
        # For placeholder, generate simple themed lyrics
        lines = []
        
        if prompt:
            lines.append(prompt)
        
        # Generate based on theme and genre
        theme_words = self._get_theme_words(theme, genre)
        
        while len(lines) < max_lines:
            line = self._generate_line(theme_words, len(lines))
            lines.append(line)
        
        return "\n".join(lines)
    
    def _generate_section(self, section_type: str, theme: Optional[str], 
                         genre: str, num_lines: int) -> str:
        """Generate a song section.
        
        Args:
            section_type: Type of section (verse, chorus, etc.)
            theme: Theme for lyrics
            genre: Music genre
            num_lines: Number of lines to generate
            
        Returns:
            Section lyrics
        """
        theme_words = self._get_theme_words(theme, genre)
        lines = []
        
        for i in range(num_lines):
            line = self._generate_line(theme_words, i, section_type)
            lines.append(line)
        
        return "\n".join(lines)
    
    def _generate_line(self, theme_words: List[str], line_num: int, 
                      section_type: str = "verse") -> str:
        """Generate a single line.
        
        Args:
            theme_words: Words related to theme
            line_num: Line number
            section_type: Section type
            
        Returns:
            Generated line
        """
        # Placeholder implementation
        # In production, this would use the model
        import random
        
        templates = [
            f"In the {random.choice(theme_words)}, I find my way",
            f"Through the {random.choice(theme_words)}, night and day",
            f"When the {random.choice(theme_words)} calls my name",
            f"I feel the {random.choice(theme_words)}, burning flame",
            f"Lost in {random.choice(theme_words)}, so far away",
            f"Dancing with {random.choice(theme_words)}, come what may",
        ]
        
        return random.choice(templates)
    
    def _get_theme_words(self, theme: Optional[str], genre: str) -> List[str]:
        """Get words related to theme and genre.
        
        Args:
            theme: Theme
            genre: Genre
            
        Returns:
            List of theme words
        """
        # Default theme words
        words = ['love', 'dreams', 'light', 'stars', 'heart', 'soul', 'time', 'hope']
        
        # Add theme-specific words
        if theme:
            words.extend(theme.lower().split())
        
        # Add genre-specific words
        genre_words = {
            'rock': ['thunder', 'fire', 'power', 'rebel', 'storm'],
            'pop': ['dance', 'shine', 'bright', 'party', 'feel'],
            'country': ['road', 'home', 'fields', 'sunset', 'freedom'],
            'hip-hop': ['street', 'flow', 'rhythm', 'hustle', 'real'],
            'r&b': ['groove', 'smooth', 'vibe', 'passion', 'desire'],
        }
        
        words.extend(genre_words.get(genre.lower(), []))
        
        return words
    
    def save_lyrics(self, lyrics: GeneratedLyrics, output_path: str) -> None:
        """Save lyrics to file.
        
        Args:
            lyrics: Generated lyrics
            output_path: Output file path
            
        Raises:
            AudioGenerationError: If saving fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Genre: {lyrics.genre}\n")
                if lyrics.theme:
                    f.write(f"Theme: {lyrics.theme}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(lyrics.text)
            
            logger.info(f"✓ Lyrics saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save lyrics: {e}")
            raise AudioGenerationError(
                f"Failed to save lyrics: {e}",
                details={'output_path': str(output_path)}
            )
    
    def clone_lyricmind_repo(self) -> bool:
        """Clone LyricMind-AI repository to local directory.
        
        Returns:
            True if successful, False otherwise
        """
        repo_url = "https://github.com/BenZuckier/LyricMind-AI.git"
        clone_dir = Path(self.config.cache_dir) / "LyricMind-AI"
        
        if clone_dir.exists():
            logger.info(f"LyricMind-AI already cloned at {clone_dir}")
            return True
        
        try:
            logger.info(f"Cloning LyricMind-AI from {repo_url}")
            
            result = subprocess.run(
                ['git', 'clone', repo_url, str(clone_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully cloned LyricMind-AI to {clone_dir}")
                return True
            else:
                logger.error(f"Failed to clone repository: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Repository clone timed out after 5 minutes")
            return False
        except FileNotFoundError:
            logger.error("Git is not installed or not in PATH")
            return False
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
