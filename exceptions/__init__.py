"""Custom exceptions for the MAGE system.

This module defines all custom exceptions used throughout the MAGE codebase
to provide clear error handling and debugging information.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class MAGEException(Exception):
    """Base exception class for all MAGE-related errors.
    
    All custom exceptions in MAGE inherit from this class to provide
    a consistent error handling interface.
    
    Attributes:
        message: The error message
        error_code: Optional error code for categorization
        details: Optional dictionary containing additional error context
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        """Initialize the MAGE exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional context about the error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
        super().__init__(self.message)
        logger.error(
            f"{self.__class__.__name__}: {message}",
            extra={"error_code": error_code, "details": details}
        )


class AudioGenerationError(MAGEException):
    """Exception raised when audio generation fails.
    
    This exception is raised when the audio generation process encounters
    an error that prevents it from completing successfully.
    """
    
    def __init__(
        self,
        message: str = "Audio generation failed",
        error_code: Optional[str] = "AUDIO_GEN_ERROR",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class ModelLoadError(MAGEException):
    """Exception raised when a model fails to load.
    
    This exception is raised when the system cannot load or initialize
    an AI model required for audio generation.
    """
    
    def __init__(
        self,
        message: str = "Failed to load model",
        error_code: Optional[str] = "MODEL_LOAD_ERROR",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class ConfigurationError(MAGEException):
    """Exception raised when configuration is invalid.
    
    This exception is raised when the configuration file is missing,
    malformed, or contains invalid values.
    """
    
    def __init__(
        self,
        message: str = "Invalid configuration",
        error_code: Optional[str] = "CONFIG_ERROR",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class AudioProcessingError(MAGEException):
    """Exception raised when audio processing fails.
    
    This exception is raised when post-processing operations on
    generated audio encounter errors.
    """
    
    def __init__(
        self,
        message: str = "Audio processing failed",
        error_code: Optional[str] = "AUDIO_PROC_ERROR",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class InvalidParameterError(MAGEException):
    """Exception raised when invalid parameters are provided.
    
    This exception is raised when function or method parameters
    are outside acceptable ranges or of incorrect types.
    """
    
    def __init__(
        self,
        message: str = "Invalid parameter value",
        error_code: Optional[str] = "INVALID_PARAM",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class ResourceNotFoundError(MAGEException):
    """Exception raised when a required resource is not found.
    
    This exception is raised when the system cannot locate a required
    file, model, or other resource.
    """
    
    def __init__(
        self,
        message: str = "Required resource not found",
        error_code: Optional[str] = "RESOURCE_NOT_FOUND",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class ExportError(MAGEException):
    """Exception raised when audio export fails.
    
    This exception is raised when the system cannot export generated
    audio to the specified format or location.
    """
    
    def __init__(
        self,
        message: str = "Failed to export audio",
        error_code: Optional[str] = "EXPORT_ERROR",
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
