"""Cross-platform GPU detection and device management.

This module provides utilities for detecting and selecting the optimal
compute device across different platforms and GPU vendors.
"""

import platform
from typing import Optional, Dict, Any
from dataclasses import dataclass

from mage.utils import MAGELogger

logger = MAGELogger.get_logger(__name__)


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    vendor: str  # "nvidia", "amd", "apple", "cpu"
    compute_capability: Optional[str] = None
    memory_total: Optional[int] = None  # In bytes
    is_available: bool = True


class GPUDetector:
    """Detect and manage GPU/compute devices across platforms."""
    
    def __init__(self):
        """Initialize the GPU detector."""
        self._device_info: Optional[DeviceInfo] = None
        self._torch_available = False
        
        try:
            import torch
            self._torch_available = True
            self._torch = torch
            logger.debug("PyTorch available for GPU detection")
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
    
    def detect_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Detect the best available compute device.
        
        Args:
            preferred_device: Preferred device ("auto", "cuda", "mps", "cpu")
            
        Returns:
            DeviceInfo object with device information
        """
        logger.debug(f"Detecting device with preference: {preferred_device}")
        
        if not self._torch_available:
            return self._get_cpu_info()
        
        if preferred_device == "cpu":
            return self._get_cpu_info()
        
        # Try CUDA (works for both NVIDIA and AMD with ROCm)
        if preferred_device in ["auto", "cuda"]:
            cuda_info = self._detect_cuda()
            if cuda_info.is_available:
                self._device_info = cuda_info
                return cuda_info
        
        # Try MPS (Apple Silicon)
        if preferred_device in ["auto", "mps"]:
            mps_info = self._detect_mps()
            if mps_info.is_available:
                self._device_info = mps_info
                return mps_info
        
        # Fallback to CPU
        self._device_info = self._get_cpu_info()
        return self._device_info
    
    def _detect_cuda(self) -> DeviceInfo:
        """Detect CUDA device (NVIDIA or AMD with ROCm).
        
        Returns:
            DeviceInfo for CUDA device
        """
        try:
            if not self._torch.cuda.is_available():
                logger.debug("CUDA not available")
                return DeviceInfo(
                    device_type="cuda",
                    device_name="N/A",
                    vendor="none",
                    is_available=False
                )
            
            # Detect vendor
            vendor = "nvidia"  # Default
            device_name = "NVIDIA GPU"
            
            # Check if ROCm is being used
            if hasattr(self._torch.version, 'hip') and self._torch.version.hip:
                vendor = "amd"
                device_name = "AMD GPU (ROCm)"
                logger.info(f"Detected AMD GPU with ROCm support")
            else:
                # NVIDIA CUDA
                device_name = self._torch.cuda.get_device_name(0)
                logger.info(f"Detected NVIDIA GPU: {device_name}")
            
            # Get compute capability (NVIDIA only)
            capability = None
            if vendor == "nvidia":
                try:
                    major, minor = self._torch.cuda.get_device_capability(0)
                    capability = f"{major}.{minor}"
                except Exception as e:
                    logger.warning(f"Could not get compute capability: {e}")
            
            # Get memory info
            try:
                memory_total = self._torch.cuda.get_device_properties(0).total_memory
            except Exception as e:
                logger.warning(f"Could not get GPU memory: {e}")
                memory_total = None
            
            device_info = DeviceInfo(
                device_type="cuda",
                device_name=device_name,
                vendor=vendor,
                compute_capability=capability,
                memory_total=memory_total,
                is_available=True
            )
            
            if memory_total:
                logger.info(
                    f"CUDA device initialized: {device_name} "
                    f"({memory_total / 1024**3:.2f} GB)"
                )
            else:
                logger.info(f"CUDA device initialized: {device_name}")
            
            return device_info
            
        except Exception as e:
            logger.error(f"Error detecting CUDA device: {e}", exc_info=True)
            return DeviceInfo(
                device_type="cuda",
                device_name="Unknown GPU",
                vendor="unknown",
                is_available=False
            )
    
    def _detect_mps(self) -> DeviceInfo:
        """Detect Apple Metal Performance Shaders (MPS) device.
        
        Returns:
            DeviceInfo for MPS device
        """
        try:
            if (hasattr(self._torch.backends, 'mps') and 
                self._torch.backends.mps.is_available()):
                
                logger.info("Detected Apple Silicon with MPS support")
                
                return DeviceInfo(
                    device_type="mps",
                    device_name="Apple Silicon GPU",
                    vendor="apple",
                    is_available=True
                )
        except Exception as e:
            logger.debug(f"MPS not available: {e}")
        
        return DeviceInfo(
            device_type="mps",
            device_name="N/A",
            vendor="apple",
            is_available=False
        )
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information.
        
        Returns:
            DeviceInfo for CPU
        """
        cpu_name = platform.processor() or "CPU"
        memory_total = None
        
        try:
            import psutil
            memory_total = psutil.virtual_memory().total
            logger.debug(f"CPU memory: {memory_total / 1024**3:.2f} GB")
        except ImportError:
            logger.debug("psutil not available for memory detection")
        except Exception as e:
            logger.debug(f"Could not get CPU memory: {e}")
        
        logger.info(f"Using CPU: {cpu_name}")
        
        return DeviceInfo(
            device_type="cpu",
            device_name=cpu_name,
            vendor="cpu",
            memory_total=memory_total,
            is_available=True
        )
    
    def get_device_string(self) -> str:
        """Get PyTorch device string.
        
        Returns:
            Device string ("cuda", "mps", "cpu")
        """
        if self._device_info is None:
            self.detect_device()
        
        return self._device_info.device_type if self._device_info.is_available else "cpu"
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information as dictionary.
        
        Returns:
            Dictionary with device information
        """
        if self._device_info is None:
            self.detect_device()
        
        return {
            "type": self._device_info.device_type,
            "name": self._device_info.device_name,
            "vendor": self._device_info.vendor,
            "compute_capability": self._device_info.compute_capability,
            "memory_gb": (
                self._device_info.memory_total / 1024**3 
                if self._device_info.memory_total else None
            ),
            "available": self._device_info.is_available
        }
    
    def print_device_info(self) -> None:
        """Print detailed device information."""
        info = self.get_device_info()
        
        print("\n" + "="*50)
        print("MAGE Compute Device Information")
        print("="*50)
        print(f"Device Type:  {info['type'].upper()}")
        print(f"Device Name:  {info['name']}")
        print(f"Vendor:       {info['vendor'].upper()}")
        
        if info['compute_capability']:
            print(f"Compute Cap:  {info['compute_capability']}")
        
        if info['memory_gb']:
            print(f"Memory:       {info['memory_gb']:.2f} GB")
        
        print(f"Available:    {'Yes' if info['available'] else 'No'}")
        print("="*50 + "\n")


# Global detector instance
_detector = None


def get_gpu_detector() -> GPUDetector:
    """Get the global GPU detector instance.
    
    Returns:
        GPUDetector instance
    """
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def get_optimal_device(preferred: str = "auto") -> str:
    """Get the optimal device string for PyTorch.
    
    Args:
        preferred: Preferred device ("auto", "cuda", "mps", "cpu")
        
    Returns:
        Device string for PyTorch
    """
    detector = get_gpu_detector()
    device_info = detector.detect_device(preferred)
    return device_info.device_type if device_info.is_available else "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get information about the current compute device.
    
    Returns:
        Dictionary with device information
    """
    detector = get_gpu_detector()
    return detector.get_device_info()


def print_device_info() -> None:
    """Print information about the current compute device."""
    detector = get_gpu_detector()
    detector.print_device_info()
