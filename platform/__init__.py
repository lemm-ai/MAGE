"""Platform and GPU detection utilities."""

from mage.platform.gpu_detect import (
    GPUDetector,
    DeviceInfo,
    get_gpu_detector,
    get_optimal_device,
    get_device_info,
    print_device_info,
)

__all__ = [
    "GPUDetector",
    "DeviceInfo",
    "get_gpu_detector",
    "get_optimal_device",
    "get_device_info",
    "print_device_info",
]
