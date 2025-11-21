"""Quick test script for GPU detection."""

from mage.platform import print_device_info, get_optimal_device, get_device_info
from mage.config import Config

print("=" * 60)
print("Testing MAGE GPU Detection")
print("=" * 60)

# Test 1: Print detailed device information
print("\n1. Detailed Device Information:")
print_device_info()

# Test 2: Get optimal device string
print("\n2. Optimal Device String:")
device = get_optimal_device()
print(f"   Optimal device: {device}")

# Test 3: Get device info dictionary
print("\n3. Device Info Dictionary:")
info = get_device_info()
for key, value in info.items():
    print(f"   {key}: {value}")

# Test 4: Test with Config
print("\n4. ModelConfig Integration:")
config = Config()
print(f"   Config device setting: {config.model.device}")
resolved_device = config.model.get_device()
print(f"   Resolved device: {resolved_device}")

# Test 5: Test with explicit device preference
print("\n5. Explicit Device Preferences:")
for pref in ["auto", "cuda", "cpu"]:
    device = get_optimal_device(pref)
    print(f"   Preferred '{pref}' -> {device}")

print("\n" + "=" * 60)
print("GPU Detection Test Complete")
print("=" * 60)
