import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA version: N/A")
    
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
