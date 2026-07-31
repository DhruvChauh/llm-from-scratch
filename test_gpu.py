import torch

print("=" * 50)
print("PyTorch GPU Test")
print("=" * 50)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("VRAM:", round(vram, 2), "GB")

    print("=" * 50)
    print("GPU TEST PASSED")
    print("=" * 50)
else:
    print("=" * 50)
    print("GPU TEST FAILED")
    print("=" * 50)