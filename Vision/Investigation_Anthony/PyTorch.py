import torch

print("Torch version: " + torch.__version__)

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")