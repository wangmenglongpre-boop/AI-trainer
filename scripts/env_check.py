import platform, torch

print("=" * 50)
print("Python :", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA 不可用！检查驱动/torch 安装"

dev = torch.cuda.current_device()
print("GPU    :", torch.cuda.get_device_name(dev))
print("显存(GB):", round(torch.cuda.get_device_properties(dev).total_memory / 1024**3, 1))
print("bf16 支持:", torch.cuda.is_bf16_supported())

# GPU 矩阵乘法自检
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = (x @ y).sum().item()
torch.cuda.synchronize()
print("GPU matmul OK, sample sum =", round(z, 2))
print("=" * 50)
