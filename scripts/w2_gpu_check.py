"""GPU readiness check: device, versions, bf16 support, matmul throughput."""
import platform, sys, time
import torch

print("=" * 60)
print("System")
print("=" * 60)
print(f"Python   : {sys.version.split()[0]}")
print(f"Platform : {platform.platform()}")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA built-in : {torch.version.cuda}")
print(f"cuDNN         : {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("\n[FAIL] CUDA not available in this env. Check driver / pytorch install.")
    sys.exit(1)

print("\n" + "=" * 60)
print("GPU devices")
print("=" * 60)
n_gpu = torch.cuda.device_count()
print(f"device_count: {n_gpu}")
for i in range(n_gpu):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}")
    print(f"      compute capability : {props.major}.{props.minor}")
    print(f"      total memory       : {props.total_memory / 1024**3:.2f} GB")
    print(f"      multi-processors   : {props.multi_processor_count}")

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

print("\n" + "=" * 60)
print("Dtype support")
print("=" * 60)
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
print(f"tf32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")

print("\n" + "=" * 60)
print("Matmul throughput  (fp32 / fp16 / bf16, 4096x4096 @ 4096x4096)")
print("=" * 60)
N = 4096
for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    try:
        a = torch.randn(N, N, device=dev, dtype=dtype)
        b = torch.randn(N, N, device=dev, dtype=dtype)
        for _ in range(3):  # warmup
            c = a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 10
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tflops = 2 * N**3 / dt / 1e12
        print(f"  {dtype_name}: {dt*1000:7.2f} ms/iter   {tflops:6.1f} TFLOPS")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {dtype_name}: FAIL ({e})")

print("\n" + "=" * 60)
print("Memory snapshot")
print("=" * 60)
free, total = torch.cuda.mem_get_info()
print(f"free / total : {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")
print(f"allocated    : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"reserved     : {torch.cuda.memory_reserved()/1024**3:.2f} GB")

print("\n[OK] GPU is ready for training.")
