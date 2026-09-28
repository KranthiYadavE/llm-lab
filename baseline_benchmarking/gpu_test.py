import torch, time
x = torch.empty(512*1024*1024, dtype= torch.uint8, device ="cuda")
y = torch.empty_like(x)

for _ in range(3): y.copy_(x)
torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(20): y.copy_(x)

torch.cuda.synchronize()

dt = time.perf_counter() - t0

print(f"{20 * 2 * 512e6 / dt / 1e9: .0f} GB/s achieveble")
