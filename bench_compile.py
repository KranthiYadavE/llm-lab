import time, torch

from transformers import AutoModelForCausalLM, AutoTokenizer

m = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16).cuda().eval()

model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)


x= tok("The capital of France is", return_tensors= "pt").to("cuda")
N=200

print("warming up (Compiling)...")
model.generate(**x, max_new_tokens =20, do_sample=False)
torch.cuda.synchronize()

t0 = time.perf_counter()

model.generate(**x, max_new_tokens=N, min_new_tokens=N, do_sample=False)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print(f"{N/dt:.1f} tokens/sec (compiled)")

