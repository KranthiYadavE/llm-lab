import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16).cuda().eval()

x = tok("The capital of france is", return_tensors="pt").to("cuda")

model.generate(**x, max_new_tokens=20, do_sample=False)
torch.cuda.synchronize()

N=200
t0 = time.perf_counter()
model.generate(**x,max_new_tokens =N, min_new_tokens=N, do_sample=False)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print(f"{N/dt:.1f} tokens/sec")

