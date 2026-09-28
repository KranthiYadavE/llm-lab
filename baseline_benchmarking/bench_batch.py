import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "Qwen/Qwen2.5-0.5B"

tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16).cuda().eval()

prompt = "The capital of France is"

N= 200

def run(batch_size):
    prompts= [prompt] * batch_size
    x = tok(prompts, return_tensors = "pt", padding=True).to("cuda")

    model.generate(**x, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()

    N =200
    t0 = time.perf_counter()
    model.generate(**x, max_new_tokens=N, min_new_tokens=N, do_sample=False)
    torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    total_tokens = N * batch_size
    print(f"batch={batch_size:3d}  total={total_tokens/dt:7.1f} tok/s  "
          f"per-sequence={total_tokens/dt/batch_size:6.1f} tok/s  time={dt:.2f}s")


for b in [1,4,16, 32, 64, 128, 256]:
    run(b)
