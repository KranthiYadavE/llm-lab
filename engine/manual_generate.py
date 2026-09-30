import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

m= "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16).cuda().eval()

prompt = "The capital of France is"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

N=30

@torch.no_grad()
def manual_generate(input_ids, n_new_tokens):

    generated = input_ids

    out = model(input_ids=generated, use_cache=True)
    past_key_values = out.past_key_values

    next_token = torch.argmax(out.logits[:,-1,:],dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)


    for _ in range(n_new_tokens-1):
        out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

        next_token_logits = out.logits[:,-1,:]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated


manual_out = manual_generate(input_ids, N)

print("manual_out:", tokenizer.decode(manual_out[0]))

hf_out = model.generate(input_ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
print("hf_out:", tokenizer.decode(hf_out[0]))

print("MATCH:", torch.equal(manual_out, hf_out))
