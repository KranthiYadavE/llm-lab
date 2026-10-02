from unittest import result

import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers


model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda().eval()

prompts = [
    "The capital of France is",
    "Once upon a time there was",
    "2 + 2 =",
]

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
test_token_id = tokenizer.encode(" the", add_special_tokens=False)[0]
eos_id = test_token_id
#eos_id = tokenizer.eos_token_id

from transformers.cache_utils import DynamicCache

def slice_past_key_values(past_key_values, keep):
    for layer in past_key_values.layers:
        layer.keys = layer.keys[keep]
        layer.values = layer.values[keep]
    return past_key_values

enc = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids = enc.input_ids.cuda()
attention_mask = enc.attention_mask.cuda()

@torch.no_grad()
def manual_generate_batch(input_ids, attention_mask, n_new_tokens):
    batch_size, seq_len = input_ids.shape
    finished = torch.zeros(batch_size, dtype= torch.bool, device="cuda")
    active_index = list(range(batch_size))

    token_history = {idx: input_ids[row].tolist() for row, idx in enumerate(active_index)}
    result={}
    batch_size_log = []
    
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = out.past_key_values

    next_token = torch.argmax(out.logits[:,-1,:], dim=-1, keepdim=True)
    next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id),next_token)
    finished = finished | (next_token.squeeze(-1)==eos_id)
    

    attention_mask = torch.cat(
       [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device="cuda")], dim=-1,
)
    
    for _ in range(n_new_tokens-1):
        if len(active_index) == 0:
            break
        batch_size_log.append(len(active_index))
        
        if finished.any():
            keep = ~finished
            for row in finished.nonzero(as_tuple=True)[0].tolist():
                result[active_index[row]] = token_history[active_index[row]]

            next_token = next_token[keep]
         
            attention_mask = attention_mask[keep]
            past_key_values = slice_past_key_values(past_key_values, keep)
            active_index = [idx for idx, k in zip(active_index, keep.tolist()) if k]
            finished = finished[keep]

        out = model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
    
        next_token_logits = out.logits[:,-1,:]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id),next_token)
        finished = finished | (next_token.squeeze(-1)==eos_id)

        for row, idx in enumerate(active_index):
            token_history[idx].append(next_token[row].item())

        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device="cuda")],
            dim=-1,
        )
    
    for idx in active_index:
        result[idx] = token_history[idx]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(batch_size_log)), batch_size_log, marker="o", linewidth=2)
    plt.xlabel("Decode step")
    plt.ylabel("Active batch size")
    plt.title("Batch size shrinks as sequences finish")
    plt.yticks(range(0, max(batch_size_log) + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("batch_shrink.png", dpi=150)
    print("saved batch_shrink.png")
    return result

manual_result = manual_generate_batch(input_ids, attention_mask, n_new_tokens=30)
for i in range(len(prompts)):
    print(f"Prompt: {prompts[i]}")
    print(f"Manual Output: {tokenizer.decode(manual_result[i], skip_special_tokens=True)}")
    print("-" * 50)
hf_out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=30, min_new_tokens=30, do_sample=False)
print("hf_out:", tokenizer.decode(hf_out[0], skip_special_tokens=True))

