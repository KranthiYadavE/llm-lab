import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM


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
# test_token_id = tokenizer.encode(" the", add_special_tokens=False)[0]
# eos_id = test_token_id
eos_id = tokenizer.eos_token_id


enc = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids = enc.input_ids.cuda()
attention_mask = enc.attention_mask.cuda()

@torch.no_grad()
def manual_generate_batch(input_ids, attention_mask, n_new_tokens):
    batch_size, seq_len = input_ids.shape
    generated = input_ids
    finished = torch.zeros(batch_size, dtype= torch.bool, device="cuda")


    out = model(input_ids=generated, attention_mask=attention_mask, use_cache=True)
    past_key_values = out.past_key_values

    next_token = torch.argmax(out.logits[:,-1,:], dim=-1, keepdim=True)
    next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id),next_token)
    finished = finished | (next_token.squeeze(-1)==eos_id)
    generated = torch.cat([generated, next_token],dim=-1)

    attention_mask = torch.cat(
       [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device="cuda")], dim=-1,
)

    for _ in range(n_new_tokens-1):
        if finished.all():
            break
        out = model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
    
        next_token_logits = out.logits[:,-1,:]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id),next_token)
        finished = finished | (next_token.squeeze(-1)==eos_id)
        generated = torch.cat([generated, next_token], dim=-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device="cuda")],
            dim=-1,
        )
    
    return generated, finished

manual_out, manual_finished = manual_generate_batch(input_ids, attention_mask, n_new_tokens=30)
print("manual_finished:", manual_finished)
for i, seq in enumerate(manual_out):
    print(f"Prompt: {prompts[i]}")
    print(f"Manual Output: {tokenizer.decode(seq, skip_special_tokens=True)}")
    print("-" * 50)

hf_out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=30, min_new_tokens=30, do_sample=False)
print("hf_out:", tokenizer.decode(hf_out[0], skip_special_tokens=True))

print("MATCH:", torch.equal(manual_out, hf_out))