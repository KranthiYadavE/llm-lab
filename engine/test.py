import torch

num_layers = 2
B = 3
num_kv_heads = 2
head_dim = 3
sequence_length = 4

past_key_values = tuple(
    (torch.randn(B, num_kv_heads, sequence_length, head_dim), torch.randn(B, num_kv_heads, sequence_length, head_dim))
    for _ in range(num_layers)
)


print(len(past_key_values))  # should print 2
print(past_key_values[0][0].shape)  

def slice_past_key_values(past_key_values, keep):
    new_cache = tuple(
        (key[keep], value[keep])
        for key, value in past_key_values
    )
    return new_cache
keep = torch.tensor([True, False, True])
sliced_cache = slice_past_key_values(past_key_values, keep)
print(len(sliced_cache))  # should print 2
print(sliced_cache[0][0].shape)  # should print (2, 2, 4, 3)