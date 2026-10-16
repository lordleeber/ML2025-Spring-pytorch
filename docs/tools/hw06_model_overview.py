# Model overview for chapter 0: load Llama-3.2-1B-Instruct in bf16 on the CPU (the GPU stays free)
# and print the config, the shape of every parameter in layer 0, parameter counts by group,
# and what the chat template turns a one-question chat into.
# usage: HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python docs/tools/hw06_model_overview.py
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = 'meta-llama/Llama-3.2-1B-Instruct'
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(NAME)
c = model.config
print(f'layers {c.num_hidden_layers}, hidden {c.hidden_size}, heads {c.num_attention_heads}, kv heads '
      f'{c.num_key_value_heads}, head_dim {c.head_dim}, mlp {c.intermediate_size}, vocab {c.vocab_size}, '
      f'tied embeddings {c.tie_word_embeddings}, rope_theta {c.rope_theta}')
print('--- layer 0')
for n, p in model.model.layers[0].named_parameters():
    print(f'{n:32} {str(tuple(p.shape)):16} {p.numel():>12,}')
groups = Counter()
for n, p in model.named_parameters():
    g = 'embed_tokens' if 'embed_tokens' in n else 'final norm' if n == 'model.norm.weight' else \
        'attention' if 'self_attn' in n else 'mlp' if 'mlp' in n else 'layer norms'
    groups[g] += p.numel()
print('--- parameters by group (lm_head shares embed_tokens)')
for g, k in groups.items():
    print(f'{g:14} {k:>14,}')
print(f'{"total":14} {sum(p.numel() for p in model.parameters()):>14,}')
print('lm_head is embed_tokens:', model.lm_head.weight is model.model.embed_tokens.weight)
print('--- chat template, one user turn, add_generation_prompt=True')
print(tok.apply_chat_template([{'role': 'user', 'content': 'Q: 1+1?'}], tokenize=False, add_generation_prompt=True))
