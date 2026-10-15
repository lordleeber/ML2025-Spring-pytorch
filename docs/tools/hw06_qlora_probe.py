# QLoRA facts for chapter 2: build the model exactly like hw6.py's load_model() and report which
# modules are 4-bit, what stays in 16-bit, GPU memory, the LoRA parameter count per module and dtype,
# how A and B start, and what the extra [PAD] token does to the embedding (a lookup of id 128256).
# usage: HF_HUB_OFFLINE=1 python docs/tools/hw06_qlora_probe.py
import collections
import hashlib
import sys

import torch

sys.path.insert(0, 'HW06')
import hw6  # noqa: E402

torch.manual_seed(0)
sft_model, tok, peft_config, peft_model = hw6.load_model(None)
print('GPU memory allocated after load_model (GB):', round(torch.cuda.memory_allocated() / 1e9, 3))
kinds = collections.Counter(type(m).__name__ for n, m in sft_model.named_modules()
                            if n.endswith(('_proj', 'embed_tokens', 'lm_head')) or n.endswith(('.base_layer',)))
print('module types (projections, base layers, embeddings, lm_head):', dict(kinds))
print('embed_tokens', type(sft_model.model.embed_tokens).__name__, sft_model.model.embed_tokens.weight.dtype,
      '| lm_head', type(sft_model.lm_head).__name__, sft_model.lm_head.weight.dtype)
norm = sft_model.model.layers[0].input_layernorm.weight
print('layer norm dtype', norm.dtype)
q = sft_model.model.layers[0].self_attn.q_proj
print('q_proj:', type(q).__name__, '| base layer', type(q.base_layer).__name__, '| base weight', q.base_layer.weight.dtype, tuple(q.base_layer.weight.shape))
print('lora_A', tuple(q.lora_A['default'].weight.shape), q.lora_A['default'].weight.dtype,
      '| lora_B', tuple(q.lora_B['default'].weight.shape), '| scaling', q.scaling['default'])
print('B all zero:', all(torch.count_nonzero(p) == 0 for n, p in peft_model.named_parameters() if 'lora_B' in n))
per = collections.Counter()
for n, p in peft_model.named_parameters():
    if p.requires_grad:
        per[n.split('.')[-4]] += p.numel()
print('trainable per module type (all 16 layers):', dict(per))
peft_model.print_trainable_parameters()
h = hashlib.sha256()
for n, p in peft_model.named_parameters():
    if 'lora_A' in n:
        h.update(p.detach().float().cpu().numpy().tobytes())
print('LoRA A init sha256 with torch.manual_seed(0):', h.hexdigest()[:16])
print('tokenizer size', len(tok), 'pad id', tok.pad_token_id, 'embedding rows', sft_model.get_input_embeddings().weight.shape[0])
try:
    with torch.no_grad():
        sft_model(torch.tensor([[tok.bos_token_id, tok.pad_token_id]], device='cuda'))
    print('forward with [PAD]: ok')
except Exception as e:
    print('forward with [PAD]:', type(e).__name__, str(e).split('\n')[0][:120])
