# Why does the split HW06/hw6.py diverge from the notebook at the 8th GSM8K answer even though
# every checkpoint is identical? Suspect: the notebook keeps the PeftModel returned by
# PeftModel.from_pretrained alive (as `pipeline.model`), hw6.py first discarded it; once it is
# garbage-collected something about the model may change. This probe evaluates one checkpoint
# on the first N public questions with the result kept vs discarded (plus an explicit
# gc.collect() variant), using hw6.py's own functions and the same RNG sequence each time.
# usage: HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw06_gc_probe.py
#        <checkpoint dir> keep|discard|discard_gc [N]
import gc
import random
import sys

import torch

torch.use_deterministic_algorithms(True)
sys.path.insert(0, 'HW06')
import hw6  # noqa: E402  (seeds random with 42 at import, as the notebook does)

adapter, mode = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
torch.manual_seed(0)


class A:
    max_new_tokens, greedy = 256, False


sft_model, tok, _, _ = hw6.load_model(None)
train = hw6.load_jsonlines('HW06/data/gsm8k_train.jsonl')
hw6.format_train_data(train, tok, 1)
gen = hw6.pipeline('text-generation', model=sft_model, tokenizer=tok, pad_token_id=tok.eos_token_id,
                   max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9)
kept = hw6.PeftModel.from_pretrained(sft_model, adapter)
if mode != 'keep':
    del kept
if mode == 'discard_gc':
    gc.collect()
print('lora_B layer0 q_proj sum after load:',
      dict(sft_model.named_parameters())['model.layers.0.self_attn.q_proj.lora_B.default.weight'].sum().item())
pub = hw6.load_jsonlines('HW06/data/gsm8k_test_public.jsonl')[:n]
preds = []
for i, q in enumerate(pub):
    r = hw6.get_response(gen, hw6.nshot_chats(train, 1, q['question'], None, 'test'))
    preds.append(hw6.extract_ans_from_response(r))
    if i == 6:
        gc.collect()  # force a collection mid-run, like the one that may happen in a long run
print(mode, preds)
print('lora_B layer0 q_proj sum at end:',
      dict(sft_model.named_parameters())['model.layers.0.self_attn.q_proj.lora_B.default.weight'].sum().item())
print('lora modules active adapters:', {type(m).__name__ for m in sft_model.modules() if hasattr(m, 'active_adapters')} )
