# Does Python's `random` reach the evaluation untouched by training? transformers' Trainer calls
# set_seed(1126) in __init__ (random.seed included). This probe trains a tiny random Llama on the CPU
# with the same SFTConfig as hw6.py (3 steps, LoRA via peft_config as in the notebook) and compares
# random.getstate() after trainer.train() with the state right after random.seed(1126).
# usage: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python docs/tools/hw06_pyrandom_probe.py
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer

tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
tok.add_special_tokens({'pad_token': '[PAD]'})
model = LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
cfg = LoraConfig(r=8, lora_alpha=16, bias='none', task_type='CAUSAL_LM',
                 target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'])
pm = get_peft_model(model, cfg)
ds = Dataset.from_list([{'text': f'Q: {i} + {i}? A: {2 * i}'} for i in range(64)])
random.seed(42)
for _ in range(100):
    random.random()  # stand-in for the formatting loop's draws
args = SFTConfig(seed=1126, data_seed=1126, output_dir='/tmp/hw06_pyrandom_probe', per_device_train_batch_size=1,
                 gradient_accumulation_steps=4, optim='adamw_torch', max_steps=3, logging_steps=1, save_strategy='no',
                 lr_scheduler_type='linear', learning_rate=2e-4, group_by_length=True, max_seq_length=64,
                 dataset_text_field='text', report_to='none', use_cpu=True)
trainer = SFTTrainer(model=pm, train_dataset=ds, peft_config=cfg, processing_class=tok, args=args)
after_init = random.getstate()
trainer.train()
after_train = random.getstate()
random.seed(1126)
fresh = random.getstate()
print('state after SFTTrainer() == state after random.seed(1126):', after_init == fresh)
print('state after trainer.train() == state after random.seed(1126):', after_train == fresh)
