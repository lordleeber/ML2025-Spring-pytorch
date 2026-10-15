# Does the notebook's cell [21] really evaluate the checkpoint named in adapter_path?
# It does `pipeline.model = PeftModel.from_pretrained(sft_model, adapter_path)`, which sets an
# attribute on the transformers.pipeline function, not on `generator`. This probe rebuilds the
# notebook's objects (cell [12]: 4-bit model + get_peft_model, which injects LoRA into sft_model
# in place), stands in for "weights after training" with a recognisable LoRA B (all 0.01), then
# runs cell [21] verbatim and asks which LoRA weights `generator` actually holds.
# usage: HF_HUB_OFFLINE=1 python docs/tools/hw06_adapter_probe.py <checkpoint dir>
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

adapter_path = sys.argv[1]
sft_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B-Instruct', low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                           bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16))
sft_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
peft_config = LoraConfig(r=8, lora_alpha=16, bias='none', task_type='CAUSAL_LM',
                         target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'])
peft_model = get_peft_model(sft_model, peft_config)
with torch.no_grad():
  for n, p in peft_model.named_parameters():
    if 'lora_B' in n:
      p.fill_(0.01)  # stand-in for the trained weights left in memory by trainer.train()

# ---- the notebook's cell [21], unchanged except for the generation settings that don't matter here
generator = pipeline('text-generation', model=sft_model, tokenizer=sft_tokenizer,
                     pad_token_id=sft_tokenizer.eos_token_id, max_new_tokens=256,
                     do_sample=True, temperature=0.6, top_p=0.9)
pipeline.model = PeftModel.from_pretrained(sft_model, adapter_path)
# ----

key = 'model.layers.0.self_attn.q_proj.lora_B.weight'
ckpt = load_file(f'{adapter_path}/adapter_model.safetensors')['base_model.model.' + key].float().cpu()
held = dict(generator.model.named_parameters())
held_keys = [k for k in held if k.endswith('layers.0.self_attn.q_proj.lora_B.default.weight')]
w = held[held_keys[0]].detach().float().cpu()
print('generator.model is sft_model:', generator.model is sft_model)
print('type(generator.model):', type(generator.model).__name__)
print('pipeline.model is a', type(pipeline.model).__name__, '(an attribute on the function)')
print('LoRA copies of layer 0 q_proj in generator.model:', held_keys)
print('generator holds the stand-in (all 0.01):', torch.all(w == 0.01).item())
print('generator holds the checkpoint weights:', torch.equal(w, ckpt))
print('checkpoint weights are all 0.01:', torch.all(ckpt == 0.01).item())
