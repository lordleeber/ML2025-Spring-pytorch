# Where does HW06's training non-determinism come from? Repeat the same forward/backward on one
# fixed batch N times and check whether the loss and LoRA grads are bit-identical, for the
# notebook's 4-bit NF4 model vs a plain bf16 model (same LoRA config), with
# torch.use_deterministic_algorithms(True). Then repeat a 4-bit forward alone (no LoRA, no grad).
# usage: HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw06_nondet_probe.py
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.use_deterministic_algorithms(True)
NAME = 'meta-llama/Llama-3.2-1B-Instruct'
N = 4
tok = AutoTokenizer.from_pretrained(NAME)
q = json.loads(open('HW06/data/gsm8k_train.jsonl').readline())
text = tok.apply_chat_template([{'role': 'user', 'content': q['question']},
                                {'role': 'assistant', 'content': q['answer']}], tokenize=False)
ids = tok(text, return_tensors='pt').input_ids.cuda()


def load(kind):
  if kind == '4bit':
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(NAME, quantization_config=cfg, low_cpu_mem_usage=True)
  return AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16).cuda()


for kind in ['4bit', 'bf16']:
  base = load(kind)
  with torch.no_grad():
    outs = [base(ids).logits for _ in range(N)]
  print(f'{kind} forward only (no LoRA): logits identical over {N} runs:',
        all(torch.equal(outs[0], o) for o in outs[1:]))
  torch.manual_seed(1126)
  m = get_peft_model(base, LoraConfig(
      r=8, lora_alpha=16, bias='none', task_type='CAUSAL_LM',
      target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']))
  # B starts at 0 (then A gets no grad); give B small values so every LoRA grad is exercised
  with torch.no_grad():
    for n, p in m.named_parameters():
      if 'lora_B' in n:
        p.normal_(0, 1e-3)
  losses, grads = [], []
  for _ in range(N):
    m.zero_grad()
    loss = m(ids, labels=ids).loss
    loss.backward()
    losses.append(loss.detach().clone())
    grads.append([p.grad.detach().clone() for p in m.parameters() if p.requires_grad])
  print(f'{kind} + LoRA: losses {[f"{l.item():.10f}" for l in losses]}')
  print(f'{kind} + LoRA: loss identical:', all(torch.equal(losses[0], l) for l in losses[1:]),
        '| all grads identical:',
        all(torch.equal(g0, g) for gs in grads[1:] for g0, g in zip(grads[0], gs)))
  del m, base
  torch.cuda.empty_cache()
