# Generate HW06 answers the way the notebook does (4-bit model, same pipeline and sampling,
# same few-shot draws), for a model the notebook never evaluates on its own: the base model
# with no fine-tuning, or a LoRA checkpoint loaded from disk. Unlike the notebook it also keeps
# the raw GSM8K responses, so truncation and extraction can be checked afterwards.
# The few-shot examples come from random.seed(42) followed by the 7,473 random.sample calls of
# the training-set formatting loop, exactly as in the notebook, so the test prompts match ref runs.
# usage: HF_HUB_OFFLINE=1 python docs/tools/hw06_gen.py --out <file.json> [--adapter <ckpt dir>]
#        [--seed 0] [--max-new-tokens 256] [--greedy] [--n-shot 1] [--skip-gsm8k] [--skip-ailuminate]
import argparse
import csv
import json
import random
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--adapter', default=None)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max-new-tokens', type=int, default=256)
ap.add_argument('--greedy', action='store_true')
ap.add_argument('--n-shot', type=int, default=1)
ap.add_argument('--skip-gsm8k', action='store_true')
ap.add_argument('--skip-ailuminate', action='store_true')
ap.add_argument('--data', default='HW06/data')
args = ap.parse_args()

NAME = 'meta-llama/Llama-3.2-1B-Instruct'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                         bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(NAME, quantization_config=bnb, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(NAME)
tok.add_special_tokens({'pad_token': '[PAD]'})
if args.adapter:
  model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

gen_kwargs = dict(do_sample=False) if args.greedy else dict(do_sample=True, temperature=0.6, top_p=0.9)
generator = pipeline('text-generation', model=model, tokenizer=tok, pad_token_id=tok.eos_token_id,
                     max_new_tokens=args.max_new_tokens, **gen_kwargs)


def load_jsonlines(path):
  return [json.loads(line) for line in open(path)]


def nshot_chats(nshot_data, n, question):  # the notebook's nshot_chats in 'test' mode
  chats = []
  for qna in random.sample(nshot_data, n):
    chats.append({'role': 'user', 'content': f'Q: {qna["question"]}'})
    chats.append({'role': 'assistant', 'content': f'A: {qna["answer"]}'})
  chats.append({'role': 'user', 'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.'})
  return chats


def extract_ans_from_response(answer):  # the notebook's version, unchanged
  answer = answer.split('####')[-1].strip()
  for remove_char in [',', '$', '%', 'g']:
    answer = answer.replace(remove_char, '')
  return answer


def respond(chats):
  out = generator(chats)[0]['generated_text'][-1]['content']
  return out


train = load_jsonlines(f'{args.data}/gsm8k_train.jsonl')
random.seed(42)
for _ in train:  # the training-set formatting loop draws one few-shot example per question
  random.sample(train, args.n_shot)
torch.manual_seed(args.seed)

result = {'args': vars(args), 'gsm8k_public': [], 'ailuminate': []}
t0 = time.time()
if not args.skip_gsm8k:
  correct = 0
  for qna in tqdm(load_jsonlines(f'{args.data}/gsm8k_test_public.jsonl'), desc='gsm8k public'):
    resp = respond(nshot_chats(train, args.n_shot, qna['question']))
    n_new = len(tok(resp, add_special_tokens=False)['input_ids'])
    pred, true = extract_ans_from_response(resp), extract_ans_from_response(qna['answer'])
    correct += pred == true
    result['gsm8k_public'].append({'response': resp, 'pred': pred, 'true': true, 'n_tokens': n_new})
  result['gsm8k_public_acc'] = correct / len(result['gsm8k_public'])
  print(f'GSM8K public accuracy: {result["gsm8k_public_acc"]:.3f}')
if not args.skip_ailuminate:
  rows = list(csv.DictReader(open(f'{args.data}/ailuminate_test.csv')))
  for row in tqdm(rows, desc='ailuminate'):
    resp = respond([{'role': 'user', 'content': row['prompt_text']}])
    result['ailuminate'].append({'id': row['release_prompt_id'], 'hazard': row['hazard'],
                                 'persona': row['persona'], 'response': resp})
result['seconds'] = time.time() - t0
json.dump(result, open(args.out, 'w'), ensure_ascii=False, indent=1)
print('wrote', args.out, f'({result["seconds"]:.0f} s)')
